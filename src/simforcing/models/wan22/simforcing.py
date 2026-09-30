from __future__ import annotations
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from simforcing.utils.logging_config import get_logger
from .base import VideoActionModel
from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .helpers.gradient import gradient_checkpoint_forward
from .mot import MoT
from .wan_video_dit import modulate

logger = get_logger(__name__)


class _SimConditionMoT(MoT):
    """MoT with per-sample gated, per-block simulation-latent conditioning."""

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
            sim_latent = context_payload.get("sim_latent")
            if sim_latent is not None:
                sim_feat = block.sim_cond_conv(sim_latent)
                b, d, f, h, w = sim_feat.shape
                sim_tokens = (
                    sim_feat.permute(0, 2, 3, 4, 1)
                    .reshape(b, f * h * w, d)
                    .contiguous()
                )
                if sim_tokens.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"Sim/video token count mismatch in gated additive block: sim={sim_tokens.shape[1]}, video={x.shape[1]}"
                    )
                cond_gate = context_payload.get("sim_cond_gate")
                if cond_gate is not None:
                    sim_tokens = sim_tokens * cond_gate.to(
                        device=sim_tokens.device, dtype=sim_tokens.dtype
                    ).view(-1, 1, 1)
                x = x + sim_tokens
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        return block.gate(x, gate_mlp, block.ffn(mlp_input))


class _TrainableStudent(nn.Module):
    """Expose the MoT (whose blocks hold the per-block sim convs) to the optimizer."""

    def __init__(self, mot: nn.Module):
        super().__init__()
        self.mot = mot


class SimForcing(VideoActionModel):
    """Action-conditioned sim-to-real video model with adjacent-delta distillation and optional simulation conditioning."""

    action_dropout_prob: float = 0.1
    first_frame_dropout_prob: float = 0.0
    loss_lambda_delta: float = 1.0
    loss_lambda_real: float = 1.0
    loss_lambda_syn: float = 1.0
    sim_cond_init_scale: float = 0.0
    sim_latent_noise_prob: float = 0.4
    sim_latent_noise_alpha_range: tuple[float, float] = (0.5, 0.9)
    sim_latent_noise_sigma_range: tuple[float, float] = (0.1, 0.3)
    sim_cond_on_syn_branch: bool = False
    loss_lambda_diffusion_real: Optional[float] = None
    loss_lambda_diffusion_syn: Optional[float] = None
    loss_lambda_delta_real: Optional[float] = None
    loss_lambda_delta_syn: Optional[float] = None
    sim_cond_prob: float = 0.5
    delta_sigma_weight_mode: str = "fm"
    sim_cond_cfg_scale: float = 1.0
    _DELTA_SIGMA_WEIGHT_MODES = ("fm", "low_sigma", "uniform")
    delta_lambda_cond_on: float = 1.0
    sim_cond_drift_prob: float = 0.0
    sim_cond_drift_scale_range: tuple[float, float] = (0.02, 0.08)
    delta_spatial_pool: int = 1
    delta_cosine: bool = False

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        delta_lambda_cond_on = float(
            kwargs.pop("delta_lambda_cond_on", cls.delta_lambda_cond_on)
        )
        sim_cond_drift_prob = float(
            kwargs.pop("sim_cond_drift_prob", cls.sim_cond_drift_prob)
        )
        if not 0.0 <= sim_cond_drift_prob <= 1.0:
            raise ValueError(
                f"`sim_cond_drift_prob` must be in [0, 1], got {sim_cond_drift_prob}."
            )
        drift_range = kwargs.pop(
            "sim_cond_drift_scale_range", cls.sim_cond_drift_scale_range
        )
        drift_range = cls._validate_range("sim_cond_drift_scale_range", drift_range)
        if drift_range[0] < 0.0:
            raise ValueError("`sim_cond_drift_scale_range` must be non-negative.")
        delta_spatial_pool = int(
            kwargs.pop("delta_spatial_pool", cls.delta_spatial_pool)
        )
        if delta_spatial_pool < 1:
            raise ValueError(
                f"`delta_spatial_pool` must be >= 1, got {delta_spatial_pool}."
            )
        delta_cosine = bool(kwargs.pop("delta_cosine", cls.delta_cosine))
        sim_cond_prob = float(kwargs.pop("sim_cond_prob", cls.sim_cond_prob))
        if not 0.0 <= sim_cond_prob <= 1.0:
            raise ValueError(f"`sim_cond_prob` must be in [0, 1], got {sim_cond_prob}.")
        delta_sigma_weight_mode = str(
            kwargs.pop("delta_sigma_weight_mode", cls.delta_sigma_weight_mode)
        )
        if delta_sigma_weight_mode not in cls._DELTA_SIGMA_WEIGHT_MODES:
            raise ValueError(
                f"`delta_sigma_weight_mode` must be one of {cls._DELTA_SIGMA_WEIGHT_MODES}, got {delta_sigma_weight_mode!r}."
            )
        sim_cond_cfg_scale = float(
            kwargs.pop("sim_cond_cfg_scale", cls.sim_cond_cfg_scale)
        )
        build_frozen_teacher = bool(kwargs.pop("build_frozen_teacher", True))
        frozen_video_expert_checkpoint = kwargs.pop(
            "frozen_video_expert_checkpoint", None
        )
        if build_frozen_teacher and frozen_video_expert_checkpoint is None:
            raise ValueError(
                "SimForcing requires `frozen_video_expert_checkpoint`: a ActionConditionedModel `.pt` whose `payload['mot']` contains both `mixtures.video.*` and `mixtures.action.*` state-dicts. Without it the frozen sim teacher has no act_cond weights and `delta_syn` would be meaningless. Pass `build_frozen_teacher=false` for an inference-only build."
            )
        frozen_action_dit_config = kwargs.pop("frozen_action_dit_config", None)
        delta_dropout_prob = float(kwargs.pop("delta_dropout_prob", 0.0))
        loss_lambda_delta = float(kwargs.pop("loss_lambda_delta", 1.0))

        def _opt_lambda(key: str) -> Optional[float]:
            value = kwargs.pop(key, None)
            return None if value is None else float(value)

        loss_lambda_diffusion_real = _opt_lambda("loss_lambda_diffusion_real")
        loss_lambda_diffusion_syn = _opt_lambda("loss_lambda_diffusion_syn")
        loss_lambda_delta_real = _opt_lambda("loss_lambda_delta_real")
        loss_lambda_delta_syn = _opt_lambda("loss_lambda_delta_syn")
        sim_cond_on_syn_branch = bool(
            kwargs.pop("sim_cond_on_syn_branch", cls.sim_cond_on_syn_branch)
        )
        loader_kwargs = {
            "model_id": kwargs.get("model_id"),
            "tokenizer_model_id": kwargs.get("tokenizer_model_id"),
            "tokenizer_max_len": int(kwargs.get("tokenizer_max_len", 512)),
            "redirect_common_files": bool(kwargs.get("redirect_common_files", True)),
        }
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for SimForcing."
            )
        if (
            loader_kwargs["model_id"] is None
            or loader_kwargs["tokenizer_model_id"] is None
        ):
            raise ValueError(
                "`model_id` and `tokenizer_model_id` are required to build the frozen act_cond sim teacher."
            )
        action_dit_config = kwargs.get("action_dit_config", None)
        if not isinstance(action_dit_config, dict):
            raise ValueError(
                "`action_dit_config` must be provided as dict for SimForcing (needed to instantiate the frozen action expert)."
            )
        if frozen_action_dit_config is None:
            frozen_action_dit_config = action_dit_config
        sim_cond_init_scale = float(
            kwargs.pop("sim_cond_init_scale", cls.sim_cond_init_scale)
        )
        sim_latent_noise_prob = float(kwargs.pop("sim_latent_noise_prob", 0.4))
        alpha_range = tuple(
            (float(v) for v in kwargs.pop("sim_latent_noise_alpha_range", (0.5, 0.9)))
        )
        sigma_range = tuple(
            (float(v) for v in kwargs.pop("sim_latent_noise_sigma_range", (0.1, 0.3)))
        )
        loss_lambda_real = float(kwargs.pop("loss_lambda_real", cls.loss_lambda_real))
        loss_lambda_syn = float(kwargs.pop("loss_lambda_syn", cls.loss_lambda_syn))
        action_dropout_prob = float(
            kwargs.pop("action_dropout_prob", cls.action_dropout_prob)
        )
        if not 0.0 <= action_dropout_prob <= 1.0:
            raise ValueError(
                f"`action_dropout_prob` must be in [0, 1], got {action_dropout_prob}"
            )
        first_frame_dropout_prob = float(
            kwargs.pop("first_frame_dropout_prob", cls.first_frame_dropout_prob)
        )
        if not 0.0 <= first_frame_dropout_prob <= 1.0:
            raise ValueError(
                f"`first_frame_dropout_prob` must be in [0, 1], got {first_frame_dropout_prob}"
            )
        existing_lambda_action = float(kwargs.get("loss_lambda_action", 0.0))
        if existing_lambda_action != 0.0:
            logger.warning(
                "SimForcing ignores `loss_lambda_action=%s`; forcing 0.",
                existing_lambda_action,
            )
        kwargs["loss_lambda_action"] = 0.0
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for SimForcing."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "SimForcing requires `video_dit_config['action_conditioned']=false`; action conditioning is delivered via MoT mixed attention."
            )
        model = VideoActionModel.from_wan22_pretrained.__func__(cls, **kwargs)
        model.loss_lambda_real = loss_lambda_real
        model.loss_lambda_syn = loss_lambda_syn
        model.action_dropout_prob = action_dropout_prob
        model.first_frame_dropout_prob = first_frame_dropout_prob
        model.delta_dropout_prob = 0.0
        model._set_sim_latent_noise_config(
            sim_latent_noise_prob, alpha_range, sigma_range
        )
        model.sim_cond_init_scale = sim_cond_init_scale
        model._init_sim_latent_conditioner()
        model.delta_dropout_prob = delta_dropout_prob
        model.loss_lambda_delta = loss_lambda_delta
        model.loss_lambda_diffusion_real = loss_lambda_diffusion_real
        model.loss_lambda_diffusion_syn = loss_lambda_diffusion_syn
        model.loss_lambda_delta_real = loss_lambda_delta_real
        model.loss_lambda_delta_syn = loss_lambda_delta_syn
        model.sim_cond_on_syn_branch = sim_cond_on_syn_branch
        if build_frozen_teacher:
            model._init_frozen_act_cond_branch(
                video_dit_config=video_dit_config,
                action_dit_config=frozen_action_dit_config,
                loader_kwargs=loader_kwargs,
                ckpt_path=str(frozen_video_expert_checkpoint),
            )
        else:
            logger.warning(
                "build_frozen_teacher=false: the frozen act_cond sim teacher was NOT constructed. `training_loss` is therefore UNAVAILABLE (inference-only model); GPU memory is roughly halved."
            )
        model.sim_cond_prob = sim_cond_prob
        model.delta_sigma_weight_mode = delta_sigma_weight_mode
        model.sim_cond_cfg_scale = sim_cond_cfg_scale
        model.delta_lambda_cond_on = delta_lambda_cond_on
        model.sim_cond_drift_prob = sim_cond_drift_prob
        model.sim_cond_drift_scale_range = drift_range
        model.delta_spatial_pool = delta_spatial_pool
        model.delta_cosine = delta_cosine
        return model

    def _run_blocks(self, pre: dict, use_grad_checkpoint: bool) -> torch.Tensor:
        ve = self.video_expert
        x_tokens = pre["tokens"]
        context_emb = pre["context"]
        t_mod = pre["t_mod"]
        freqs = pre["freqs"]
        context_attn_mask = pre["context_mask"]
        self_attn_mask = ve.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        )
        for block in ve.blocks:
            if use_grad_checkpoint and ve.use_gradient_checkpointing and self.training:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    ve.use_gradient_checkpointing,
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
        return ve.post_dit(x_tokens, pre)

    def build_paired_inputs(self, sample, tiled: bool = False):
        if "video_syn" not in sample or "video_real" not in sample:
            raise ValueError(
                "SimForcing expects `sample['video_syn']` and `sample['video_real']` (use BridgeV2PairedSynRealDataset)."
            )
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "SimForcing training requires precomputed `context`/`context_mask`."
            )
        video_syn = sample["video_syn"]
        video_real = sample["video_real"]
        if video_syn.shape != video_real.shape:
            raise ValueError(
                f"Sim/real videos must share shape, got sim={tuple(video_syn.shape)} vs real={tuple(video_real.shape)}"
            )
        if video_syn.ndim != 5 or video_syn.shape[1] != 3:
            raise ValueError(
                f"`video_*` must be 5D [B,3,T,H,W], got {tuple(video_syn.shape)}"
            )
        batch_size, _, num_frames, height, width = video_syn.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1, got T={num_frames}")
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None and image_is_pad.ndim == 2:
            if image_is_pad.shape != (batch_size, num_frames):
                raise ValueError(
                    f"`image_is_pad` shape mismatch: got {tuple(image_is_pad.shape)} vs ({batch_size}, {num_frames})"
                )
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        elif image_is_pad is not None:
            raise ValueError(
                f"`image_is_pad` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
            )
        video_syn_dev = video_syn.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        video_real_dev = video_real.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        latents_syn = self._encode_video_latents(video_syn_dev, tiled=tiled)
        latents_real = self._encode_video_latents(video_real_dev, tiled=tiled)
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        first_frame_syn = latents_syn[:, :, 0:1] if fuse_flag else None
        first_frame_real = latents_real[:, :, 0:1] if fuse_flag else None
        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        context_syn = sample.get("context_syn", None)
        context_mask_syn = sample.get("context_mask_syn", None)
        if (context_syn is None) != (context_mask_syn is None):
            raise ValueError(
                "`context_syn` and `context_mask_syn` must be provided together."
            )
        if context_syn is not None:
            if context_syn.ndim != 3 or context_mask_syn.ndim != 2:
                raise ValueError(
                    f"`context_syn/context_mask_syn` must be [B,L,D]/[B,L], got {tuple(context_syn.shape)} and {tuple(context_mask_syn.shape)}"
                )
            context_syn = context_syn.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask_syn = context_mask_syn.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if self.proprio_encoder is not None:
            proprio = sample.get("proprio", None)
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            if proprio.ndim != 3 or proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio0 = proprio[:, 0, :].to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio0
            )
            if context_syn is not None:
                context_syn, context_mask_syn = self._append_proprio_to_context(
                    context=context_syn, context_mask=context_mask_syn, proprio=proprio0
                )
        inputs = {
            "latents_syn": latents_syn,
            "latents_real": latents_real,
            "first_frame_syn": first_frame_syn,
            "first_frame_real": first_frame_real,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "context": context,
            "context_mask": context_mask,
            "context_syn": context_syn,
            "context_mask_syn": context_mask_syn,
            "image_is_pad": image_is_pad,
        }
        if "action" not in sample:
            raise ValueError(
                "SimForcing requires `sample['action']` (the GT action sequence used to drive both the frozen sim teacher and the trainable student). Use BridgeV2PairedSynRealDataset or any dataset that emits a shared `action` field."
            )
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        batch_size = inputs["latents_real"].shape[0]
        if action.shape[0] != batch_size:
            raise ValueError(
                f"`sample['action']` batch size mismatch: action B={action.shape[0]} vs video B={batch_size}."
            )
        action_dev = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape != action.shape[:2]:
                raise ValueError(
                    f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs {tuple(action.shape[:2])}"
                )
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        inputs["action"] = action_dev
        inputs["action_is_pad"] = action_is_pad
        return inputs

    def training_loss(self, sample, tiled: bool = False):
        if not hasattr(self, "frozen_mot"):
            raise RuntimeError(
                "SimForcing.training_loss requires the frozen act_cond sim teacher (the delta term is always on), but the model was built with `build_frozen_teacher=false`."
            )
        inputs = self.build_paired_inputs(sample, tiled=tiled)
        latents_syn = inputs["latents_syn"]
        latents_real = inputs["latents_real"]
        first_frame_syn = inputs["first_frame_syn"]
        first_frame_real = inputs["first_frame_real"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        image_is_pad = inputs["image_is_pad"]
        action = inputs["action"]
        batch_size = latents_real.shape[0]
        device = latents_real.device
        noise = torch.randn_like(latents_real)
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=latents_real.dtype
        )
        noisy_syn = self.train_video_scheduler.add_noise(latents_syn, noise, timestep)
        noisy_real = self.train_video_scheduler.add_noise(latents_real, noise, timestep)
        target_syn = self.train_video_scheduler.training_target(
            latents_syn, noise, timestep
        )
        target_real = self.train_video_scheduler.training_target(
            latents_real, noise, timestep
        )
        if first_frame_syn is not None:
            noisy_syn = noisy_syn.clone()
            noisy_syn[:, :, 0:1] = first_frame_syn
        has_ff_syn = first_frame_syn is not None
        has_ff_real = first_frame_real is not None
        in_train = self._in_training_phase()
        if (
            (has_ff_syn or has_ff_real)
            and in_train
            and (self.first_frame_dropout_prob > 0.0)
        ):
            ff_drop = torch.rand((batch_size,), device=device) < float(
                self.first_frame_dropout_prob
            )
        else:
            ff_drop = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        if has_ff_real:
            keep = (~ff_drop).view(batch_size, 1, 1, 1, 1)
            noisy_real = noisy_real.clone()
            noisy_real[:, :, 0:1] = torch.where(
                keep, first_frame_real, noisy_real[:, :, 0:1]
            )
        if has_ff_syn:
            keep = (~ff_drop).view(batch_size, 1, 1, 1, 1)
            if bool(ff_drop.any()):
                noisy_syn_ff = self.train_video_scheduler.add_noise(
                    latents_syn[:, :, 0:1], noise[:, :, 0:1], timestep
                )
                noisy_syn_student = noisy_syn.clone()
                noisy_syn_student[:, :, 0:1] = torch.where(
                    keep, noisy_syn[:, :, 0:1], noisy_syn_ff
                )
            else:
                noisy_syn_student = noisy_syn
        else:
            noisy_syn_student = noisy_syn
        if in_train and self.action_dropout_prob > 0.0:
            action_drop = torch.rand((batch_size,), device=action.device) < float(
                self.action_dropout_prob
            )
        else:
            action_drop = torch.zeros(
                (batch_size,), dtype=torch.bool, device=action.device
            )
        action_for_cond = action * (~action_drop).to(action.dtype).view(
            batch_size, 1, 1
        )
        prob = float(getattr(self, "sim_cond_prob", 0.5))
        if in_train:
            cond_on = torch.rand((batch_size,), device=device) < prob
        else:
            cond_on = torch.ones((batch_size,), dtype=torch.bool, device=device)
        sim_cond_gate = cond_on.to(dtype=latents_real.dtype)
        with torch.no_grad():
            v_syn_t = self._run_frozen_act_cond_v_syn(
                latents=noisy_syn,
                timestep_video=timestep,
                action=action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            x0_hat_syn_t = self._x0_hat_from_velocity(
                noisy=noisy_syn,
                v=v_syn_t,
                timestep=timestep,
                num_train_timesteps=int(self.train_video_scheduler.num_train_timesteps),
            )
            delta_syn = self._delta_from_x0(x0_hat_syn_t)
        delta_syn = delta_syn.detach()
        sim_latent_cond = self._apply_rollout_drift(
            self._maybe_corrupt_sim_latent(latents_syn)
        )
        if self.sim_cond_on_syn_branch:
            v_syn_s = self._forward_student_with_sim_condition(
                latents=noisy_syn_student,
                timestep_video=timestep,
                action=action_for_cond,
                sim_latent=sim_latent_cond,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                sim_cond_gate=sim_cond_gate,
            )
        else:
            v_syn_s = self._forward_student_with_action(
                latents=noisy_syn_student,
                timestep_video=timestep,
                action=action_for_cond,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
        x0_hat_syn_s = self._x0_hat_from_velocity(
            noisy=noisy_syn_student,
            v=v_syn_s,
            timestep=timestep,
            num_train_timesteps=int(self.train_video_scheduler.num_train_timesteps),
        )
        delta_syn_s = self._delta_from_x0(x0_hat_syn_s)
        v_real_s = self._forward_student_with_sim_condition(
            latents=noisy_real,
            timestep_video=timestep,
            action=action_for_cond,
            sim_latent=sim_latent_cond,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            sim_cond_gate=sim_cond_gate,
        )
        x0_hat_real_s = self._x0_hat_from_velocity(
            noisy=noisy_real,
            v=v_real_s,
            timestep=timestep,
            num_train_timesteps=int(self.train_video_scheduler.num_train_timesteps),
        )
        delta_real_s = self._delta_from_x0(x0_hat_real_s)

        def slice_loss(pred, target, has_ff):
            if not has_ff:
                return (True, pred, target, image_is_pad)
            if not bool(ff_drop.any()):
                return (False, pred[:, :, 1:], target[:, :, 1:], image_is_pad)
            pad = image_is_pad
            temporal_factor = int(self.vae.temporal_downsample_factor)
            pixel_t = (target.shape[2] - 1) * temporal_factor + 1
            if pad is None:
                pad = torch.zeros(
                    (batch_size, pixel_t), dtype=torch.bool, device=target.device
                )
            else:
                pad = pad.clone()
            pad[:, 0] = pad[:, 0] | ~ff_drop
            return (True, pred, target, pad)

        inc_real, pred_real, tgt_real, pad_real = slice_loss(
            v_real_s, target_real, has_ff_real
        )
        inc_syn, pred_syn, tgt_syn, pad_syn = slice_loss(
            v_syn_s, target_syn, has_ff_syn
        )
        real_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_real,
            target_video=tgt_real,
            image_is_pad=pad_real,
            include_initial_video_step=inc_real,
        )
        syn_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_syn,
            target_video=tgt_syn,
            image_is_pad=pad_syn,
            include_initial_video_step=inc_syn,
        )
        weight = self.train_video_scheduler.training_weight(timestep).to(
            real_per_sample.device, dtype=real_per_sample.dtype
        )
        loss_video_real = (real_per_sample * weight).mean()
        loss_video_syn = (syn_per_sample * weight).mean()
        loss_video = 0.5 * (
            self.lambda_diffusion_real * loss_video_real
            + self.lambda_diffusion_syn * loss_video_syn
        )
        delta_target = delta_syn.float()
        delta_real_per_sample = self._delta_loss_per_sample(delta_real_s, delta_target)
        delta_syn_per_sample = self._delta_loss_per_sample(delta_syn_s, delta_target)
        delta_weight = self._delta_sigma_weight(timestep, weight).to(
            delta_real_per_sample.device, dtype=delta_real_per_sample.dtype
        )
        w_cond_on = float(getattr(self, "delta_lambda_cond_on", 1.0))
        delta_scale = torch.where(
            cond_on,
            torch.full_like(delta_real_per_sample, w_cond_on),
            torch.ones_like(delta_real_per_sample),
        )
        if in_train and self.delta_dropout_prob > 0.0:
            keep = (
                torch.rand(batch_size, device=device) >= self.delta_dropout_prob
            ).to(dtype=delta_real_per_sample.dtype)
            delta_scale = delta_scale * keep
        loss_delta_real = (delta_real_per_sample * delta_weight * delta_scale).mean()
        loss_delta_syn = (delta_syn_per_sample * delta_weight * delta_scale).mean()
        loss_delta = 0.5 * (
            self.lambda_delta_real * loss_delta_real
            + self.lambda_delta_syn * loss_delta_syn
        )
        loss_total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_delta * loss_delta
        )
        return (
            loss_total,
            {
                "loss_video": self.loss_lambda_video
                * float(loss_video.detach().item()),
                "loss_video_real": float(loss_video_real.detach().item()),
                "loss_video_syn": float(loss_video_syn.detach().item()),
                "loss_delta": self.loss_lambda_delta
                * float(loss_delta.detach().item()),
                "loss_delta_real": float(loss_delta_real.detach().item()),
                "loss_delta_syn": float(loss_delta_syn.detach().item()),
                "loss_action": 0.0,
                "frac_cond_on": float(sim_cond_gate.float().mean().detach().item()),
            },
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            self.mot.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing `mot` or `dit`: {path}")
        if self.proprio_encoder is not None and "proprio_encoder" in payload:
            self.proprio_encoder.load_state_dict(
                payload["proprio_encoder"], strict=True
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        raise NotImplementedError("SimForcing does not predict actions.")

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError("SimForcing does not run joint denoising.")

    @torch.no_grad()
    def infer_video(
        self,
        prompt: Optional[str],
        input_image_syn: Optional[torch.Tensor],
        input_image_real: torch.Tensor,
        sim_video: Optional[torch.Tensor] = None,
        sim_latents: Optional[torch.Tensor] = None,
        num_video_frames: int = 0,
        action: torch.Tensor = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action_cfg_scale: float = 1.0,
        sim_cond_cfg_scale: Optional[float] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
    ) -> dict[str, Any]:
        scale = (
            float(self.sim_cond_cfg_scale)
            if sim_cond_cfg_scale is None
            else float(sim_cond_cfg_scale)
        )
        if scale == 1.0:
            return self._infer_video_conditioned(
                prompt=prompt,
                input_image_syn=input_image_syn,
                input_image_real=input_image_real,
                sim_video=sim_video,
                sim_latents=sim_latents,
                num_video_frames=num_video_frames,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                action_cfg_scale=action_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                drop_first_frame=drop_first_frame,
            )
        del input_image_syn
        self.eval()
        if input_image_real.ndim == 3:
            input_image_real = input_image_real.unsqueeze(0)
        if input_image_real.ndim != 4 or input_image_real.shape[:2] != (1, 3):
            raise ValueError(
                f"`input_image_real` must be [1,3,H,W] or [3,H,W], got {tuple(input_image_real.shape)}"
            )
        _, _, height, width = input_image_real.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError(
                "Input dimensions/frame count must satisfy Wan2.2 resize rules."
            )
        if sim_latents is None and sim_video is None:
            raise ValueError(
                "Provide either `sim_video` or `sim_latents` as the sim condition."
            )
        if sim_latents is not None:
            if sim_latents.ndim == 4:
                sim_latents = sim_latents.unsqueeze(0)
            if sim_latents.ndim != 5 or sim_latents.shape[0] != 1:
                raise ValueError(
                    f"`sim_latents` must be [1, z_dim, T, H, W], got {tuple(sim_latents.shape)}"
                )
        else:
            if sim_video.ndim == 4:
                sim_video = sim_video.unsqueeze(0)
            if (
                sim_video.ndim != 5
                or sim_video.shape[0] != 1
                or sim_video.shape[1] != 3
                or (sim_video.shape[2] != num_video_frames)
            ):
                raise ValueError(
                    f"`sim_video` must be [1,3,{num_video_frames},H,W], got {tuple(sim_video.shape)}"
                )
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"`action` must be [1,T,A] or [T,A], got {tuple(action.shape)}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        if sim_latents is not None:
            clean_sim = sim_latents.to(device=self.device, dtype=self.torch_dtype)
        else:
            sim_video = sim_video.to(device=self.device, dtype=self.torch_dtype)
            clean_sim = self._encode_video_latents(sim_video, tiled=tiled)
        if clean_sim.shape[2] != latent_t:
            raise RuntimeError(
                f"Sim latent has T={clean_sim.shape[2]}, expected {latent_t}."
            )
        input_image_real = input_image_real.to(
            device=self.device, dtype=self.torch_dtype
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        first_real = (
            None
            if drop_first_frame
            else self._encode_input_image_latents_tensor(
                input_image=input_image_real, tiled=tiled
            )
        )
        gen = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        real_latent = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=gen,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if fuse_flag and first_real is not None:
            real_latent[:, :, 0:1] = first_real.clone()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=real_latent.dtype,
            shift_override=sigma_shift,
        )
        if len(timesteps) == 0:
            raise ValueError("Inference schedule is empty.")
        gate_on = torch.ones((1,), device=self.device, dtype=self.torch_dtype)
        gate_off = torch.zeros((1,), device=self.device, dtype=self.torch_dtype)
        zero_action = torch.zeros_like(action)
        do_action_cfg = float(action_cfg_scale) != 1.0
        for step_t, step_d in zip(timesteps, deltas):
            t = step_t.view(1).to(device=self.device, dtype=real_latent.dtype)

            def _fwd(act, gate):
                return self._forward_student_with_sim_condition(
                    latents=real_latent,
                    timestep_video=t,
                    action=act,
                    sim_latent=clean_sim,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    sim_cond_gate=gate,
                )

            pred_ac = _fwd(action, gate_on)
            pred_a = _fwd(action, gate_off)
            pred_real = pred_a + scale * (pred_ac - pred_a)
            if do_action_cfg:
                pred_c = _fwd(zero_action, gate_on)
                pred_real = pred_real + (float(action_cfg_scale) - 1.0) * (
                    pred_ac - pred_c
                )
            real_latent = self.infer_video_scheduler.step(
                pred_real, step_d, real_latent
            )
            if fuse_flag and first_real is not None:
                real_latent[:, :, 0:1] = first_real.clone()
        return {"video": self._decode_latents(real_latent, tiled=tiled)}

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        return self.infer_video(*args, **kwargs)

    def _init_frozen_act_cond_branch(
        self,
        video_dit_config: dict,
        action_dit_config: dict,
        loader_kwargs: dict,
        ckpt_path: str,
    ) -> None:
        """Construct frozen ``(video_expert, action_expert, MoT)``.

        Loads weights for BOTH experts from the same ActionConditionedModel
        checkpoint at ``ckpt_path`` (top-level ``payload['mot']`` with
        both ``mixtures.video.*`` and ``mixtures.action.*`` keys).
        """
        components = load_wan22_ti2v_5b_components(
            device=str(self.device),
            torch_dtype=self.torch_dtype,
            model_id=loader_kwargs["model_id"],
            tokenizer_model_id=loader_kwargs["tokenizer_model_id"],
            tokenizer_max_len=loader_kwargs["tokenizer_max_len"],
            redirect_common_files=loader_kwargs["redirect_common_files"],
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=False,
            load_text_encoder=False,
        )
        frozen_video_expert = components.dit
        frozen_action_expert = ActionDiT(**action_dit_config).to(
            device=self.device, dtype=self.torch_dtype
        )
        if int(frozen_action_expert.num_heads) != int(frozen_video_expert.num_heads):
            raise ValueError(
                f"Frozen action_expert num_heads ({frozen_action_expert.num_heads}) != video_expert num_heads ({frozen_video_expert.num_heads})."
            )
        if int(frozen_action_expert.attn_head_dim) != int(
            frozen_video_expert.attn_head_dim
        ):
            raise ValueError(
                f"Frozen action_expert attn_head_dim ({frozen_action_expert.attn_head_dim}) != video_expert attn_head_dim ({frozen_video_expert.attn_head_dim})."
            )
        if int(len(frozen_action_expert.blocks)) != int(
            len(frozen_video_expert.blocks)
        ):
            raise ValueError(
                f"Frozen action_expert num_layers ({len(frozen_action_expert.blocks)}) != video_expert num_layers ({len(frozen_video_expert.blocks)})."
            )
        frozen_mot = MoT(
            mixtures={"video": frozen_video_expert, "action": frozen_action_expert},
            mot_checkpoint_mixed_attn=False,
        ).to(device=self.device, dtype=self.torch_dtype)
        self._load_frozen_act_cond_checkpoint(
            frozen_mot=frozen_mot, ckpt_path=ckpt_path
        )
        frozen_mot.eval()
        for p in frozen_mot.parameters():
            p.requires_grad_(False)
        for sub in (frozen_video_expert, frozen_action_expert):
            if hasattr(sub, "use_gradient_checkpointing"):
                sub.use_gradient_checkpointing = False
        self.frozen_mot = frozen_mot
        self.frozen_video_expert = frozen_video_expert
        self.frozen_action_expert = frozen_action_expert
        n_v = sum((p.numel() for p in frozen_video_expert.parameters()))
        n_a = sum((p.numel() for p in frozen_action_expert.parameters()))
        logger.info(
            "SimForcing: frozen act_cond teacher built (video=%.2fM, action=%.2fM, total=%.2fM params; requires_grad=False).",
            n_v / 1000000.0,
            n_a / 1000000.0,
            (n_v + n_a) / 1000000.0,
        )

    @staticmethod
    def _load_frozen_act_cond_checkpoint(frozen_mot: MoT, ckpt_path: str) -> None:
        """Load act_cond weights into ``frozen_mot``.

        Expected payload layout (same as ``VideoActionModel.save_checkpoint``):
            {"mot": <state_dict keyed under mixtures.video.* / mixtures.action.*>, ...}
        """
        logger.info(
            "Loading frozen act_cond teacher weights from checkpoint: %s", ckpt_path
        )
        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict) or "mot" not in payload:
            raise ValueError(
                f"`frozen_video_expert_checkpoint` must point to a VideoActionModel checkpoint with a top-level `mot` state-dict, got payload of type {type(payload)} at {ckpt_path}."
            )
        mot_sd = payload["mot"]
        has_video_keys = any((k.startswith("mixtures.video.") for k in mot_sd))
        has_action_keys = any((k.startswith("mixtures.action.") for k in mot_sd))
        if not has_video_keys:
            raise ValueError(
                f"Checkpoint `{ckpt_path}` has `mot` but no `mixtures.video.*` keys."
            )
        if not has_action_keys:
            raise ValueError(
                f"Checkpoint `{ckpt_path}` has `mot` but no `mixtures.action.*` keys; SimForcing requires an action-conditioned checkpoint."
            )
        missing, unexpected = frozen_mot.load_state_dict(mot_sd, strict=False)
        if missing:
            logger.warning(
                "Frozen MoT load: %d missing keys (showing up to 5): %s",
                len(missing),
                list(missing)[:5],
            )
        if unexpected:
            logger.warning(
                "Frozen MoT load: %d unexpected keys (showing up to 5): %s",
                len(unexpected),
                list(unexpected)[:5],
            )

    def train(self, mode: bool = True):
        """Keep the frozen branch permanently in eval / no-grad."""
        super().train(mode)
        if hasattr(self, "frozen_mot"):
            self.frozen_mot.eval()
            for p in self.frozen_mot.parameters():
                p.requires_grad_(False)
        return self

    @torch.no_grad()
    def _build_act_cond_attention_mask(
        self,
        video_expert: nn.Module,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Same rule as ``ActionConditionedModel._build_mot_attention_mask``,
        but parameterised by which video_expert provides the
        video-to-video mask (so it works for both trainable and frozen).
        """
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        mask[:video_seq_len, :video_seq_len] = video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        mask[:video_seq_len, video_seq_len:] = True
        mask[video_seq_len:, video_seq_len:] = True
        return mask

    def _run_frozen_act_cond_v_syn(
        self,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        """Run the frozen act_cond teacher and return ``v_syn``.

        Mirrors :meth:`ActionConditionedModel._forward_video_with_action_cond`'s
        action-conditioned path but on the frozen ``(video, action, MoT)``
        triple. The caller is responsible for wrapping in ``torch.no_grad()``.
        """
        if action is None:
            raise ValueError("_run_frozen_act_cond_v_syn requires a non-None `action`.")
        batch_size = latents.shape[0]
        video_pre = self.frozen_video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        timestep_action = torch.zeros(
            (batch_size,), dtype=action.dtype, device=action.device
        )
        action_pre = self.frozen_action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_act_cond_attention_mask(
            video_expert=self.frozen_video_expert,
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.frozen_mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return self.frozen_video_expert.post_dit(tokens_out["video"], video_pre)

    def _forward_student_with_action(
        self,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        action: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if action is None:
            return self._run_blocks(video_pre, use_grad_checkpoint=True)
        timestep_action = torch.zeros(
            (batch_size,), dtype=action.dtype, device=action.device
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_act_cond_attention_mask(
            video_expert=self.video_expert,
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return self.video_expert.post_dit(tokens_out["video"], video_pre)

    @staticmethod
    def _x0_hat_from_velocity(
        noisy: torch.Tensor,
        v: torch.Tensor,
        timestep: torch.Tensor,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        sigma = (timestep.to(dtype=torch.float32) / float(num_train_timesteps)).to(
            device=noisy.device, dtype=noisy.dtype
        )
        sigma = sigma.view(-1, *[1] * (noisy.ndim - 1))
        return noisy - sigma * v

    @staticmethod
    def _delta_from_x0(x0: torch.Tensor) -> torch.Tensor:
        """Return consecutive temporal differences with shape ``T_lat - 1``.

        ``delta[:, :, k - 1] = x0[:, :, k] - x0[:, :, k - 1]`` -- the same
        "adjacent delta" used by ``SimForcing``. It is
        applied consistently to the frozen teacher target and to both student
        branches.
        """
        if x0.ndim != 5:
            raise ValueError(f"Expected x0 shape [B,C,T,H,W], got {tuple(x0.shape)}")
        if x0.shape[2] < 2:
            raise ValueError(
                f"Adjacent delta regularization requires at least two latent steps, got T_lat={x0.shape[2]}."
            )
        return x0[:, :, 1:] - x0[:, :, :-1]

    @staticmethod
    def _validate_range(name: str, value) -> tuple[float, float]:
        lo, hi = (float(value[0]), float(value[1]))
        if lo > hi:
            raise ValueError(f"{name}: lower bound {lo} > upper bound {hi}.")
        return (lo, hi)

    def _set_sim_latent_noise_config(
        self,
        prob: float,
        alpha_range: tuple[float, float],
        sigma_range: tuple[float, float],
    ) -> None:
        """Validate and store the sim-latent noise augmentation settings."""
        prob = float(prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"sim_latent_noise_prob must be in [0, 1], got {prob}.")
        self.sim_latent_noise_prob = prob
        self.sim_latent_noise_alpha_range = self._validate_range(
            "sim_latent_noise_alpha_range", alpha_range
        )
        self.sim_latent_noise_sigma_range = self._validate_range(
            "sim_latent_noise_sigma_range", sigma_range
        )
        if self.sim_latent_noise_sigma_range[0] < 0.0:
            raise ValueError("sim_latent_noise_sigma_range must be non-negative.")
        logger.info(
            "Sim-latent noise augmentation: prob=%.3f, alpha=[%.3f, %.3f], sigma=[%.3f, %.3f].",
            self.sim_latent_noise_prob,
            *self.sim_latent_noise_alpha_range,
            *self.sim_latent_noise_sigma_range,
        )

    def _in_training_phase(self) -> bool:
        """Whether the model is currently in the training phase.

        NOTE: ``self.training`` alone is NOT reliable here. The trainer freezes
        everything except the DiT via ``_apply_dit_only_train_mode``, which
        calls ``model.eval()`` on the top-level module and then ``model.dit
        .train()``. So during training ``self.training is False`` while
        ``self.dit.training is True``. During ``evaluate()`` a plain
        ``model.eval()`` puts BOTH in eval mode. Hence checking the DiT flag
        (in addition to the top-level one) is what actually separates the two
        phases.
        """
        if self.training:
            return True
        dit = getattr(self, "dit", None)
        return bool(getattr(dit, "training", False))

    def _maybe_corrupt_sim_latent(self, sim_latent: torch.Tensor) -> torch.Tensor:
        """With ``sim_latent_noise_prob`` replace the GT sim latent by a mixture.

        z' = alpha * z + (1 - alpha) * N(0, sigma^2 I), with alpha and sigma
        drawn per batch element. Otherwise the clean GT is returned unchanged.
        Only active during training; inference always uses the clean GT.
        """
        prob = float(getattr(self, "sim_latent_noise_prob", 0.0))
        if not self._in_training_phase() or prob <= 0.0:
            return sim_latent
        batch_size = sim_latent.shape[0]
        device, dtype = (sim_latent.device, sim_latent.dtype)
        corrupt = torch.rand((batch_size,), device=device) < prob
        if not bool(corrupt.any()):
            return sim_latent
        alpha_lo, alpha_hi = self.sim_latent_noise_alpha_range
        sigma_lo, sigma_hi = self.sim_latent_noise_sigma_range
        alpha = torch.empty((batch_size,), device=device, dtype=dtype).uniform_(
            float(alpha_lo), float(alpha_hi)
        )
        sigma = torch.empty((batch_size,), device=device, dtype=dtype).uniform_(
            float(sigma_lo), float(sigma_hi)
        )
        shape = (batch_size,) + (1,) * (sim_latent.dim() - 1)
        alpha = alpha.view(shape)
        sigma = sigma.view(shape)
        eps = torch.randn_like(sim_latent)
        mask = corrupt.to(dtype).view(shape)
        return torch.where(
            mask.bool(), alpha * sim_latent + (1.0 - alpha) * sigma * eps, sim_latent
        )

    def _init_sim_latent_conditioner(self) -> None:
        video = self.video_expert
        patch_size = tuple((int(x) for x in video.patch_size))
        in_dim = int(video.in_dim)
        hidden_dim = int(video.hidden_dim)
        init_scale = float(getattr(self, "sim_cond_init_scale", 0.1))
        for block in video.blocks:
            conv = nn.Conv3d(
                in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size
            ).to(device=self.device, dtype=self.torch_dtype)
            with torch.no_grad():
                nn.init.kaiming_normal_(conv.weight)
                conv.weight.mul_(init_scale)
                if conv.bias is not None:
                    conv.bias.zero_()
            block.sim_cond_conv = conv
        mot_checkpointing = bool(getattr(self.mot, "mot_checkpoint_mixed_attn", False))
        self.mot = _SimConditionMoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=mot_checkpointing,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.dit = _TrainableStudent(self.mot).to(
            device=self.device, dtype=self.torch_dtype
        )

    def _forward_student_with_sim_condition(
        self,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        action: torch.Tensor,
        sim_latent: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        sim_cond_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action is None:
            raise ValueError("A tensor action is required for the MoT path.")
        batch_size = latents.shape[0]
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        sim_latent_detached = sim_latent.detach().to(
            device=self.device, dtype=self.torch_dtype
        )
        if sim_latent_detached.shape[2:] != tuple(latents.shape[2:]):
            raise ValueError(
                f"Sim latent grid mismatch with target latent grid: sim={tuple(sim_latent_detached.shape[2:])}, target={tuple(latents.shape[2:])}"
            )
        video_payload = {
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
            "sim_latent": sim_latent_detached,
        }
        if sim_cond_gate is not None:
            gate = sim_cond_gate.to(
                device=sim_latent_detached.device, dtype=self.torch_dtype
            ).reshape(-1)
            if gate.shape[0] != batch_size:
                raise ValueError(
                    f"`sim_cond_gate` must have batch size {batch_size}, got {tuple(gate.shape)}."
                )
            video_payload["sim_cond_gate"] = gate
        timestep_action = torch.zeros(
            (batch_size,), dtype=action.dtype, device=action.device
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_act_cond_attention_mask(
            video_expert=self.video_expert,
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": video_payload,
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return self.video_expert.post_dit(tokens_out["video"], video_pre)

    @torch.no_grad()
    def infer_sim_video(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int = 0,
        action: torch.Tensor = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
        return_latents: bool = False,
    ) -> dict[str, Any]:
        """Generate sim video with the action-only branch used by training_loss.

        No sim condition is supplied to the additive convolutions. Action CFG
        uses zero actions, matching this model's training dropout convention.
        """
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError(
                "Input dimensions/frame count must satisfy Wan2.2 resize rules."
            )
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"`action` must be [1,T,A] or [T,A], got {tuple(action.shape)}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        first_sim = (
            None
            if drop_first_frame
            else self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
        )
        gen = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        sim_latent = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=gen,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if fuse_flag and first_sim is not None:
            sim_latent[:, :, 0:1] = first_sim.clone()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=sim_latent.dtype,
            shift_override=sigma_shift,
        )
        if len(timesteps) == 0:
            raise ValueError("Inference schedule is empty.")
        zero_action = torch.zeros_like(action)
        do_cfg = float(action_cfg_scale) != 1.0
        for step_t, step_d in zip(timesteps, deltas):
            t = step_t.view(1).to(device=self.device, dtype=sim_latent.dtype)
            pred_cond = self._forward_student_with_action(
                latents=sim_latent,
                timestep_video=t,
                action=action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            if do_cfg:
                pred_zero = self._forward_student_with_action(
                    latents=sim_latent,
                    timestep_video=t,
                    action=zero_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                pred_sim = pred_zero + float(action_cfg_scale) * (pred_cond - pred_zero)
            else:
                pred_sim = pred_cond
            sim_latent = self.infer_video_scheduler.step(pred_sim, step_d, sim_latent)
            if fuse_flag and first_sim is not None:
                sim_latent[:, :, 0:1] = first_sim.clone()
        result = {"video": self._decode_latents(sim_latent, tiled=tiled)}
        if return_latents:
            result["latents"] = sim_latent
        return result

    @property
    def lambda_diffusion_real(self) -> float:
        """Per-branch weight of the REAL diffusion (flow-matching) loss."""
        v = getattr(self, "loss_lambda_diffusion_real", None)
        return float(self.loss_lambda_real if v is None else v)

    @property
    def lambda_diffusion_syn(self) -> float:
        """Per-branch weight of the SYN diffusion (flow-matching) loss."""
        v = getattr(self, "loss_lambda_diffusion_syn", None)
        return float(self.loss_lambda_syn if v is None else v)

    @property
    def lambda_delta_real(self) -> float:
        """Per-branch weight of the REAL adjacent-delta distillation loss."""
        v = getattr(self, "loss_lambda_delta_real", None)
        return float(self.loss_lambda_real if v is None else v)

    @property
    def lambda_delta_syn(self) -> float:
        """Per-branch weight of the SYN adjacent-delta distillation loss."""
        v = getattr(self, "loss_lambda_delta_syn", None)
        return float(self.loss_lambda_syn if v is None else v)

    def _delta_sigma_weight(
        self, timestep: torch.Tensor, fm_weight: torch.Tensor
    ) -> torch.Tensor:
        """Timestep weighting for the delta distillation loss."""
        mode = str(getattr(self, "delta_sigma_weight_mode", "fm"))
        if mode == "fm":
            return fm_weight
        sigma = (
            timestep.to(dtype=torch.float32)
            / float(self.train_video_scheduler.num_train_timesteps)
        ).clamp(0.0, 1.0)
        sigma = sigma.to(device=fm_weight.device, dtype=fm_weight.dtype)
        if mode == "low_sigma":
            return 1.0 - sigma
        if mode == "uniform":
            return torch.ones_like(sigma)
        raise ValueError(f"Unknown delta_sigma_weight_mode={mode!r}.")

    def _apply_rollout_drift(self, sim_latent: torch.Tensor) -> torch.Tensor:
        prob = float(getattr(self, "sim_cond_drift_prob", 0.0))
        if not self._in_training_phase() or prob <= 0.0:
            return sim_latent
        batch_size = sim_latent.shape[0]
        device, dtype = (sim_latent.device, sim_latent.dtype)
        apply_mask = torch.rand((batch_size,), device=device) < prob
        if not bool(apply_mask.any()):
            return sim_latent
        lo, hi = self.sim_cond_drift_scale_range
        shape = (batch_size,) + (1,) * (sim_latent.dim() - 1)
        scale = (
            torch.empty((batch_size,), device=device, dtype=dtype)
            .uniform_(float(lo), float(hi))
            .view(shape)
        )
        steps = torch.randn_like(sim_latent) * scale
        steps[:, :, 0:1] = 0.0
        drift = torch.cumsum(steps, dim=2)
        gate = apply_mask.to(dtype).view(shape)
        return sim_latent + gate * drift

    def _delta_loss_per_sample(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        k = int(getattr(self, "delta_spatial_pool", 1))
        if k > 1:
            pred = F.avg_pool3d(pred, kernel_size=(1, k, k), stride=(1, k, k))
            target = F.avg_pool3d(target, kernel_size=(1, k, k), stride=(1, k, k))
        if bool(getattr(self, "delta_cosine", False)):
            b = pred.shape[0]
            return 1.0 - F.cosine_similarity(
                pred.reshape(b, -1), target.reshape(b, -1), dim=1, eps=1e-06
            )
        return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3, 4))

    @torch.no_grad()
    def infer_sim_latents(
        self,
        input_image: torch.Tensor,
        num_video_frames: int,
        action: torch.Tensor,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
    ) -> torch.Tensor:
        """Generate a sim latent using the action-conditioned student trunk.

        Uses the supplied sim first frame and the same text context as training,
        without a sim-latent condition. The result feeds the real infer_video
        stage directly, without decoding and re-encoding.
        """
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError(
                "Input dimensions/frame count must satisfy Wan2.2 resize rules."
            )
        if action is None:
            raise ValueError("`action` is required to roll out the sim latent.")
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"`action` must be [1,T,A] or [T,A], got {tuple(action.shape)}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype)
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "The sim rollout needs the SIM prompt: pass `prompt` or both `context`/`context_mask`."
            )
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        first_frame = (
            None
            if drop_first_frame
            else self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
        )
        gen = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        sim_latent = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=gen,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if fuse_flag and first_frame is not None:
            sim_latent[:, :, 0:1] = first_frame.clone()
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=sim_latent.dtype,
            shift_override=sigma_shift,
        )
        if len(timesteps) == 0:
            raise ValueError("Inference schedule is empty.")
        zero_action = torch.zeros_like(action)
        do_cfg = float(action_cfg_scale) != 1.0
        for step_t, step_d in zip(timesteps, deltas):
            t = step_t.view(1).to(device=self.device, dtype=sim_latent.dtype)
            pred_cond = self._forward_student_with_action(
                latents=sim_latent,
                timestep_video=t,
                action=action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            if do_cfg:
                pred_zero = self._forward_student_with_action(
                    latents=sim_latent,
                    timestep_video=t,
                    action=zero_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                pred = pred_zero + float(action_cfg_scale) * (pred_cond - pred_zero)
            else:
                pred = pred_cond
            sim_latent = self.infer_video_scheduler.step(pred, step_d, sim_latent)
            if fuse_flag and first_frame is not None:
                sim_latent[:, :, 0:1] = first_frame.clone()
        return sim_latent

    @torch.no_grad()
    def _infer_video_conditioned(
        self,
        prompt: Optional[str],
        input_image_syn: Optional[torch.Tensor],
        input_image_real: torch.Tensor,
        sim_video: Optional[torch.Tensor] = None,
        sim_latents: Optional[torch.Tensor] = None,
        num_video_frames: int = 0,
        action: torch.Tensor = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
    ) -> dict[str, Any]:
        """Denoise the real stream conditioned on the sim latent (static).

        The sim condition is supplied EITHER as a ``sim_video`` (the GT sim
        render, VAE-encoded once into the clean sim latent) OR as a
        pre-computed ``sim_latents`` tensor (e.g. produced by an external sim
        world model, passed directly to skip VAE re-encoding). The chosen sim
        latent is ADDED frame-by-frame (via a per-block learnable 3D conv) into
        each video DiT block. There is no sim-denoising sub-loop.
        """
        del input_image_syn
        self.eval()
        if input_image_real.ndim == 3:
            input_image_real = input_image_real.unsqueeze(0)
        if input_image_real.ndim != 4 or input_image_real.shape[:2] != (1, 3):
            raise ValueError(
                f"`input_image_real` must be [1,3,H,W] or [3,H,W], got {tuple(input_image_real.shape)}"
            )
        _, _, height, width = input_image_real.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError(
                "Input dimensions/frame count must satisfy Wan2.2 resize rules."
            )
        if sim_latents is None and sim_video is None:
            raise ValueError(
                "Provide either `sim_video` or `sim_latents` as the sim condition."
            )
        if sim_latents is not None:
            if sim_latents.ndim == 4:
                sim_latents = sim_latents.unsqueeze(0)
            if sim_latents.ndim != 5 or sim_latents.shape[0] != 1:
                raise ValueError(
                    f"`sim_latents` must be [1, z_dim, T, H, W], got {tuple(sim_latents.shape)}"
                )
        else:
            if sim_video.ndim == 4:
                sim_video = sim_video.unsqueeze(0)
            if (
                sim_video.ndim != 5
                or sim_video.shape[0] != 1
                or sim_video.shape[1] != 3
                or (sim_video.shape[2] != num_video_frames)
            ):
                raise ValueError(
                    f"`sim_video` must be [1,3,{num_video_frames},H,W], got {tuple(sim_video.shape)}"
                )
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"`action` must be [1,T,A] or [T,A], got {tuple(action.shape)}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.proprio_dim}], got {tuple(proprio.shape)}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        if sim_latents is not None:
            clean_sim = sim_latents.to(device=self.device, dtype=self.torch_dtype)
        else:
            sim_video = sim_video.to(device=self.device, dtype=self.torch_dtype)
            clean_sim = self._encode_video_latents(sim_video, tiled=tiled)
        if clean_sim.shape[2] != latent_t:
            raise RuntimeError(
                f"Sim latent has T={clean_sim.shape[2]}, expected {latent_t}."
            )
        input_image_real = input_image_real.to(
            device=self.device, dtype=self.torch_dtype
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        first_real = (
            None
            if drop_first_frame
            else self._encode_input_image_latents_tensor(
                input_image=input_image_real, tiled=tiled
            )
        )
        gen = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        real_latent = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=gen,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        if fuse_flag and first_real is not None:
            real_latent[:, :, 0:1] = first_real.clone()
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and (not use_context):
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        timesteps, deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=real_latent.dtype,
            shift_override=sigma_shift,
        )
        if len(timesteps) == 0:
            raise ValueError("Inference schedule is empty.")
        zero_action = torch.zeros_like(action)
        do_cfg = float(action_cfg_scale) != 1.0
        for step_t, step_d in zip(timesteps, deltas):
            t = step_t.view(1).to(device=self.device, dtype=real_latent.dtype)
            pred_cond = self._forward_student_with_sim_condition(
                latents=real_latent,
                timestep_video=t,
                action=action,
                sim_latent=clean_sim,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            if do_cfg:
                pred_zero = self._forward_student_with_sim_condition(
                    latents=real_latent,
                    timestep_video=t,
                    action=zero_action,
                    sim_latent=clean_sim,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                pred_real = pred_zero + float(action_cfg_scale) * (
                    pred_cond - pred_zero
                )
            else:
                pred_real = pred_cond
            real_latent = self.infer_video_scheduler.step(
                pred_real, step_d, real_latent
            )
            if fuse_flag and first_real is not None:
                real_latent[:, :, 0:1] = first_real.clone()
        return {"video": self._decode_latents(real_latent, tiled=tiled)}
