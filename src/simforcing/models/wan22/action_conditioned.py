"""Action-conditioned video generation VideoActionModel variant.

Architecture difference vs. base ``VideoActionModel``:
    * Base model: video and action are *independently* denoised; action
      attends to the first video frame for grounding, but video does **not**
      see the action.
    * This variant (``ActionConditionedModel``): video is the only diffusion target.
      The Action-DiT is repurposed as a *clean* feature extractor over the
      ground-truth action sequence (run with ``timestep_action=0``), and the
      resulting action tokens are exposed to *all* video tokens through MoT
      mixed attention. The action expert never sees video tokens.

Notes:
    * No action diffusion loss; ``loss_lambda_action`` is forced to 0.
    * Random action dropout (``action_dropout_prob``) is applied during
      training to enable classifier-free guidance at inference time
      (``action_cfg_scale``).
    * When the action is dropped (or absent at inference time) the action
      branch is skipped entirely, so the video expert runs as a regular
      text-conditioned DiT.

Designed for the Bridge-V2 dataset (1 action <-> 1 video frame), but the
implementation only requires that ``action_horizon`` be divisible by
``num_video_transitions`` and that ``num_heads``/``attn_head_dim``/
``num_layers`` match between the two experts (already enforced upstream).
"""

from typing import Any, Optional
import torch
from simforcing.utils.logging_config import get_logger
from .base import VideoActionModel
from .helpers.gradient import gradient_checkpoint_forward

logger = get_logger(__name__)


class ActionConditionedModel(VideoActionModel):
    """Action-conditioned video generation: action features condition video."""

    action_dropout_prob: float = 0.1
    first_frame_dropout_prob: float = 0.0
    loss_weight_sim: float = 1.0
    loss_weight_real: float = 1.0

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for ActionConditionedModel."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "ActionConditionedModel requires `video_dit_config['action_conditioned']=false`. Action conditioning is provided through MoT mixed attention with the ActionDiT expert, NOT through the video DiT's built-in `action_embedding`."
            )
        existing_lambda_action = float(kwargs.get("loss_lambda_action", 0.0))
        if existing_lambda_action != 0.0:
            logger.warning(
                "ActionConditionedModel ignores `loss_lambda_action=%s`; the action branch is a feature extractor and produces no diffusion loss. Forcing it to 0.",
                existing_lambda_action,
            )
        kwargs["loss_lambda_action"] = 0.0
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
        loss_weight_sim = float(kwargs.pop("loss_weight_sim", cls.loss_weight_sim))
        loss_weight_real = float(kwargs.pop("loss_weight_real", cls.loss_weight_real))
        if loss_weight_sim < 0.0 or loss_weight_real < 0.0:
            raise ValueError(
                f"`loss_weight_sim`/`loss_weight_real` must be >= 0, got sim={loss_weight_sim}, real={loss_weight_real}"
            )
        model = super().from_wan22_pretrained(**kwargs)
        model.action_dropout_prob = action_dropout_prob
        model.first_frame_dropout_prob = first_frame_dropout_prob
        model.loss_weight_sim = loss_weight_sim
        model.loss_weight_real = loss_weight_real
        logger.info(
            "ActionConditionedModel initialized: action_dropout_prob=%.3f, first_frame_dropout_prob=%.3f, loss_lambda_action=0.0, loss_weight_sim=%.3f, loss_weight_real=%.3f",
            action_dropout_prob,
            first_frame_dropout_prob,
            loss_weight_sim,
            loss_weight_real,
        )
        return model

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        mask[:video_seq_len, :video_seq_len] = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                device=device,
            )
        )
        mask[:video_seq_len, video_seq_len:] = True
        mask[video_seq_len:, video_seq_len:] = True
        return mask

    def _forward_video_with_action_cond(
        self,
        latents: torch.Tensor,
        timestep_video: torch.Tensor,
        action: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        first_frame_is_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict video noise with clean-action conditioning via MoT.

        If ``action`` is None, the action branch is skipped entirely and the
        video expert runs as a plain text/proprio-conditioned DiT.

        ``first_frame_is_clean`` is an optional ``[B]`` bool tensor telling
        the video expert which samples carry a CLEAN (GT-injected) first
        latent frame. ``None`` keeps the legacy behaviour (first frame is
        always treated as a sigma=0 anchor). Pass it whenever the first
        latent frame may be left noisy, so the per-token timestep embedding
        matches the latent's real noise level.
        """
        batch_size = latents.shape[0]
        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            first_frame_is_clean=first_frame_is_clean,
        )
        if action is None:
            x_tokens = video_pre["tokens"]
            context_emb = video_pre["context"]
            t_mod = video_pre["t_mod"]
            freqs = video_pre["freqs"]
            context_attn_mask = video_pre["context_mask"]
            self_attn_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            for block in self.video_expert.blocks:
                if self.video_expert.use_gradient_checkpointing and self.training:
                    x_tokens = gradient_checkpoint_forward(
                        block,
                        self.video_expert.use_gradient_checkpointing,
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
            return self.video_expert.post_dit(x_tokens, video_pre)
        timestep_action = torch.zeros(
            (batch_size,), dtype=action.dtype, device=action.device
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
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

    def _domain_weight_from_sample(
        self, sample: dict, batch_size: int, ref: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Build a per-sample multiplicative weight tensor from `sample['half']`.

        Returns ``None`` when no per-domain re-weighting is needed
        (i.e. both weights are 1.0, or the field is missing). Otherwise
        returns a [B] float tensor on the same device/dtype as `ref`.
        """
        ws = float(self.loss_weight_sim)
        wr = float(self.loss_weight_real)
        if ws == 1.0 and wr == 1.0:
            return None
        halves = sample.get("half", None)
        if halves is None:
            return None
        if isinstance(halves, str):
            halves = [halves]
        if len(halves) != batch_size:
            logger.warning(
                "ActionConditionedModel: sample['half'] length=%d mismatches batch_size=%d; skipping per-domain loss weighting.",
                len(halves),
                batch_size,
            )
            return None
        weights = []
        for h in halves:
            if h == "left":
                weights.append(ws)
            elif h == "right":
                weights.append(wr)
            else:
                weights.append(1.0)
        return torch.tensor(weights, device=ref.device, dtype=ref.dtype)

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        first_frame_latents = inputs["first_frame_latents"]
        has_first_frame_cond = first_frame_latents is not None
        if (
            has_first_frame_cond
            and self.training
            and (self.first_frame_dropout_prob > 0.0)
        ):
            ff_drop_mask = torch.rand(
                (batch_size,), device=input_latents.device
            ) < float(self.first_frame_dropout_prob)
        else:
            ff_drop_mask = torch.zeros(
                (batch_size,), dtype=torch.bool, device=input_latents.device
            )
        if has_first_frame_cond:
            keep_ff = (~ff_drop_mask).view(batch_size, 1, 1, 1, 1)
            latents[:, :, 0:1] = torch.where(
                keep_ff, first_frame_latents, latents[:, :, 0:1]
            )
        if self.training and self.action_dropout_prob > 0.0:
            drop_mask = torch.rand((batch_size,), device=action.device) < float(
                self.action_dropout_prob
            )
        else:
            drop_mask = torch.zeros(
                (batch_size,), dtype=torch.bool, device=action.device
            )
        if bool(drop_mask.all()):
            action_for_cond: Optional[torch.Tensor] = None
        elif bool(drop_mask.any()):
            keep = (~drop_mask).to(dtype=action.dtype).view(batch_size, 1, 1)
            action_for_cond = action * keep
        else:
            action_for_cond = action
        pred_video = self._forward_video_with_action_cond(
            latents=latents,
            timestep_video=timestep_video,
            action=action_for_cond,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            first_frame_is_clean=~ff_drop_mask if has_first_frame_cond else None,
        )
        if not has_first_frame_cond:
            include_initial_video_step = True
        elif not bool(ff_drop_mask.any()):
            include_initial_video_step = False
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        else:
            include_initial_video_step = True
            temporal_factor = int(self.vae.temporal_downsample_factor)
            latent_t = target_video.shape[2]
            num_pixel_frames = (latent_t - 1) * temporal_factor + 1
            if image_is_pad is None:
                image_is_pad = torch.zeros(
                    (batch_size, num_pixel_frames),
                    dtype=torch.bool,
                    device=target_video.device,
                )
            else:
                image_is_pad = image_is_pad.clone()
            image_is_pad[:, 0] = image_is_pad[:, 0] | ~ff_drop_mask
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        domain_weight = self._domain_weight_from_sample(
            sample=sample, batch_size=batch_size, ref=loss_video_per_sample
        )
        if domain_weight is not None:
            loss_video_weighted_per_sample = (
                loss_video_per_sample * video_weight * domain_weight
            )
            loss_video = loss_video_weighted_per_sample.mean()
        else:
            loss_video = (loss_video_per_sample * video_weight).mean()
        loss_total = self.loss_lambda_video * loss_video
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": 0.0,
        }
        halves = sample.get("half", None)
        if (
            halves is not None
            and isinstance(halves, (list, tuple))
            and (len(halves) == batch_size)
        ):
            with torch.no_grad():
                raw_per_sample = (loss_video_per_sample * video_weight).detach()
                sim_mask = torch.tensor(
                    [h == "left" for h in halves],
                    device=raw_per_sample.device,
                    dtype=torch.bool,
                )
                real_mask = torch.tensor(
                    [h == "right" for h in halves],
                    device=raw_per_sample.device,
                    dtype=torch.bool,
                )
                if sim_mask.any():
                    loss_dict["loss_video_sim"] = float(
                        raw_per_sample[sim_mask].mean().item()
                    )
                if real_mask.any():
                    loss_dict["loss_video_real"] = float(
                        raw_per_sample[real_mask].mean().item()
                    )
        return (loss_total, loss_dict)

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        raise NotImplementedError(
            "ActionConditionedModel is a video-generation model conditioned on a given ground-truth action sequence; it does not predict actions."
        )

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError(
            "ActionConditionedModel does not run joint denoising; use `infer_video` (or `infer`) with a ground-truth `action` tensor."
        )

    @torch.no_grad()
    def infer_video(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
        return_latents: bool = False,
    ) -> dict[str, Any]:
        """Action-conditioned video generation.

        Args:
            action: Required ground-truth action sequence used to condition
                video denoising. Shape ``[T, a_dim]`` or ``[1, T, a_dim]``.
                ``T`` must be divisible by ``num_video_frames - 1``.
            action_cfg_scale: Classifier-free guidance scale for the action
                conditioning. ``1.0`` disables CFG (single forward pass);
                values ``>1`` interpolate ``cond + (s-1)*(cond - uncond)``.
                Requires the model to have been trained with
                ``action_dropout_prob > 0`` for meaningful uncond behaviour.
            drop_first_frame: If True, the first-frame latent grounding is
                disabled at inference: the first latent step is denoised from
                pure noise just like the rest of the clip and is *not*
                re-clamped to the encoded ``input_image``. ``input_image`` is
                still used to derive the spatial shape (and so must satisfy
                the usual size checks). Use this only with a checkpoint that
                was trained with ``first_frame_dropout_prob > 0``; otherwise
                quality will degrade.
        """
        del negative_prompt, text_cfg_scale
        self.eval()
        if action is None:
            raise ValueError(
                "`action` is required for ActionConditionedModel.infer_video."
            )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3 or action.shape[0] != 1:
            raise ValueError(
                f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        num_video_transitions = num_video_frames - 1
        if num_video_transitions <= 0:
            raise ValueError(f"`num_video_frames` must be > 1, got {num_video_frames}")
        if action_horizon % num_video_transitions != 0:
            raise ValueError(
                f"`action` horizon must be divisible by num_video_frames - 1 ({num_video_transitions}), got {action_horizon}"
            )
        if int(action.shape[2]) != int(self.action_expert.action_dim):
            raise ValueError(
                f"`action` last dim must be {self.action_expert.action_dim}, got {action.shape[2]}"
            )
        action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if not drop_first_frame and self.first_frame_dropout_prob >= 0.9:
            drop_first_frame = True
            logger.warning(
                "Forcing drop_first_frame=True because first_frame_dropout_prob=%.3f (the checkpoint almost never saw a clean first frame during training).",
                self.first_frame_dropout_prob,
            )
        if drop_first_frame:
            first_frame_latents = None
            if self.first_frame_dropout_prob <= 0.0:
                logger.warning(
                    "infer_video(drop_first_frame=True) was requested but the model reports first_frame_dropout_prob=%.3f. Results may be poor unless the checkpoint was trained with first_frame_dropout_prob > 0.",
                    self.first_frame_dropout_prob,
                )
        else:
            first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=input_image, tiled=tiled
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
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
                    "`context` and `context_mask` must be both provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
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
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )
        do_action_cfg = float(action_cfg_scale) != 1.0
        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        ff_clean_flag = (
            None
            if first_frame_latents is not None
            else torch.zeros((1,), dtype=torch.bool, device=self.device)
        )
        for step_t_video, step_delta_video in zip(
            infer_timesteps_video, infer_deltas_video
        ):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            pred_cond = self._forward_video_with_action_cond(
                latents=latents_video,
                timestep_video=timestep_video,
                action=action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                first_frame_is_clean=ff_clean_flag,
            )
            if do_action_cfg:
                pred_uncond = self._forward_video_with_action_cond(
                    latents=latents_video,
                    timestep_video=timestep_video,
                    action=None,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    first_frame_is_clean=ff_clean_flag,
                )
                pred_video = pred_uncond + float(action_cfg_scale) * (
                    pred_cond - pred_uncond
                )
            else:
                pred_video = pred_cond
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            if first_frame_latents is not None:
                latents_video[:, :, 0:1] = first_frame_latents.clone()
        decoded = self._decode_latents(latents_video, tiled=tiled)
        if return_latents:
            return {"video": decoded, "latents": latents_video}
        return {"video": decoded}

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        drop_first_frame: bool = False,
    ):
        del action_horizon
        if action is None:
            raise ValueError(
                "ActionConditionedModel.infer requires a ground-truth `action` tensor (action-conditioned video generation)."
            )
        return self.infer_video(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            action_cfg_scale=action_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            drop_first_frame=drop_first_frame,
        )
