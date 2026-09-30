from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from simforcing.utils.logging_config import get_logger
from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class VideoActionModel(torch.nn.Module):
    """Shared video/action backbone, latent encoding and checkpoint IO."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(
                torch_dtype
            )
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "./checkpoints/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "./checkpoints/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError(
                "`video_dit_config` is required for VideoActionModel.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError(
                "`video_dit_config['text_dim']` is required for VideoActionModel."
            )
        print(f"Loading WAN2.2 components from {model_id}...")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for MoT mixed attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for MoT mixed attention."
            )
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return (height, width, num_frames)

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (prompt_emb.to(device=self.device), mask)

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(
                f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
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
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(
        self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "VideoActionModel training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )
        if "action" not in sample:
            raise ValueError(
                "`sample['action']` is required for VideoActionModel training."
            )
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    f"`sample['action_is_pad']` shape mismatch: got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )
        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    f"`sample['image_is_pad']` shape mismatch: got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        input_video = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
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
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)
        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                f"Cannot align `image_is_pad` with video latent steps: num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad
        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                f"Video-loss mask shape mismatch: mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )
        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

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
            ckpt_mot = payload["mot"]
            cur_mot = self.mot.state_dict()
            mismatched = {
                k: (tuple(v.shape), tuple(cur_mot[k].shape))
                for k, v in ckpt_mot.items()
                if k in cur_mot and cur_mot[k].shape != v.shape
            }
            if mismatched:
                logger.warning(
                    "`mot` shape mismatch for keys %s; skipping those and loading the rest with strict=False.",
                    mismatched,
                )
                ckpt_mot = {k: v for k, v in ckpt_mot.items() if k not in mismatched}
            self.mot.load_state_dict(ckpt_mot, strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                ckpt_pe = payload["proprio_encoder"]
                cur_pe = self.proprio_encoder.state_dict()
                shape_mismatch = any(
                    (
                        k in cur_pe and cur_pe[k].shape != v.shape
                        for k, v in ckpt_pe.items()
                    )
                )
                if shape_mismatch:
                    logger.warning(
                        "`proprio_encoder` shape mismatch between checkpoint and current model (likely different `proprio_dim`); skipping load and keeping randomly initialized params. Checkpoint shapes: %s; current shapes: %s",
                        {k: tuple(v.shape) for k, v in ckpt_pe.items()},
                        {k: tuple(v.shape) for k, v in cur_pe.items()},
                    )
                else:
                    self.proprio_encoder.load_state_dict(ckpt_pe, strict=True)
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
