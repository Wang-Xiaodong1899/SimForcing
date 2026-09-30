"""BridgeData V2 paired sim/real dataset.

For every sampled window this dataset returns BOTH the sim-render (left
half of the source mp4) and the real-camera (right half) views, sharing
the SAME action / proprio / instruction / start_frame. Unlike
``BridgeV2VideoDataset(paired_halves=True)`` which emits the two halves
as separate samples, this dataset emits them as a single sample with
two video tensors (``video_syn`` and ``video_real``). This is the input
format expected by ``SimForcing``: the sim latent is denoised
in a no-grad forward pass to produce per-frame latent deltas that
condition the real-latent prediction.

Sample format (per ``__getitem__``):
    {
        "video_syn":  [3, T, H, W] float in [-1, 1]   # left half, sim
        "video_real": [3, T, H, W] float in [-1, 1]   # right half, real
        # The following fields are shared (identical for sim/real):
        "action":         [T-1, action_dim]
        "proprio":        [T-1, proprio_dim]
        "prompt":         str
        "context":        [L, D]
        "context_mask":   [L]
        "image_is_pad":   [T] (all False)
        "action_is_pad":  [T-1] (all False)
        "proprio_is_pad": [T-1] (all False)
    }

With ``use_domain_prompts=True`` the shared prompt is replaced by one
template per target domain -- ``prompt`` / ``context`` / ``context_mask``
carry REAL_PROMPT (matching ``video_real``) and three extra fields
``prompt_syn`` / ``context_syn`` / ``context_mask_syn`` carry SIM_PROMPT
(matching ``video_syn``).
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

import numpy as np
import torch

from .bridge_video_dataset import BridgeV2VideoDataset, _read_video_decord
from simforcing.datasets.prompts import DEFAULT_PROMPT, build_prompt
from simforcing.utils.logging_config import get_logger

logger = get_logger(__name__)


class BridgeV2PairedSynRealDataset(BridgeV2VideoDataset):
    """Returns both sim (left half) and real (right half) views per sample."""

    def __init__(self, *args, use_domain_prompts: bool = False, **kwargs):
        # Force the underlying half-selection flags so the parent class
        # doesn't try to crop the video itself -- we crop both halves
        # manually in ``_load_paired_window``.
        kwargs["use_left_half_only"] = False
        kwargs["use_right_half_only"] = False
        kwargs["random_half"] = False
        kwargs["paired_halves"] = False
        # When True the two views no longer share one prompt: the sim view is
        # described with SIM_PROMPT ("A simulation-rendered video ...") and the
        # real view with REAL_PROMPT ("A real-world video ..."), so each branch
        # of a paired model is conditioned on the template matching its own
        # target domain. `prompt` / `context` stay on the REAL template (the
        # deployment target and what the trainer's PSNR eval rolls out), and the
        # extra `prompt_syn` / `context_syn` / `context_mask_syn` fields carry
        # the sim one. When False (default) both views share DEFAULT_PROMPT and
        # no extra field is emitted, i.e. the historical behaviour.
        self.use_domain_prompts = bool(use_domain_prompts)
        super().__init__(*args, **kwargs)
        logger.info(
            "BridgeV2PairedSynRealDataset(%s) initialized: %d episodes, "
            "epoch_size=%d (one sample = sim+real pair), "
            "use_domain_prompts=%s.",
            "train" if self.is_training_set else "val",
            len(self._episode_pool),
            self._epoch_size,
            self.use_domain_prompts,
        )

    def __len__(self) -> int:
        # paired_halves doubling is disabled in this class; one sample per
        # (episode, start_frame).
        return self._epoch_size

    # ------------------------------------------------------------------ #
    # Window construction (both halves)                                  #
    # ------------------------------------------------------------------ #

    def _load_paired_window(
        self,
        di: int,
        ep_local: int,
        start: int,
    ) -> Dict[str, Any]:
        rec = self._indices[di].get_record(ep_local)
        num_steps = int(rec["num_steps"])
        T = self.num_frames
        if start < 0 or start + T > num_steps:
            raise ValueError(
                f"Invalid window start={start}, num_steps={num_steps}, T={T}."
            )

        arrs = self._indices[di].load_episode_arrays(ep_local)
        action_full = torch.from_numpy(arrs["action"]).float()  # [N, action_dim]
        state_full = torch.from_numpy(arrs["state"]).float()    # [N, proprio_dim]
        instruction = rec["instruction"]
        video_path = rec["_resolved_video_path"]

        # ---- Read full video ONCE, then split into two halves ---- #
        try:
            full_video = _read_video_decord(video_path)  # [F, C, H, W] uint8
        except Exception as e:
            raise RuntimeError(f"Failed to read {video_path}: {e}") from e
        if full_video.shape[0] < num_steps:
            raise RuntimeError(
                f"Video {video_path} has {full_video.shape[0]} frames but "
                f"num_steps={num_steps}; cache may be stale. "
                f"Delete <dataset_dir>/meta/video_frame_counts.json and retry."
            )

        W = full_video.shape[-1]
        if W % 2 != 0:
            raise ValueError(
                f"Paired sim/real requires an even width, got W={W} for {video_path}."
            )
        half_w = W // 2
        left_video = full_video[..., :half_w]   # sim render
        right_video = full_video[..., half_w:]  # real camera

        per_step_left = left_video[start : start + T]    # [T, C, H, W]
        per_step_right = right_video[start : start + T]  # [T, C, H, W]

        T_video_indices = torch.tensor(self.video_sample_indices, dtype=torch.long)
        video_syn = per_step_left[T_video_indices]   # [T_v, C, H, W]
        video_real = per_step_right[T_video_indices] # [T_v, C, H, W]

        # Apply identical image transforms to both halves.
        video_syn = self.normalize_transform(
            self.crop_transform(self.resize_transform(video_syn))
        )
        video_real = self.normalize_transform(
            self.crop_transform(self.resize_transform(video_real))
        )
        video_syn = video_syn.permute(1, 0, 2, 3).contiguous()    # [C, T_v, H, W]
        video_real = video_real.permute(1, 0, 2, 3).contiguous()  # [C, T_v, H, W]

        if video_syn.shape != video_real.shape:
            raise RuntimeError(
                "Sim/real video shape mismatch after transforms: "
                f"sim={tuple(video_syn.shape)}, real={tuple(video_real.shape)}"
            )

        # ---- Action / proprio (shared between sim and real) ---- #
        action = action_full[start : start + (T - 1)].clone()  # [T-1, action_dim]
        proprio = state_full[start : start + T].clone()        # [T, proprio_dim]
        if self._action_norm is not None:
            action = self._action_norm.forward(action)
        if self._state_norm is not None:
            proprio = self._state_norm.forward(proprio)
        proprio_window = proprio[:-1, :]                       # [T-1, proprio_dim]

        zero_pad_action = torch.zeros(T - 1, dtype=torch.bool)
        zero_pad_image = torch.zeros(video_real.shape[1], dtype=torch.bool)

        if video_real.shape[1] <= 1:
            raise ValueError(
                f"`video` must have at least 2 frames, got shape {tuple(video_real.shape)}"
            )
        if action.shape[0] % (video_real.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon ({action.shape[0]}) must be divisible by "
                f"`video` transitions ({video_real.shape[1] - 1})."
            )

        # ---- Instruction / text embedding ---- #
        if self.override_instruction is not None:
            instruction = self.override_instruction
        if self.use_domain_prompts:
            # One template per target domain: `half="left"` -> SIM_PROMPT,
            # `half="right"` -> REAL_PROMPT (see PROMPT_TEMPLATE_BY_HALF).
            prompt = build_prompt(instruction, half="right")
            prompt_syn = build_prompt(instruction, half="left")
        else:
            prompt = DEFAULT_PROMPT.format(task=instruction)
            prompt_syn = None
        context, context_mask = self._get_cached_text_context(prompt)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        sample = {
            "video_syn": video_syn,
            "video_real": video_real,
            # Alias for trainer / standard eval path: the PSNR rollout in
            # `Wan22Trainer.evaluate()` reads `sample["video"]` to drive
            # `_to_batched_eval_sample` and to compute PSNR/SSIM against
            # ground truth. Here we use the REAL view, since real-camera
            # frames are the deployment target. Models that consume this
            # dataset (e.g. SimForcing) read `video_syn`/`video_real`
            # directly and never look at this alias, so it is loss-free.
            "video": video_real,
            "action": action,
            "proprio": proprio_window,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": zero_pad_image,
            "action_is_pad": zero_pad_action,
            "proprio_is_pad": zero_pad_action.clone(),
            # 'right' = real camera. Recorded for downstream domain-aware
            # logic that may inspect `sample['half']`.
            "half": "right",
        }

        if prompt_syn is not None:
            context_syn, context_mask_syn = self._get_cached_text_context(prompt_syn)
            context_syn[~context_mask_syn] = 0.0
            context_mask_syn = torch.ones_like(context_mask_syn)
            sample["prompt_syn"] = prompt_syn
            sample["context_syn"] = context_syn
            sample["context_mask_syn"] = context_mask_syn

        return sample

    # ------------------------------------------------------------------ #
    # __getitem__                                                        #
    # ------------------------------------------------------------------ #

    def _get(self, idx: int) -> Dict[str, Any]:
        ep_idx = idx % len(self._episode_pool)
        di, ep_local = self._episode_pool[ep_idx]
        num_steps = int(self._indices[di].get_record(ep_local)["num_steps"])
        max_start = num_steps - self.num_frames  # >= 0 by construction
        start = int(np.random.randint(0, max_start + 1))
        return self._load_paired_window(di, ep_local, start)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            return self._get(idx)
        except Exception as e:
            print(
                f"Error processing paired sim/real sample idx {idx}: {e}. "
                "Returning a random sample instead."
            )
            print(traceback.format_exc())
            random_idx = int(np.random.randint(len(self)))
            return self._get(random_idx)
