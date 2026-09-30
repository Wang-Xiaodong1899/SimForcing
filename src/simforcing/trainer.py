import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .datasets.prompts import retarget_prompt
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)

# Maps a dataset window tag (`sample['half']`) to the domain of the TARGET
# video, i.e. the frames the model must predict. Note that
# 'first_real_rest_sim' is a SIM target even though its conditioning first
# frame is a real camera image.
_EVAL_DOMAIN_BY_HALF = {
    "left": "sim",
    "first_real_rest_sim": "sim",
    "right": "real",
}


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        # When true, every eval sample is rolled out TWICE -- once with the
        # SIM prompt and once with the REAL prompt -- keeping the first frame,
        # action, proprio and seed identical. Doubles eval inference cost and
        # is only meaningful for a checkpoint trained on mixed sim/real
        # targets, so it defaults to off.
        self.eval_both_domains = bool(cfg.get("eval_both_domains", False))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            (self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
             if self.accelerator.state.deepspeed_plugin is not None else "disabled"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Load model-only checkpoints before ZeRO creates its FP32 master
        # weights. Loading only the model after prepare() leaves stale master
        # weights that can overwrite the checkpoint on the first update.
        self._resume_or_load_checkpoint(before_prepare=True)

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint(before_prepare=False)

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self, *, before_prepare: bool):
        """Load weights before prepare; restore full state only after prepare."""
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if resume_path.is_dir():
            if not before_prepare:
                logger.info("Resuming full training state from directory: %s", resume)
                self.load_training_state(str(resume_path))
            return
        if not before_prepare:
            return
        logger.info("Loading model weights before optimizer/DeepSpeed initialization: %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        logger.info(
            "Loaded model weights; optimizer and scheduler will start fresh from "
            "these weights, with the training step reset to zero."
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            # Domain tag of the TARGET video ('left'/'first_real_rest_sim' ->
            # sim, 'right' -> real, 'full'/None -> unknown). Propagated so
            # `evaluate()` can report per-domain metrics when the val set
            # randomly mixes sim and real targets.
            "half": sample.get("half", None),
        }

    def _eval_context_for_prompt(self, prompt: str):
        """Look up a precomputed T5 embedding for an arbitrary ``prompt``.

        Used by the ``eval_both_domains`` A/B rollout, which needs the text
        embedding of a prompt the dataset did not emit for this sample.
        Returns ``(context [L, D], mask [L])`` or ``None`` when the val
        dataset has no embedding cache or the prompt is missing from it.

        The mask post-processing intentionally mirrors the datasets
        (zero out padded positions, then mark every position valid) so the
        A/B run sees exactly the same context format as training.
        """
        dataset = self.val_dataset
        getter = getattr(dataset, "_get_cached_text_context", None)
        if getter is None:
            return None
        try:
            context, context_mask = getter(prompt)
        except FileNotFoundError:
            return None
        context = context.clone()
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # Domain of the target video for THIS rank's eval sample. With a
        # mixed-target val set (`random_half` + `first_frame_real_rest_sim`)
        # this varies per eval call and per rank, so it is reported alongside
        # the metrics and baked into the saved mp4 filename.
        eval_domain = _EVAL_DOMAIN_BY_HALF.get(sample.get("half"), "unknown")
        eval_domain_suffix = "" if eval_domain == "unknown" else f"_{eval_domain}"

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 2b. Optional A/B rollout: rerun inference on the SAME sample (same
        # first frame, same action/proprio, same seed) with the prompt
        # re-targeted to each domain. Only the domain tag inside the prompt
        # differs between the two runs, so a visible difference is direct
        # evidence that the prompt controls the output domain.
        ab_videos = None
        if self.eval_both_domains and eval_domain in ("sim", "real"):
            ab_videos = {eval_domain: pred_video}
            other_domain = "real" if eval_domain == "sim" else "sim"
            other_prompt = retarget_prompt(prompt, other_domain)
            if other_prompt is None:
                logger.warning(
                    "eval_both_domains: cannot recover the task text from prompt %r; "
                    "skipping the cross-domain rollout.",
                    prompt,
                )
            else:
                other_kwargs = dict(infer_kwargs)
                other_context = self._eval_context_for_prompt(other_prompt)
                if other_context is not None:
                    other_kwargs["prompt"] = None
                    other_kwargs["context"] = other_context[0]
                    other_kwargs["context_mask"] = other_context[1]
                elif sample["context"] is not None:
                    logger.warning(
                        "eval_both_domains: the val dataset uses precomputed text "
                        "embeddings but %r is not in the cache; skipping the "
                        "cross-domain rollout. Rerun scripts/precompute_text_embeds.py.",
                        other_prompt,
                    )
                    other_kwargs = None
                else:
                    other_kwargs["prompt"] = other_prompt
                    other_kwargs.pop("context", None)
                    other_kwargs.pop("context_mask", None)
                if other_kwargs is not None:
                    # The second rollout doubles peak eval memory unless
                    # the first one's cached blocks are returned to the
                    # allocator. Eval runs inside the training loop, so
                    # optimizer state and gradients are still resident and
                    # there is little headroom left.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    ab_videos[other_domain] = model.infer(**other_kwargs)["video"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}"
            f"{eval_domain_suffix}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        # A/B comparison mp4: rows are [SIM prompt rollout | REAL prompt
        # rollout | GT]. Row order is fixed (sim on top) regardless of which
        # domain this sample's GT belongs to, so the clips stay directly
        # comparable across eval steps and ranks.
        ab_video_path = None
        if ab_videos is not None and len(ab_videos) == 2:
            ab_rows = [
                pil_frames_to_video_tensor(ab_videos["sim"]),
                pil_frames_to_video_tensor(ab_videos["real"]),
                gt_video_tensor,
            ]
            ab_stitched = torch.cat(ab_rows, dim=2).contiguous()
            ab_frames = []
            for t in range(ab_stitched.shape[1]):
                frame = (
                    ab_stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
                ).astype(np.uint8)
                ab_frames.append(Image.fromarray(frame))
            ab_video_path = os.path.join(
                self.eval_dir,
                f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}"
                f"_ab_sim_real_gt{eval_domain_suffix}.mp4",
            )
            save_mp4(ab_frames, ab_video_path, fps=8)

            # Quantify how much the prompt actually changed the output. A
            # well-separated model has a LOW psnr here (the two rollouts
            # differ); a model that ignores the domain tag converges toward
            # identical outputs and drives it up.
            ab_psnr_sim_vs_real = video_psnr(pred=ab_rows[0], target=ab_rows[1])
            ab_ssim_sim_vs_real = video_ssim(pred=ab_rows[0], target=ab_rows[1])
        else:
            ab_psnr_sim_vs_real = None
            ab_ssim_sim_vs_real = None

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
                # Per-domain indicators + the metric they weight. Needed
                # because each rank independently draws its own eval sample,
                # so a mixed-target val set means different ranks may be
                # measuring different domains.
                1.0 if eval_domain == "sim" else 0.0,
                1.0 if eval_domain == "real" else 0.0,
                # A/B prompt-separation metrics (sim-prompt rollout vs
                # real-prompt rollout on the SAME sample). Column 11 is the
                # validity flag: ranks that skipped the A/B run contribute 0
                # and are excluded from the average.
                1.0 if ab_psnr_sim_vs_real is not None else 0.0,
                float(ab_psnr_sim_vs_real) if ab_psnr_sim_vs_real is not None else 0.0,
                float(ab_ssim_sim_vs_real) if ab_ssim_sim_vs_real is not None else 0.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        # Per-domain PSNR/SSIM (rollout vs GT), averaged only over the ranks
        # that actually evaluated that domain. Absent when no rank drew it.
        per_domain: dict[str, float] = {}
        for domain_name, col in (("sim", 9), ("real", 10)):
            domain_mask = gathered_metrics[:, col] > 0.5
            count = int(domain_mask.sum().item())
            if count == 0:
                continue
            per_domain[f"psnr_rg_{domain_name}"] = float(
                gathered_metrics[domain_mask, 1].mean().item()
            )
            per_domain[f"ssim_rg_{domain_name}"] = float(
                gathered_metrics[domain_mask, 2].mean().item()
            )
            per_domain[f"val_loss_{domain_name}"] = float(
                gathered_metrics[domain_mask, 0].mean().item()
            )
            per_domain[f"count_{domain_name}"] = float(count)

        ab_mask = gathered_metrics[:, 11] > 0.5
        if int(ab_mask.sum().item()) > 0:
            per_domain["ab_psnr_sim_vs_real"] = float(
                gathered_metrics[ab_mask, 12].mean().item()
            )
            per_domain["ab_ssim_sim_vs_real"] = float(
                gathered_metrics[ab_mask, 13].mean().item()
            )
            per_domain["count_ab"] = float(int(ab_mask.sum().item()))

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
            "eval_domain": eval_domain,
        }
        if ab_video_path is not None:
            result["ab_video_path"] = ab_video_path
        result.update(per_domain)
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            for domain_name in ("sim", "real"):
                                key = f"psnr_rg_{domain_name}"
                                if key in metrics:
                                    description += " psnr_%s=%.4f(n=%d)" % (
                                        domain_name,
                                        metrics[key],
                                        int(metrics[f"count_{domain_name}"]),
                                    )
                            if "ab_psnr_sim_vs_real" in metrics:
                                description += " ab_psnr_sim_vs_real=%.4f(n=%d)" % (
                                    metrics["ab_psnr_sim_vs_real"],
                                    int(metrics["count_ab"]),
                                )
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            # Per-domain curves for mixed sim/real val sets.
                            # Absent keys simply produce gaps in wandb, which
                            # is the correct behaviour when a domain was not
                            # drawn at this eval step.
                            for domain_name in ("sim", "real"):
                                for stem in ("psnr_rg", "ssim_rg", "val_loss"):
                                    key = f"{stem}_{domain_name}"
                                    if key in metrics:
                                        eval_payload[f"eval/{key}"] = float(metrics[key])
                            # Prompt-separation: how different the SIM-prompt
                            # and REAL-prompt rollouts are on the same input.
                            # Should DROP as the model learns to honour the
                            # domain tag; a flat/high value means it is being
                            # ignored.
                            for key in ("ab_psnr_sim_vs_real", "ab_ssim_sim_vs_real"):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
