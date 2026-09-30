import logging
import os
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _optional_float(value) -> float | None:
    """Return ``float(value)`` or ``None`` when the config key is unset/null."""
    return None if value is None else float(value)


def create_action_conditioned(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    action_dropout_prob: float = 0.1,
    first_frame_dropout_prob: float = 0.1,
    loss_weight_sim: float = 1.0,
    loss_weight_real: float = 1.0,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.action_conditioned import ActionConditionedModel

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(
            f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}"
        )
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(
            f"`video_scheduler` must be dict-like, got {type(video_scheduler)}"
        )
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        action_scheduler = {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must be dict-like, got {type(action_scheduler)}"
        )
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    return ActionConditionedModel.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(
            action_scheduler.get("num_train_timesteps", 1000)
        ),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=0.0,
        action_dropout_prob=float(action_dropout_prob),
        first_frame_dropout_prob=float(first_frame_dropout_prob),
        loss_weight_sim=float(loss_weight_sim),
        loss_weight_real=float(loss_weight_real),
    )


def create_simforcing(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    frozen_video_expert_checkpoint: str | None = None,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    frozen_action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    sim_cond_init_scale: float = 0.0,
    action_dropout_prob: float = 0.0,
    first_frame_dropout_prob: float = 0.0,
    sim_latent_noise_prob: float = 0.4,
    sim_latent_noise_alpha_range=(0.5, 0.9),
    sim_latent_noise_sigma_range=(0.1, 0.3),
    delta_dropout_prob: float = 0.0,
    sim_cond_on_syn_branch: bool = False,
    sim_cond_prob: float = 0.5,
    delta_lambda_cond_on: float = 1.0,
    sim_cond_drift_prob: float = 0.0,
    sim_cond_drift_scale_range=(0.02, 0.08),
    delta_spatial_pool: int = 1,
    delta_cosine: bool = False,
    delta_sigma_weight_mode: str = "fm",
    sim_cond_cfg_scale: float = 1.0,
    build_frozen_teacher: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.simforcing import SimForcing

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(
            f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}"
        )
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )
    if isinstance(frozen_action_dit_config, DictConfig):
        frozen_action_dit_config = OmegaConf.to_container(
            frozen_action_dit_config, resolve=True
        )
    if frozen_action_dit_config is not None and (
        not isinstance(frozen_action_dit_config, dict)
    ):
        raise ValueError(
            f"`frozen_action_dit_config` must resolve to a dict or be None, got {type(frozen_action_dit_config)}"
        )
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(
            f"`video_scheduler` must be dict-like, got {type(video_scheduler)}"
        )
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        action_scheduler = {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
        }
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must be dict-like, got {type(action_scheduler)}"
        )
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    return SimForcing.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(
            action_scheduler.get("num_train_timesteps", 1000)
        ),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=0.0,
        loss_lambda_real=float(loss.get("lambda_real", 1.0)),
        loss_lambda_syn=float(loss.get("lambda_syn", 0.1)),
        loss_lambda_delta=float(loss.get("lambda_delta", 0.5)),
        loss_lambda_diffusion_real=_optional_float(loss.get("lambda_diffusion_real")),
        loss_lambda_diffusion_syn=_optional_float(loss.get("lambda_diffusion_syn")),
        loss_lambda_delta_real=_optional_float(loss.get("lambda_delta_real")),
        loss_lambda_delta_syn=_optional_float(loss.get("lambda_delta_syn")),
        sim_cond_init_scale=float(sim_cond_init_scale),
        action_dropout_prob=float(action_dropout_prob),
        first_frame_dropout_prob=float(first_frame_dropout_prob),
        sim_latent_noise_prob=float(sim_latent_noise_prob),
        sim_latent_noise_alpha_range=tuple(
            (float(v) for v in sim_latent_noise_alpha_range)
        ),
        sim_latent_noise_sigma_range=tuple(
            (float(v) for v in sim_latent_noise_sigma_range)
        ),
        delta_dropout_prob=float(delta_dropout_prob),
        sim_cond_on_syn_branch=bool(sim_cond_on_syn_branch),
        sim_cond_prob=float(sim_cond_prob),
        delta_lambda_cond_on=float(delta_lambda_cond_on),
        sim_cond_drift_prob=float(sim_cond_drift_prob),
        sim_cond_drift_scale_range=tuple(
            (float(v) for v in sim_cond_drift_scale_range)
        ),
        delta_spatial_pool=int(delta_spatial_pool),
        delta_cosine=bool(delta_cosine),
        delta_sigma_weight_mode=str(delta_sigma_weight_mode),
        sim_cond_cfg_scale=float(sim_cond_cfg_scale),
        frozen_video_expert_checkpoint=(
            None
            if frozen_video_expert_checkpoint is None
            else str(frozen_video_expert_checkpoint)
        ),
        frozen_action_dit_config=frozen_action_dit_config,
        build_frozen_teacher=bool(build_frozen_teacher),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info(
            "Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats
        )
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return (train_ds, val_ds)


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)
    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)
    trainer = Wan22Trainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    trainer.train()
