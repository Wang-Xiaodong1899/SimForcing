"""Generate a Bridge rollout from a saved baseline or SimForcing checkpoint."""

import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from simforcing.models.wan22 import SimForcing
from simforcing.utils.pytorch_utils import set_global_seed
from simforcing.utils.video_io import save_mp4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["bridge_real_act_cond_224_1e-4", "bridge_real_simforcing_224_1e-4"],
        default="bridge_real_simforcing_224_1e-4",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model-only .pt checkpoint."
    )
    parser.add_argument(
        "--norm-stats",
        required=True,
        help="dataset_stats.json saved by the training run.",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-cfg", type=float, default=1.0)
    parser.add_argument("--action-cfg", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/prediction.mp4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. data.val.dataset_dirs=[/path/to/data].",
    )
    args = parser.parse_args()
    for path in (args.checkpoint, args.norm_stats):
        if not Path(path).is_file():
            parser.error(f"File does not exist: {path}")
    set_global_seed(args.seed)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base="1.3",
    ):
        cfg = compose(
            config_name="train", overrides=[f"task={args.task}", *args.overrides]
        )
    # Complete weights come from the model checkpoint; no teacher is needed.
    cfg.model.skip_dit_load_from_pretrain = True
    if "build_frozen_teacher" in cfg.model:
        cfg.model.build_frozen_teacher = False
    model = instantiate(
        cfg.model,
        device=args.device,
        model_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    model.load_checkpoint(args.checkpoint)
    dataset = instantiate(cfg.data[args.split], pretrained_norm_stats=args.norm_stats)
    item = dataset[args.index]
    common = dict(
        prompt=None,
        action=item["action"],
        proprio=item["proprio"][0],
        context=item["context"],
        context_mask=item["context_mask"],
        num_inference_steps=args.steps,
        action_cfg_scale=args.action_cfg,
        seed=args.seed,
    )
    if isinstance(model, SimForcing):
        # Use a predicted simulation trajectory as the condition.
        sim_args = dict(common)
        sim_args.pop("prompt")
        sim_args["context"] = item.get("context_syn", item["context"])
        sim_args["context_mask"] = item.get("context_mask_syn", item["context_mask"])
        num_frames = item["video_real"].shape[1]
        sim_latents = model.infer_sim_latents(
            input_image=item["video_syn"][:, 0], num_video_frames=num_frames, **sim_args
        )
        result = model.infer_video(
            input_image_syn=None,
            input_image_real=item["video_real"][:, 0],
            sim_latents=sim_latents,
            num_video_frames=num_frames,
            sim_cond_cfg_scale=args.sim_cfg,
            **common,
        )
    else:
        result = model.infer_video(
            input_image=item["video"][:, 0],
            num_video_frames=item["video"].shape[1],
            **common,
        )
    save_mp4(result["video"], args.output, fps=args.fps)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
