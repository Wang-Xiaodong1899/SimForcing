from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-tfds-path", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/bridge_train",
        help="Destination dataset_dir (will create meta/ and data/ inside).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:2000]",
        help="tfds split spec, e.g. 'train', 'train[:1000]', 'test'.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episodes to process.",
    )
    parser.add_argument(
        "--require-video",
        action="store_true",
        default=True,
        help="Skip episodes whose <video_root>/<idx>.mp4 is missing (default).",
    )
    parser.add_argument(
        "--no-require-video",
        dest="require_video",
        action="store_false",
        help="Keep episodes even if their mp4 file does not exist.",
    )
    return parser.parse_args()


def _instr_to_str(value) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        v = arr.item()
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="replace")
        return str(v)
    if arr.dtype.kind in ("S", "O"):
        v = arr.flatten()[0]
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="replace")
        return str(v)
    raise ValueError(f"Unexpected instruction type: {type(value)}")


def _episode_to_arrays(steps: List[dict]):
    """Convert tfds steps -> (action[N,7], state[N,7])."""
    from transforms3d.euler import euler2axangle

    N = len(steps)
    action = np.zeros((N, 7), dtype=np.float32)
    state = np.zeros((N, 7), dtype=np.float32)
    for i, st in enumerate(steps):
        wv = np.asarray(st["action"]["world_vector"], dtype=np.float32)
        rd = np.asarray(st["action"]["rotation_delta"], dtype=np.float64)
        ax, ang = euler2axangle(float(rd[0]), float(rd[1]), float(rd[2]))
        rot_axangle = (np.asarray(ax, dtype=np.float32) * float(ang)).astype(np.float32)
        og = float(np.asarray(st["action"]["open_gripper"]).astype(np.float32))
        gripper = 2.0 * og - 1.0
        action[i, 0:3] = wv
        action[i, 3:6] = rot_axangle
        action[i, 6] = gripper
        s = np.asarray(st["observation"]["state"], dtype=np.float32)
        if s.shape[0] != 7:
            raise RuntimeError(f"Expected observation.state shape (7,), got {s.shape}")
        state[i] = s
    return (action, state)


def main():
    args = parse_args()
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    import tensorflow_datasets as tfds

    out_dir = Path(args.out_dir)
    meta_dir = out_dir / "meta"
    data_dir = out_dir / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading bridge tfds from {args.bridge_tfds_path}")
    builder = tfds.builder_from_directory(builder_dir=args.bridge_tfds_path)
    dset = builder.as_dataset(split=args.split)
    video_root = Path(args.video_root)
    tasks_jsonl = meta_dir / "tasks.jsonl"
    eps_jsonl = meta_dir / "episodes.jsonl"
    task_to_idx: dict[str, int] = {}
    n_kept = 0
    n_total = 0
    n_missing_video = 0
    with eps_jsonl.open("w", encoding="utf-8") as eps_f:
        for ep_idx, episode in enumerate(dset):
            n_total += 1
            if args.max_episodes is not None and ep_idx >= args.max_episodes:
                break
            video_path = video_root / f"{ep_idx}.mp4"
            if args.require_video and (not video_path.exists()):
                n_missing_video += 1
                continue
            steps = list(episode["steps"])
            if len(steps) == 0:
                continue
            instruction = _instr_to_str(
                steps[0]["observation"]["natural_language_instruction"]
            )
            if instruction not in task_to_idx:
                task_to_idx[instruction] = len(task_to_idx)
            task_index = task_to_idx[instruction]
            action, state = _episode_to_arrays(steps)
            np.savez_compressed(
                data_dir / f"episode_{ep_idx}.npz", action=action, state=state
            )
            rec = {
                "episode_index": ep_idx,
                "num_steps": int(action.shape[0]),
                "instruction": instruction,
                "task_index": task_index,
                "video_path": str(video_path),
            }
            eps_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept % 200 == 0:
                print(f"  processed {n_kept} kept / {ep_idx + 1} scanned")
    with tasks_jsonl.open("w", encoding="utf-8") as f:
        for task, idx in sorted(task_to_idx.items(), key=lambda kv: kv[1]):
            f.write(
                json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n"
            )
    print("Done.")
    print(f"  total scanned   : {n_total}")
    print(f"  kept            : {n_kept}")
    print(f"  missing video   : {n_missing_video}")
    print(f"  unique tasks    : {len(task_to_idx)}")
    print(f"  out dir         : {out_dir}")


if __name__ == "__main__":
    main()
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
