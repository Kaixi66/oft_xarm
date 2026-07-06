#!/usr/bin/env python3
"""Convert the local XArm LeRobot parquet datasets to OpenVLA-OFT RLDS/TFDS.

Default inputs:
  /root/angli/hf_cache/lerobot/lab/xarm_setting1_51
  /root/angli/hf_cache/lerobot/lab/xarm_setting2_51

Default output:
  /workspace/kaixi/RealWorld/rlds_data/

The TFDS dataset name intentionally matches OpenVLA-OFT's existing XArm config:
  utokyo_xarm_pick_and_place_converted_externally_to_rlds
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np


DATASET_NAME = "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
DEFAULT_DATASET_ROOTS = (
    Path("/root/angli/hf_cache/lerobot/lab/xarm_setting1_51"),
    Path("/root/angli/hf_cache/lerobot/lab/xarm_setting2_51"),
)
DEFAULT_TFDS_DATA_DIR = Path("/workspace/kaixi/RealWorld/rlds_data")


def import_runtime_deps() -> tuple[Any, Any, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    missing = []
    modules = {}
    for module_name, package_name in (
        ("pyarrow.parquet", "pyarrow"),
        ("tensorflow_datasets", "tensorflow-datasets"),
        ("PIL.Image", "pillow"),
    ):
        try:
            modules[module_name] = __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        pkgs = " ".join(sorted(set(missing)))
        raise SystemExit(
            f"Missing Python dependencies: {pkgs}\n"
            f"Install them in this Python environment with:\n"
            f"  {sys.executable} -m pip install {pkgs}"
        )
    return modules["pyarrow.parquet"], modules["tensorflow_datasets"], modules["PIL.Image"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    for row in read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        idx = row.get("task_index", row.get("index"))
        if idx is not None and row.get("task"):
            tasks[int(idx)] = str(row["task"])
    return tasks


def collect_episode_paths(dataset_roots: Iterable[Path]) -> list[tuple[Path, Path]]:
    episodes = []
    for root in dataset_roots:
        root = root.resolve()
        data_dir = root / "data" / "chunk-000"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Missing LeRobot parquet directory: {data_dir}")
        for path in sorted(data_dir.glob("episode_*.parquet")):
            episodes.append((root, path))
    if not episodes:
        raise FileNotFoundError("No LeRobot episode_*.parquet files found")
    return episodes


def split_episodes(
    episodes: list[tuple[Path, Path]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")
    if val_ratio == 0.0 or len(episodes) <= 1:
        return episodes, []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(episodes) * val_ratio)))
    n_val = min(n_val, len(episodes) - 1)
    val_indices = set(indices[:n_val].tolist())

    train = [episode for idx, episode in enumerate(episodes) if idx not in val_indices]
    val = [episode for idx, episode in enumerate(episodes) if idx in val_indices]
    return train, val


def filter_noop_rows(
    rows: list[dict[str, Any]],
    *,
    pos_thresh: float,
    rot_thresh: float,
) -> list[dict[str, Any]]:
    """Drop no-op frames: negligible translation and rotation with no gripper
    change relative to the previous frame. Mirrors the `remove_zero` filtering
    used for the pi0 checkpoints; without it ~30% of the frames are idle and
    the policy learns to stand still."""
    kept = []
    prev_gripper = float(np.asarray(rows[0]["actions"], dtype=np.float32)[6])
    for row in rows:
        action = np.asarray(row["actions"], dtype=np.float32)
        gripper = float(action[6])
        gripper_changed = abs(gripper - prev_gripper) > 1e-6
        moving = (
            float(np.linalg.norm(action[:3])) >= pos_thresh
            or float(np.linalg.norm(action[3:6])) >= rot_thresh
        )
        if moving or gripper_changed:
            kept.append(row)
        prev_gripper = gripper
    return kept


def image_record_to_array(record: dict[str, Any], dataset_root: Path, image_module: Any) -> np.ndarray:
    if record.get("bytes"):
        image = image_module.open(BytesIO(record["bytes"]))
    elif record.get("path"):
        path = Path(record["path"])
        if not path.is_absolute():
            path = dataset_root / path
        image = image_module.open(path)
    else:
        raise ValueError("Image record has neither bytes nor path")

    with image:
        image = image.convert("RGB")
        if image.size != (224, 224):
            resampling = getattr(image_module, "Resampling", image_module).BICUBIC
            image = image.resize((224, 224), resample=resampling)
        return np.asarray(image, dtype=np.uint8)


def episode_to_example(
    *,
    dataset_root: Path,
    parquet_path: Path,
    pq: Any,
    image_module: Any,
    task_map: dict[int, str],
    max_frames: int | None,
    noop_thresholds: tuple[float, float] | None,
) -> dict[str, Any]:
    rows = pq.read_table(
        parquet_path,
        columns=["image", "wrist_image", "state", "actions", "task_index"],
    ).to_pylist()
    if max_frames is not None:
        rows = rows[:max_frames]
    if not rows:
        raise ValueError(f"Empty episode: {parquet_path}")

    if noop_thresholds is not None:
        pos_thresh, rot_thresh = noop_thresholds
        kept = filter_noop_rows(rows, pos_thresh=pos_thresh, rot_thresh=rot_thresh)
        print(f"[filter] {dataset_root.name}/{parquet_path.name}: kept {len(kept)}/{len(rows)} frames")
        if len(kept) >= 2:
            rows = kept
        else:
            print(f"[filter] {parquet_path.name}: too few frames left, keeping episode unfiltered")

    steps = []
    for frame_idx, row in enumerate(rows):
        state = np.asarray(row["state"], dtype=np.float32)
        action = np.asarray(row["actions"], dtype=np.float32)
        if state.shape != (6,):
            raise ValueError(f"{parquet_path}: expected state shape (6,), got {state.shape}")
        if action.shape != (7,):
            raise ValueError(f"{parquet_path}: expected action shape (7,), got {action.shape}")

        task = task_map.get(int(row.get("task_index", 0)), dataset_root.name)
        steps.append(
            {
                "observation": {
                    "image": image_record_to_array(row["image"], dataset_root, image_module),
                    "hand_image": image_record_to_array(row["wrist_image"], dataset_root, image_module),
                    "end_effector_pose": state,
                },
                "action": action,
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if frame_idx == len(rows) - 1 else 0.0),
                "is_first": np.bool_(frame_idx == 0),
                "is_last": np.bool_(frame_idx == len(rows) - 1),
                "is_terminal": np.bool_(frame_idx == len(rows) - 1),
                "language_instruction": task,
            }
        )

    return {
        "steps": steps,
        "episode_metadata": {
            "file_path": str(parquet_path),
        },
    }


def make_builder_class(
    *,
    tfds: Any,
    pq: Any,
    image_module: Any,
    train_episodes: list[tuple[Path, Path]],
    val_episodes: list[tuple[Path, Path]],
    tasks_by_root: dict[Path, dict[int, str]],
    max_frames: int | None,
    noop_thresholds: tuple[float, float] | None,
) -> type:
    class UtokyoXarmPickAndPlaceConvertedExternallyToRlds(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "Local LeRobot XArm conversion."}
        pkg_dir_path = Path(__file__).resolve().parent

        def _info(self) -> Any:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "image": tfds.features.Image(
                                            shape=(224, 224, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "hand_image": tfds.features.Image(
                                            shape=(224, 224, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "end_effector_pose": tfds.features.Tensor(shape=(6,), dtype=np.float32),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(),
                            }
                        ),
                    }
                ),
                description="Local XArm LeRobot datasets converted for OpenVLA-OFT.",
            )

        def _split_generators(self, dl_manager: Any) -> dict[str, Any]:
            del dl_manager
            splits = {"train": self._generate_examples(train_episodes)}
            if val_episodes:
                splits["val"] = self._generate_examples(val_episodes)
            return splits

        def _generate_examples(self, episodes: list[tuple[Path, Path]]):
            for dataset_root, parquet_path in episodes:
                key = f"{dataset_root.name}-{parquet_path.stem}"
                yield key, episode_to_example(
                    dataset_root=dataset_root,
                    parquet_path=parquet_path,
                    pq=pq,
                    image_module=image_module,
                    task_map=tasks_by_root[dataset_root],
                    max_frames=max_frames,
                    noop_thresholds=noop_thresholds,
                )

    return UtokyoXarmPickAndPlaceConvertedExternallyToRlds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        type=Path,
        dest="dataset_roots",
        help="LeRobot dataset root. Can be passed multiple times.",
    )
    parser.add_argument("--tfds-data-dir", type=Path, default=DEFAULT_TFDS_DATA_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument(
        "--filter-noops",
        action="store_true",
        help="Drop frames with near-zero motion and unchanged gripper (mirrors pi0 remove_zero).",
    )
    parser.add_argument(
        "--noop-pos-thresh",
        type=float,
        default=0.02,
        help="Translation threshold in cm/step below which a frame counts as idle.",
    )
    parser.add_argument(
        "--noop-rot-thresh",
        type=float,
        default=0.002,
        help="Rotation threshold in rad/step below which a frame counts as idle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pq, tfds, image_module = import_runtime_deps()

    dataset_roots = tuple(args.dataset_roots or DEFAULT_DATASET_ROOTS)
    episodes = collect_episode_paths(dataset_roots)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    train_episodes, val_episodes = split_episodes(episodes, val_ratio=args.val_ratio, seed=args.seed)
    tasks_by_root = {root.resolve(): load_tasks(root.resolve()) for root in dataset_roots}

    output_dataset_dir = args.tfds_data_dir / DATASET_NAME
    if args.overwrite and output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)
    args.tfds_data_dir.mkdir(parents=True, exist_ok=True)

    noop_thresholds = (args.noop_pos_thresh, args.noop_rot_thresh) if args.filter_noops else None

    builder_cls = make_builder_class(
        tfds=tfds,
        pq=pq,
        image_module=image_module,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        tasks_by_root=tasks_by_root,
        max_frames=args.max_frames_per_episode,
        noop_thresholds=noop_thresholds,
    )
    builder = builder_cls(data_dir=str(args.tfds_data_dir))
    if builder.name != DATASET_NAME:
        raise RuntimeError(f"Unexpected TFDS name {builder.name}; expected {DATASET_NAME}")

    print(f"[info] dataset: {builder.name}")
    print(f"[info] data root: {args.tfds_data_dir}")
    print(f"[info] train episodes: {len(train_episodes)}")
    print(f"[info] val episodes: {len(val_episodes)}")
    if noop_thresholds is not None:
        print(f"[info] no-op filter: pos<{noop_thresholds[0]}cm and rot<{noop_thresholds[1]}rad (gripper changes kept)")
    else:
        print("[info] no-op filter: OFF (pass --filter-noops to enable)")
    builder.download_and_prepare()
    print(f"[done] wrote: {builder.data_dir}")


if __name__ == "__main__":
    main()
