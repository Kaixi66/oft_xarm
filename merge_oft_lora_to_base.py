#!/usr/bin/env python3
"""Merge OpenVLA-OFT LoRA checkpoints into the OpenVLA base model.

The output directory contains a standalone HF model plus the OFT-specific
components needed by deploy.py: action head, proprio projector, processor files,
and dataset statistics. The LoRA adapter itself is intentionally not copied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


COPY_PATTERNS = (
    "*.json",
    "*.model",
    "processing_prismatic.py",
    "action_head--*.pt",
    "proprio_projector--*.pt",
    "noisy_action_projector--*.pt",
    "vision_backbone--*.pt",
)


def copy_oft_sidecars(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in COPY_PATTERNS:
        for path in src.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dst / path.name)


def register_openvla_auto_classes() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default="openvla/openvla-7b")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    adapter_dir = checkpoint_dir / "lora_adapter"

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Missing LoRA adapter directory: {adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    register_openvla_auto_classes()
    copy_oft_sidecars(checkpoint_dir, output_dir)

    print(f"Loading base model: {args.base_checkpoint}")
    base_vla = AutoModelForVision2Seq.from_pretrained(
        args.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    start = time.time()
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)

    print("Merging LoRA weights into base model...")
    merged_vla = merged_vla.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    merged_vla.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
