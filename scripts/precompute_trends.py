"""Precompute frozen deterministic forecasts for probability-only training.

The resulting ``data/trend_cache/<name>/<split>.npy`` files are intentionally
ignored by Git.  They are tied to one deterministic checkpoint and one data
protocol; the sidecar JSON records that provenance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from phyrd.config import load_config
from phyrd.models import build_composite_from_config
from scripts.train import build_dataset


class RankStridedSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - 1 - self.rank) // self.world_size + 1)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cache_split(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    split: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    overwrite: bool,
    rank: int,
    world_size: int,
) -> None:
    data_config = dict(config["data"])
    dataset = build_dataset(data_config, split=split, max_samples=max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.npy"
    metadata_path = output_path.with_suffix(".json")
    if (output_path.exists() or metadata_path.exists()) and (world_size == 1 or rank == 0):
        if not overwrite:
            raise FileExistsError(
                f"cache exists for split={split}: {output_path}; use --overwrite or a new directory"
            )
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    model = build_composite_from_config(
        config,
        input_frames=int(data_config["input_frames"]),
        output_frames=int(data_config["output_frames"]),
    ).to(device).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "deterministic" not in payload:
        raise KeyError(f"checkpoint has no deterministic state: {checkpoint}")
    model.deterministic.load_state_dict(payload["deterministic"], strict=True)
    model.deterministic.requires_grad_(False)
    model.deterministic.eval()

    count = len(dataset)
    frames = int(data_config["output_frames"])
    resolution = int(data_config["model_resolution"])
    if world_size > 1:
        if rank == 0:
            np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float16,
                shape=(count, frames, 1, resolution, resolution),
            ).flush()
        dist.barrier()
        cache = np.lib.format.open_memmap(output_path, mode="r+")
        sampler = RankStridedSampler(count, rank, world_size)
    else:
        cache = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float16,
            shape=(count, frames, 1, resolution, resolution),
        )
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    offset = 0
    sample_indices = list(iter(sampler)) if sampler is not None else list(range(count))
    precision = str(config.get("optimization", {}).get("precision", "bf16"))
    with torch.inference_mode():
        for batch in loader:
            history = batch["x"].to(device, non_blocking=True)
            with _autocast(device, precision):
                trend = model.predict_trend(history).clamp(0.0, 1.0)
            values = trend.float().cpu().numpy().astype(np.float16, copy=False)
            positions = sample_indices[offset : offset + values.shape[0]]
            cache[positions] = values
            offset += values.shape[0]
    cache.flush()
    dataset.close()
    if world_size > 1:
        dist.barrier()
        if rank != 0:
            return
    metadata = {
        "cache_format": "numpy_memmap_float16",
        "cache_path": str(output_path),
        "split": split,
        "samples": count,
        "shape": list(cache.shape),
        "dataset": {
            "format": data_config.get("format"),
            "root": data_config.get("root"),
            "input_frames": data_config.get("input_frames"),
            "output_frames": data_config.get("output_frames"),
            "window_start_index": data_config.get("window_start_index"),
            "model_resolution": data_config.get("model_resolution"),
            "spatial_preprocess": data_config.get("spatial_preprocess"),
        },
        "deterministic": config["model"].get("deterministic"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "dtype": "float16",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"split": split, "samples": count, "output": str(output_path)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute frozen deterministic trend caches")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val_model"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed trend precomputation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        for split in args.splits:
            _cache_split(
                config=config,
                checkpoint=checkpoint,
                split=str(split),
                output_dir=Path(args.output_dir),
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
                rank=rank,
                world_size=world_size,
            )
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
