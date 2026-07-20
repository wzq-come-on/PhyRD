from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from phyrd.config import load_config
from phyrd.data import DiffCastH5Dataset, SEVIRDataset
from phyrd.models import build_backbone, checkpoint_backbone_spec


def _dataset(data: dict[str, Any], split: str, max_samples: int | None):
    kwargs = {
        "input_frames": int(data.get("input_frames", 13)),
        "output_frames": int(data.get("output_frames", 12)),
        "window_start_index": int(data.get("window_start_index", 12)),
        "model_resolution": int(data.get("model_resolution", 384)),
        "spatial_preprocess": str(data.get("spatial_preprocess", "none")),
        "max_samples": max_samples,
    }
    if str(data.get("format", "catalog")) == "diffcast_h5":
        return DiffCastH5Dataset(data["root"], split, **kwargs)
    return SEVIRDataset(data["root"], split, **kwargs)


def _runtime(device_name: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://")
        return torch.device("cuda", local_rank), rank, world_size
    return torch.device(device_name), rank, world_size


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate train-split per-lead residual statistics for frozen SDIR"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data = dict(config["data"])
    model_config = dict(config["model"])
    checkpoint_path = Path(
        args.checkpoint or str(model_config.get("deterministic_checkpoint", ""))
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"deterministic checkpoint not found: {checkpoint_path}")
    device, rank, world_size = _runtime(str(config.get("device", "cuda:0")))
    dataset = _dataset(data, args.split, args.max_samples)
    loader_dataset = (
        Subset(dataset, range(rank, len(dataset), world_size)) if world_size > 1 else dataset
    )
    loader = DataLoader(
        loader_dataset,
        batch_size=int(data["batch_size"]),
        shuffle=False,
        num_workers=int(data["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(data["num_workers"]) > 0,
    )

    deterministic_spec = dict(model_config["deterministic"])
    model = build_backbone(
        str(deterministic_spec["name"]),
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
        params=dict(deterministic_spec["params"]),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_backbone_spec(payload.get("protocol", {})) != deterministic_spec:
        raise ValueError("checkpoint deterministic protocol does not match the config")
    model.load_state_dict(payload["deterministic"])
    model.eval().requires_grad_(False)

    leads = dataset.output_frames
    total = torch.zeros(leads, device=device, dtype=torch.float64)
    total_square = torch.zeros_like(total)
    count = torch.zeros(leads, device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    precision = str(config.get("optimization", {}).get("precision", "fp32"))
    with torch.inference_mode():
        for batch in loader:
            history = batch["x"].to(device, non_blocking=True)
            target = batch["y"].to(device, non_blocking=True)
            with _autocast(device, precision):
                trend = model(history)
            residual = (target - trend).double()
            total += residual.sum(dim=(0, 2, 3, 4))
            total_square += residual.square().sum(dim=(0, 2, 3, 4))
            pixels = residual.shape[0] * residual.shape[2] * residual.shape[3] * residual.shape[4]
            count += pixels
            sample_count += residual.shape[0]

    if world_size > 1:
        for value in (total, total_square, count, sample_count):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    center = total / count
    variance = (total_square / count - center.square()).clamp_min(0.0)
    scale = variance.sqrt().clamp_min(1e-4)
    if rank == 0:
        result = {
            "source": "train-split frozen deterministic residuals",
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "samples": int(sample_count.item()),
            "center": center.cpu().tolist(),
            "scale": scale.cpu().tolist(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    dataset.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
