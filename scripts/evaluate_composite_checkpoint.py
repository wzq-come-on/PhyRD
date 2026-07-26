"""Evaluate a registered PhyRD composite checkpoint with a chosen ensemble size."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from phyrd.config import load_config
from phyrd.models import build_composite_from_config
from phyrd.utils import write_json
try:
    from scripts.evaluation.post_training import evaluate_post_training
    from scripts.train import build_dataset, setup_runtime
except ModuleNotFoundError:
    from evaluation.post_training import evaluate_post_training
    from train import build_dataset, setup_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="report_test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    device, rank, _local_rank, world_size = setup_runtime(config.get("device", "cuda:0"))
    data_config = dict(config["data"])
    dataset = build_dataset(data_config, split=args.split, max_samples=None)
    model = build_composite_from_config(
        config,
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "deterministic" in payload:
        model.deterministic.load_state_dict(payload["deterministic"], strict=True)
    if "diffusion" not in payload:
        raise KeyError("checkpoint does not contain diffusion weights")
    model.diffusion.load_state_dict(payload["diffusion"], strict=True)
    model.eval()
    metrics = evaluate_post_training(
        model,
        dataset,
        device=device,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ensemble_size=args.ensemble_size,
        sampling_steps=args.sampling_steps,
        checkpoint_epoch=int(payload.get("epoch", -1)),
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        split=args.split,
    )
    if rank == 0:
        assert metrics is not None
        metrics["protocol"] = f"{dataset.input_frames}to{dataset.output_frames}@{dataset.model_resolution}"
        metrics["config"] = str(Path(args.config).resolve())
        write_json(Path(args.output), metrics)
        print(metrics)
    dataset.close()
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
