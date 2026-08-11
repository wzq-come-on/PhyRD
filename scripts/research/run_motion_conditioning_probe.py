"""Run a controlled real-data probe of candidate motion conditions.

This is a representation-selection experiment, not a replacement for a full DiT
ablation.  Every variant shares the same base residual stem and optimizer.  The
only difference is the optional GlobalNet or Tora-style motion branch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyrd.config import load_config
from phyrd.data.trend_cache import CachedTrendDataset
from phyrd.research.motion_conditioning import build_motion_probe
from scripts.train import build_dataset


THRESHOLDS = (16, 74, 133, 160, 181, 219)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--variant", choices=("baseline", "globalnet", "tora_adaln"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shared_state_fingerprint(model: torch.nn.Module) -> float:
    return float(sum(value.detach().double().sum() for key, value in model.state_dict().items() if key.startswith("stem.")))


def zero_init_equivalence(variant: str, hidden_size: int, patch_size: int) -> float:
    seed_everything(123)
    baseline = build_motion_probe("baseline", hidden_size=hidden_size, patch_size=patch_size).eval()
    seed_everything(123)
    candidate = build_motion_probe(variant, hidden_size=hidden_size, patch_size=patch_size).eval()
    history = torch.rand(2, 5, 1, 32, 32)
    trend = torch.rand(2, 20, 1, 32, 32)
    with torch.no_grad():
        reference = baseline(history, trend)
        output = candidate(history, trend)
    return float((reference - output).abs().max())


def motion_gradient_check(variant: str, hidden_size: int, patch_size: int) -> bool:
    if variant == "baseline":
        return True
    seed_everything(321)
    candidate = build_motion_probe(
        variant, hidden_size=hidden_size, patch_size=patch_size
    )
    projection = (
        candidate.context_projection
        if variant == "globalnet"
        else candidate.to_scale_shift
    )
    torch.nn.init.normal_(projection.weight, std=1e-3)
    history = torch.rand(2, 5, 1, 32, 32)
    trend = torch.rand(2, 20, 1, 32, 32)
    candidate(history, trend).square().mean().backward()
    gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if not name.startswith("stem.") and parameter.grad is not None
    ]
    return bool(gradients) and any(
        bool(torch.count_nonzero(gradient)) for gradient in gradients
    )


def make_loader(
    config: dict,
    cache_dir: Path,
    split: str,
    samples: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[CachedTrendDataset, DataLoader]:
    dataset = build_dataset(dict(config["data"]), split=split, max_samples=samples)
    cached = CachedTrendDataset(dataset, cache_dir / f"{split}.npy")
    loader = DataLoader(
        cached,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )
    return cached, loader


def batch_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["x"].to(device, non_blocking=True),
        batch["y"].to(device, non_blocking=True),
        batch["trend"].to(device, non_blocking=True),
    )


def weighted_residual_loss(
    prediction: torch.Tensor, target: torch.Tensor, trend: torch.Tensor
) -> torch.Tensor:
    residual = target - trend
    weight = 1.0 + 2.0 * torch.sigmoid((target.detach() - 74.0 / 255.0) / 0.05)
    error = prediction - residual
    return (weight * error.square()).mean() + 0.10 * (weight * error.abs()).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {
        "samples": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "pixels": 0.0,
        "temporal_error": 0.0,
        "temporal_pixels": 0.0,
    }
    contingency = {threshold: [0.0, 0.0, 0.0] for threshold in THRESHOLDS}
    trend_squared = trend_absolute = 0.0
    for batch in loader:
        history, target, trend = batch_tensors(batch, device)
        prediction = (trend + model(history, trend)).clamp(0.0, 1.0)
        error = prediction - target
        trend_error = trend - target
        sums["samples"] += target.shape[0]
        sums["squared_error"] += float(error.square().sum())
        sums["absolute_error"] += float(error.abs().sum())
        sums["pixels"] += float(error.numel())
        trend_squared += float(trend_error.square().sum())
        trend_absolute += float(trend_error.abs().sum())
        previous_prediction = torch.cat((history[:, -1:], prediction[:, :-1]), dim=1)
        previous_target = torch.cat((history[:, -1:], target[:, :-1]), dim=1)
        temporal_error = (prediction - previous_prediction) - (target - previous_target)
        sums["temporal_error"] += float(temporal_error.abs().sum())
        sums["temporal_pixels"] += float(temporal_error.numel())
        pred_255 = prediction * 255.0
        target_255 = target * 255.0
        for threshold in THRESHOLDS:
            pred_mask = pred_255 >= threshold
            target_mask = target_255 >= threshold
            hits = float((pred_mask & target_mask).sum())
            misses = float((~pred_mask & target_mask).sum())
            false_alarms = float((pred_mask & ~target_mask).sum())
            current = contingency[threshold]
            current[0] += hits
            current[1] += misses
            current[2] += false_alarms
    metrics = {
        "samples": int(sums["samples"]),
        "mse_normalized": sums["squared_error"] / sums["pixels"],
        "mae_normalized": sums["absolute_error"] / sums["pixels"],
        "trend_mse_normalized": trend_squared / sums["pixels"],
        "trend_mae_normalized": trend_absolute / sums["pixels"],
        "temporal_delta_mae": sums["temporal_error"] / sums["temporal_pixels"],
    }
    csi_values = []
    for threshold, (hits, misses, false_alarms) in contingency.items():
        denominator = hits + misses + false_alarms
        value = hits / denominator if denominator else 0.0
        metrics[f"csi_{threshold}"] = value
        csi_values.append(value)
    metrics["csi_mean"] = float(sum(csi_values) / len(csi_values))
    return metrics


def main() -> None:
    args = parse_args()
    equivalence_error = zero_init_equivalence(args.variant, args.hidden_size, args.patch_size)
    if equivalence_error > 1e-6:
        raise RuntimeError(f"zero-init equivalence failed: max_abs_error={equivalence_error}")
    gradient_check = motion_gradient_check(
        args.variant, args.hidden_size, args.patch_size
    )
    if not gradient_check:
        raise RuntimeError("motion branch did not receive a non-zero gradient")
    if args.check_only:
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "zero_init_max_abs_error": equivalence_error,
                    "motion_gradient_check": gradient_check,
                }
            )
        )
        return

    seed_everything(args.seed)
    device = torch.device(args.device)
    config = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_dataset, train_loader = make_loader(
        config,
        cache_dir,
        "train",
        args.train_samples,
        args.batch_size,
        args.num_workers,
        True,
    )
    val_dataset, val_loader = make_loader(
        config,
        cache_dir,
        "val_model",
        args.val_samples,
        args.batch_size,
        args.num_workers,
        False,
    )
    model = build_motion_probe(
        args.variant, hidden_size=args.hidden_size, patch_size=args.patch_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scaler_enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    train_iterator = iter(train_loader)
    losses: list[dict[str, float]] = []
    started = time.time()
    model.train()
    try:
        for step in range(1, args.steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            history, target, trend = batch_tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda",
            ):
                residual = model(history, trend)
                loss = weighted_residual_loss(residual, target, trend)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if step == 1 or step % 25 == 0 or step == args.steps:
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
                losses.append(record)
                print(json.dumps(record), flush=True)
        metrics = evaluate(model, val_loader, device)
        result = {
            "status": "complete",
            "variant": args.variant,
            "seed": args.seed,
            "steps": args.steps,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "shared_state_fingerprint": shared_state_fingerprint(model),
            "zero_init_max_abs_error": equivalence_error,
            "motion_gradient_check": gradient_check,
            "elapsed_seconds": time.time() - started,
            "loss_trace": losses,
            "metrics": metrics,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        torch.save(
            {"state_dict": model.state_dict(), "result": result},
            output_dir / "checkpoint_last.pt",
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    main()
