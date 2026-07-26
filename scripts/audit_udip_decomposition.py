"""Audit the U-DIP deformation/intensity coordinate before diffusion training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phyrd.config import load_config
from phyrd.models import build_composite_from_config
from phyrd.models.probabilistic.udip import DecompositionTargetBuilder
from scripts.train import build_dataset


def _builder_from_config(config: dict[str, object]) -> DecompositionTargetBuilder:
    model = dict(config["model"])
    probabilistic = dict(model.get("probabilistic", {}))
    params = dict(probabilistic.get("params", {}))
    return DecompositionTargetBuilder(
        downsample_factor=int(params.get("downsample_factor", 4)),
        steps=int(params.get("registration_steps", 4)),
        learning_rate=float(params.get("registration_learning_rate", 0.5)),
        smoothness_weight=float(params.get("registration_smoothness_weight", 0.05)),
        temporal_weight=float(params.get("registration_temporal_weight", 0.02)),
        magnitude_weight=float(params.get("registration_magnitude_weight", 0.001)),
        gradient_weight=float(params.get("registration_gradient_weight", 0.1)),
        max_displacement=float(params.get("max_displacement", 16.0)),
        confidence_scale=float(params.get("confidence_scale", 0.1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val_model")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = dict(config["data"])
    # CachedTrendDataset intentionally requires cache length to exactly match
    # the wrapped dataset. A small audit subset therefore computes trends
    # directly instead of opening a full-split cache with a mismatched length.
    if args.max_samples is not None:
        data_config.pop("trend_cache_dir", None)
    dataset = build_dataset(
        data_config, split=args.split, max_samples=args.max_samples
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = build_composite_from_config(
        config,
        input_frames=int(data_config["input_frames"]),
        output_frames=int(data_config["output_frames"]),
    ).to(device).eval()
    checkpoint = args.checkpoint or dict(config["model"]).get("deterministic_checkpoint")
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.deterministic.load_state_dict(payload["deterministic"], strict=True)
    model.deterministic.requires_grad_(False)
    builder = _builder_from_config(config)

    before_squared = 0.0
    after_squared = 0.0
    absolute_intensity = 0.0
    absolute_deformation = 0.0
    saturation_count = 0
    deformation_count = 0
    confidence_sum = 0.0
    confidence_count = 0
    sample_count = 0
    example: dict[str, np.ndarray] | None = None
    try:
        for batch in loader:
            history = batch["x"].to(device)
            target = batch["y"].to(device)
            cached_trend = batch.get("trend")
            if cached_trend is None:
                # Registration differentiates only with respect to its temporary
                # deformation variable. ``no_grad`` keeps the trend ordinary;
                # inference tensors cannot be saved by grid_sample backward.
                with torch.no_grad():
                    trend = model.predict_trend(history)
            else:
                trend = cached_trend.to(device)
            targets = builder(trend, target)
            before_squared += float((trend.float() - target.float()).square().sum().item())
            after_squared += float(
                (targets.warped_trend - target.float()).square().sum().item()
            )
            absolute_intensity += float(targets.intensity.abs().sum().item())
            absolute_deformation += float(targets.deformation.abs().sum().item())
            saturation_count += int(
                (targets.deformation.abs() >= builder.max_displacement * 0.999).sum().item()
            )
            deformation_count += targets.deformation.numel()
            confidence_sum += float(targets.confidence.sum().item())
            confidence_count += targets.confidence.numel()
            sample_count += history.shape[0]
            if example is None:
                example = {
                    "history": history[0].float().cpu().numpy(),
                    "target": target[0].float().cpu().numpy(),
                    "trend": trend[0].float().cpu().numpy(),
                    "warped_trend": targets.warped_trend[0].cpu().numpy(),
                    "deformation": targets.deformation[0].cpu().numpy(),
                    "intensity": targets.intensity[0].cpu().numpy(),
                    "confidence": targets.confidence[0].cpu().numpy(),
                }
    finally:
        dataset.close()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert example is not None
    np.savez_compressed(output / "example.npz", **example)
    pixels = sample_count * int(data_config["output_frames"]) * int(
        data_config["model_resolution"]
    ) ** 2
    summary = {
        "samples": sample_count,
        "mse_before": before_squared / pixels,
        "mse_after_warp": after_squared / pixels,
        "relative_mse_reduction": 1.0 - after_squared / max(before_squared, 1e-12),
        "mean_absolute_intensity": absolute_intensity / pixels,
        "mean_absolute_deformation_pixels": absolute_deformation / deformation_count,
        "deformation_saturation_fraction": saturation_count / deformation_count,
        "mean_registration_confidence": confidence_sum / confidence_count,
        "example": str(output / "example.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
