from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from phyrd.config import load_config
from phyrd.data import DiffCastH5Dataset, SEVIRDataset
from phyrd.models import PhyRDModel


def _dataset(data: dict[str, Any], split: str, max_samples: int):
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


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    return float((left * right).mean().div(denominator.clamp_min(1e-12)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check full residual sampling for collapse")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    data = dict(config["data"])
    model_config = dict(config["model"])
    indices = None
    if args.sample_indices:
        indices = [int(item) for item in args.sample_indices.split(",") if item.strip()]
    dataset = _dataset(data, args.split, None if indices is not None else args.max_samples)
    if indices is not None:
        if not indices or min(indices) < 0 or max(indices) >= len(dataset):
            raise ValueError("sample indices are empty or outside the selected split")
        loader_dataset = Subset(dataset, indices)
    else:
        loader_dataset = dataset
    loader = DataLoader(loader_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = PhyRDModel(
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
        base_channels=int(model_config["base_channels"]),
        diffusion_steps=int(model_config["diffusion_steps"]),
        freeze_deterministic=True,
        deterministic=dict(model_config["deterministic"]),
        diffusion=dict(model_config.get("diffusion", {})),
    ).to(device)
    deterministic_payload = torch.load(
        model_config["deterministic_checkpoint"], map_location="cpu", weights_only=False
    )
    residual_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.deterministic.load_state_dict(deterministic_payload["deterministic"])
    model.diffusion.load_state_dict(residual_payload["diffusion"])
    model.eval()

    records: list[dict[str, float]] = []
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            history = batch["x"].to(device)
            target = batch["y"].to(device)
            trend = model.predict_trend(history)
            ensemble = model.sample(
                history, ensemble_size=1, sampling_steps=args.sampling_steps
            )
            prediction = ensemble[:, 0]
            all_predictions.append(prediction.detach().cpu())
            all_targets.append(target.detach().cpu())
            predicted_residual = prediction - trend
            target_residual = target - trend
            records.append(
                {
                    "trend_mae": float((trend - target).abs().mean().item()),
                    "prediction_mae": float((prediction - target).abs().mean().item()),
                    "target_residual_mean": float(target_residual.mean().item()),
                    "target_residual_std": float(target_residual.std().item()),
                    "predicted_residual_mean": float(predicted_residual.mean().item()),
                    "predicted_residual_std": float(predicted_residual.std().item()),
                    "prediction_target_correlation": _correlation(prediction, target),
                    "low_saturation": float((prediction <= 1e-6).float().mean().item()),
                    "high_saturation": float((prediction >= 1.0 - 1e-6).float().mean().item()),
                }
            )
    summary = {
        key: sum(record[key] for record in records) / len(records) for key in records[0]
    }
    summary["prediction_target_correlation"] = _correlation(
        torch.cat(all_predictions), torch.cat(all_targets)
    )
    collapse_gate = {
        "finite": all(
            torch.isfinite(torch.tensor(list(record.values()))).all() for record in records
        ),
        "low_saturation_below_95pct": summary["low_saturation"] < 0.95,
        "positive_correlation": summary["prediction_target_correlation"] > 0.0,
        "residual_std_below_10x_target": summary["predicted_residual_std"]
        < 10.0 * max(summary["target_residual_std"], 1e-8),
    }
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples": len(records),
        "sample_indices": indices,
        "sampling_steps": args.sampling_steps,
        "summary": summary,
        "collapse_gate": collapse_gate,
        "passed": all(collapse_gate.values()),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    dataset.close()
    if not result["passed"]:
        raise SystemExit("residual stability gate failed")


if __name__ == "__main__":
    main()
