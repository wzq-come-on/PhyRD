from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phyrd.evaluation.risk import LogisticRiskCalibrator


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    positives = sorted_labels.sum().clamp_min(1.0)
    cumulative = sorted_labels.cumsum(0)
    ranks = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    precision = cumulative / ranks
    return float((precision * sorted_labels).sum().div(positives).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit independent risk calibrators on val_calib artifacts")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.input]
    if any(payload.get("split") != "val_calib" for payload in payloads):
        raise ValueError("all risk artifacts must come from val_calib")
    features = torch.cat([payload["features"].float() for payload in payloads])
    target_names = tuple(payloads[0]["targets"])
    if any(tuple(payload["targets"]) != target_names for payload in payloads):
        raise ValueError("risk artifacts have inconsistent target names")
    calibrators: dict[str, dict[str, object]] = {}
    metrics: dict[str, object] = {
        "split": "val_calib",
        "samples": int(features.shape[0]),
        "feature_names": payloads[0]["feature_names"],
    }
    for name in target_names:
        labels = torch.cat([payload["targets"][name].float() for payload in payloads])
        calibrator = LogisticRiskCalibrator(tuple(payloads[0]["feature_names"]))
        calibrator.fit(features, labels, steps=args.steps, learning_rate=args.learning_rate)
        probabilities = calibrator.predict_proba(features)
        brier = (probabilities - labels).square().mean().item()
        calibrators[name] = calibrator.state_dict()
        metrics[name] = {
            "prevalence": labels.mean().item(),
            "brier": brier,
            "auprc": average_precision(probabilities, labels.bool()),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "split": "val_calib",
            "feature_names": payloads[0]["feature_names"],
            "calibrators": calibrators,
        },
        args.output,
    )
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
