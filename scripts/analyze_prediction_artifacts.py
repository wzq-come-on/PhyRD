from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def sampled(values: np.ndarray, limit: int = 2_000_000) -> np.ndarray:
    flat = values.reshape(-1)
    stride = max(1, flat.size // limit)
    return flat[::stride].astype(np.float64, copy=False)


def distribution(values: np.ndarray) -> dict[str, Any]:
    sample = sampled(values)
    quantiles = np.quantile(sample, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "mean": float(sample.mean()),
        "std": float(sample.std()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p10", "p50", "p90", "p99", "max"),
                quantiles,
                strict=True,
            )
        },
        "fraction_at_zero": float(np.mean(sample <= 1e-6)),
        "fraction_at_one": float(np.mean(sample >= 1.0 - 1e-6)),
        "rain_fraction_vil16": float(np.mean(sample >= 16.0 / 255.0)),
        "rain_fraction_vil74": float(np.mean(sample >= 74.0 / 255.0)),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_sample = sampled(first)
    second_sample = sampled(second)
    size = min(first_sample.size, second_sample.size)
    first_sample = first_sample[:size]
    second_sample = second_sample[:size]
    if first_sample.std() == 0.0 or second_sample.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(first_sample, second_sample)[0, 1])


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    absolute_sum = 0.0
    squared_sum = 0.0
    count = 0
    for start in range(0, prediction.shape[0], 8):
        difference = prediction[start : start + 8].astype(np.float64) - target[
            start : start + 8
        ].astype(np.float64)
        absolute_sum += float(np.abs(difference).sum())
        squared_sum += float(np.square(difference).sum())
        count += difference.size
    return {
        "mae_unit": absolute_sum / count,
        "mse_unit": squared_sum / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose residual forecast collapse")
    parser.add_argument("--residual", required=True)
    parser.add_argument("--deterministic", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with np.load(args.residual) as artifact:
        residual_prediction = artifact["prediction"]
        target = artifact["target"]
    with np.load(args.deterministic) as artifact:
        deterministic_prediction = artifact["prediction"]

    if not (
        residual_prediction.shape == deterministic_prediction.shape == target.shape
    ):
        raise ValueError("prediction artifacts have incompatible shapes")

    predicted_correction = residual_prediction - deterministic_prediction
    target_correction = target - deterministic_prediction
    axes = (0, 2, 3, 4)
    report = {
        "shape": list(target.shape),
        "residual_prediction": distribution(residual_prediction),
        "deterministic_prediction": distribution(deterministic_prediction),
        "target": distribution(target),
        "predicted_correction": distribution(predicted_correction),
        "target_correction": distribution(target_correction),
        "prediction_target_correlation": {
            "residual": correlation(residual_prediction, target),
            "deterministic": correlation(deterministic_prediction, target),
        },
        "error_metrics": {
            "residual": error_metrics(residual_prediction, target),
            "deterministic": error_metrics(deterministic_prediction, target),
        },
        "correction_target_correlation": correlation(
            predicted_correction, target_correction
        ),
        "mae_vil_by_lead": {
            "residual": (
                np.abs(residual_prediction - target).mean(axis=axes) * 255.0
            ).tolist(),
            "deterministic": (
                np.abs(deterministic_prediction - target).mean(axis=axes) * 255.0
            ).tolist(),
        },
        "mean_abs_correction_vil": {
            "predicted": float(np.abs(predicted_correction).mean() * 255.0),
            "target": float(np.abs(target_correction).mean() * 255.0),
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
