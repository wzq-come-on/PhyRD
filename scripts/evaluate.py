from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from phyrd.evaluation import evaluate_forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PhyRD prediction artifact")
    parser.add_argument("--predictions", required=True, help="NPZ with prediction, target, optional ensemble")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-lpips", action="store_true")
    args = parser.parse_args()
    artifact = np.load(args.predictions)
    if "prediction" not in artifact or "target" not in artifact:
        raise KeyError("NPZ must contain prediction and target")
    device = torch.device(args.device)
    prediction = torch.from_numpy(artifact["prediction"]).float().to(device)
    target = torch.from_numpy(artifact["target"]).float().to(device)
    ensemble = (
        torch.from_numpy(artifact["ensemble"]).float().to(device)
        if "ensemble" in artifact
        else None
    )
    metrics = evaluate_forecasts(
        prediction,
        target,
        ensemble=ensemble,
        include_lpips=args.require_lpips,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

