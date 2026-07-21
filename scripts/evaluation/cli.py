"""Canonical evaluation command line dispatcher.

There is one public entry point (``scripts.evaluate``).  The metric
implementation remains in :mod:`common`; model-specific files are adapters
used by the dispatcher and are not independent user-facing CLIs.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Sequence

import numpy as np
import torch


def _artifact_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PhyRD prediction artifact")
    parser.add_argument("--predictions", required=True, help="NPZ with prediction, target, optional ensemble")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-lpips", action="store_true")
    args = parser.parse_args(list(argv))
    from phyrd.evaluation import evaluate_forecasts

    artifact = np.load(args.predictions)
    if "prediction" not in artifact or "target" not in artifact:
        raise KeyError("NPZ must contain prediction and target")
    device = torch.device(args.device)
    prediction = torch.from_numpy(artifact["prediction"]).float().to(device)
    target = torch.from_numpy(artifact["target"]).float().to(device)
    ensemble = torch.from_numpy(artifact["ensemble"]).float().to(device) if "ensemble" in artifact else None
    metrics = evaluate_forecasts(prediction, target, ensemble=ensemble, include_lpips=args.require_lpips)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _module_main(module_name: str, argv: Sequence[str]) -> None:
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        module.main()
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: the old ``scripts/evaluate.py --predictions``
    # interface remains an artifact evaluator when no mode is supplied.
    if "--mode" not in raw:
        _artifact_main(raw)
        return
    parser = argparse.ArgumentParser(description="PhyRD unified evaluation CLI")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("artifact", "protocol", "residual_diffcast", "deterministic_diffcast"),
    )
    parser.add_argument("--protocol", choices=("5to20", "13to12"), help="protocol for --mode protocol")
    args, remainder = parser.parse_known_args(raw)
    if args.mode == "artifact":
        _artifact_main(remainder)
    elif args.mode == "protocol":
        if not args.protocol:
            parser.error("--protocol is required with --mode protocol")
        from .common import run

        run(args.protocol, remainder)
    elif args.mode == "residual_diffcast":
        _module_main("scripts.evaluation.evaluate_residual_diffcast", remainder)
    else:
        _module_main("scripts.evaluation.evaluate_deterministic_diffcast", remainder)


if __name__ == "__main__":
    main()
