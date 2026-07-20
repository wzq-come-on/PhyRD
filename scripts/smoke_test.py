from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from phyrd.config import load_config
from phyrd.data import SEVIRDataset
from phyrd.evaluation import evaluate_forecasts
from phyrd.models import PhyRDModel
from phyrd.motion import build_motion_fields
from phyrd.physics import ProximalGuidance, proximal_correct, weak_transport_loss
from phyrd.utils import seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-SEVIR PhyRD end-to-end smoke test")
    parser.add_argument("--config", default="configs/diagnostics/smoke_sevir.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-lpips", action="store_true")
    return parser.parse_args()


def _float_dict(mapping: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[key] = float(value.detach().item())
        elif isinstance(value, (str, float, int, bool)):
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    cuda_index = (
        device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        if device.type == "cuda"
        else None
    )
    if cuda_index is not None:
        torch.cuda.init()
        torch.cuda.set_device(cuda_index)
        # PyTorch 2.4.1 in the target container rejects an explicit device
        # argument here, while the current-device form is valid (see cuda_probe.py).
        torch.cuda.reset_peak_memory_stats()

    data_config = dict(config["data"])
    if args.data_root is not None:
        data_config["root"] = args.data_root
    dataset = SEVIRDataset(
        data_config["root"],
        data_config["split"],
        input_frames=int(data_config["input_frames"]),
        output_frames=int(data_config["output_frames"]),
        window_start_index=int(data_config["window_start_index"]),
        model_resolution=int(data_config.get("model_resolution", 384)),
        spatial_preprocess=str(data_config.get("spatial_preprocess", "none")),
        max_samples=int(data_config.get("max_samples") or 1),
    )
    sample = dataset[0]
    history = sample["x"].unsqueeze(0).to(device)
    target = sample["y"].unsqueeze(0).to(device)
    model_resolution = int(data_config.get("model_resolution", 384))
    if history.shape != (1, 13, 1, model_resolution, model_resolution) or target.shape != (
        1,
        12,
        1,
        model_resolution,
        model_resolution,
    ):
        raise AssertionError(f"frozen protocol shape failure: {history.shape}, {target.shape}")

    model_config = config["model"]
    model = PhyRDModel(
        base_channels=int(model_config["base_channels"]),
        diffusion_steps=int(model_config["diffusion_steps"]),
        freeze_deterministic=bool(model_config["freeze_deterministic"]),
        deterministic=dict(model_config["deterministic"]),
        diffusion=dict(model_config.get("diffusion", {})),
    ).to(device)
    model.train()
    started = time.perf_counter()
    diffusion_result = model.diffusion_loss(history, target)
    fields = build_motion_fields(history, output_frames=12)
    physics_config = dict(config["physics"])
    physics_kwargs = {
        "robust_scale": physics_config["robust_scale"],
        "tolerance": physics_config["tolerance"],
        "gamma_nadv": physics_config["gamma_nadv"],
        "pool_sizes": tuple(physics_config["pool_sizes"]),
        "alpha_mass": physics_config["alpha_mass"],
    }
    physics_loss, physics_diagnostics = weak_transport_loss(
        diffusion_result["prediction_x0"],
        fields.flow.detach(),
        fields.c_flow.detach(),
        fields.m_nadv.detach(),
        **physics_kwargs,
    )
    total_loss = diffusion_result["loss_gen"] + physics_config["lambda_train"] * physics_loss
    total_loss.backward()
    gradient_norm = sum(
        float(parameter.grad.detach().norm().item())
        for parameter in model.diffusion.parameters()
        if parameter.grad is not None
    )
    if gradient_norm <= 0:
        raise AssertionError("diffusion gradient is zero")
    deterministic_grads = [
        parameter.grad for parameter in model.deterministic.parameters() if parameter.grad is not None
    ]
    if deterministic_grads:
        raise AssertionError("frozen deterministic branch unexpectedly received gradients")

    timestep = torch.tensor([model.diffusion.diffusion_steps // 2], device=device)
    clean_probe = torch.randn_like(target)
    noisy_probe, _ = model.diffusion.q_sample(clean_probe, timestep)
    reparameterization_error = float(
        model.diffusion.reparameterization_error(noisy_probe, clean_probe, timestep).item()
    )
    if reparameterization_error > 2e-4:
        raise AssertionError(f"sampler reparameterization error too large: {reparameterization_error}")

    proximal = proximal_correct(
        diffusion_result["clean_prediction"],
        diffusion_result["trend"],
        fields.flow.detach(),
        fields.c_flow.detach(),
        fields.m_nadv.detach(),
        step_size=float(physics_config["proximal_step_size"]),
        **physics_kwargs,
    )
    if proximal.energy_after > proximal.energy_before + 1e-7:
        raise AssertionError("proximal correction increased weak-transport energy")

    model.eval()

    def guidance_factory(trend: torch.Tensor) -> ProximalGuidance:
        return ProximalGuidance(
            trend,
            fields.flow,
            fields.c_flow,
            fields.m_nadv,
            apply_below_timestep=int(physics_config["proximal_apply_below_timestep"]),
            step_size=float(physics_config["proximal_step_size"]),
            **physics_kwargs,
        )

    ensemble = model.sample(
        history,
        ensemble_size=int(model_config["ensemble_size"]),
        sampling_steps=int(model_config["sampling_steps"]),
        guidance_factory=guidance_factory,
    )
    prediction = ensemble.mean(dim=1)
    try:
        metrics = evaluate_forecasts(
            prediction,
            target,
            ensemble=ensemble,
            thresholds=tuple(config["evaluation"]["thresholds"]),
            include_lpips=True,
            lpips_net=str(config["evaluation"]["lpips_net"]),
        )
    except Exception:
        if args.require_lpips:
            raise
        metrics = evaluate_forecasts(
            prediction,
            target,
            ensemble=ensemble,
            thresholds=tuple(config["evaluation"]["thresholds"]),
            include_lpips=False,
        )
    required = {"CSI", "CSI_pool4", "CSI_pool16", "HSS", "LPIPS", "SSIM", "CRPS", "MAE"}
    missing = required.difference(metrics)
    if missing:
        raise AssertionError(f"missing registered metrics: {sorted(missing)}")
    if args.require_lpips and not isinstance(metrics["LPIPS"], float):
        raise AssertionError("LPIPS was required but not computed")

    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "status": "code-ready",
        "warning": "untrained smoke output; never use these numbers as model evidence",
        "sample": {
            "sample_id": sample["sample_id"],
            "event_id": sample["event_id"],
            "time_utc": sample["time_utc"],
            "history_shape": list(history.shape),
            "target_shape": list(target.shape),
            "catalog_path": str(dataset.paths.catalog_path),
            "data_root": str(dataset.paths.data_root),
            "native_resolution": dataset.native_resolution,
            "model_resolution": dataset.model_resolution,
            "spatial_preprocess": dataset.spatial_preprocess,
        },
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None,
            "elapsed_seconds": elapsed,
            "peak_memory_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if cuda_index is not None
                else 0.0
            ),
        },
        "checks": {
            "loss_gen": float(diffusion_result["loss_gen"].detach().item()),
            "loss_phys": float(physics_loss.detach().item()),
            "gradient_norm": gradient_norm,
            "reparameterization_max_error": reparameterization_error,
            "proximal_energy_before": proximal.energy_before,
            "proximal_energy_after": proximal.energy_after,
            "proximal_accepted": proximal.accepted,
            **_float_dict(physics_diagnostics),
        },
        "metrics": metrics,
    }
    artifact_dir = Path(config["artifacts"]["directory"])
    write_json(artifact_dir / "smoke_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SMOKE_TEST_OK")
    dataset.close()


if __name__ == "__main__":
    main()
