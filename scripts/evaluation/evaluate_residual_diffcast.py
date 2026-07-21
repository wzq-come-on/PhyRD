from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.common import (
    _batch_xy,
    _continuous_metrics,
    _ssim,
    categorical_metrics,
    contingency,
)
from phyrd.config import load_config  # noqa: E402
from phyrd.data import DiffCastH5Dataset, SEVIRDataset  # noqa: E402
from phyrd.evaluation.probabilistic import crps_ensemble  # noqa: E402
from phyrd.evaluation.risk import build_risk_batch  # noqa: E402
from phyrd.models import build_composite_from_config  # noqa: E402
from phyrd.motion import build_motion_fields  # noqa: E402
from phyrd.physics import ProximalGuidance, weak_transport_loss  # noqa: E402


class RankStridedSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length, self.rank, self.world_size = int(length), int(rank), int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - 1 - self.rank) // self.world_size + 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the trained PhyRD residual diffusion model")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default=None, help="required for deterministic_pool configs")
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--ensemble-size", type=int, default=1)
    p.add_argument("--physics-guidance", action="store_true")
    p.add_argument("--guidance-every", type=int, default=1)
    p.add_argument("--guidance-step-size", type=float, default=0.1)
    p.add_argument("--guidance-rho", type=float, default=0.1)
    p.add_argument("--guidance-lambda-max", type=float, default=5.0)
    p.add_argument("--guidance-apply-below-timestep", type=int, default=80)
    p.add_argument("--risk-artifact", default=None)
    p.add_argument("--risk-patch-size", type=int, default=16)
    p.add_argument("--risk-error-threshold", type=float, default=32.0)
    p.add_argument("--risk-strong-threshold", type=float, default=219.0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank, device = 0, torch.device(args.device)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    dataset_kwargs = {
        "input_frames": int(data_cfg["input_frames"]),
        "output_frames": int(data_cfg["output_frames"]),
        "window_start_index": int(data_cfg.get("window_start_index", 12)),
        "model_resolution": int(data_cfg["model_resolution"]),
        "spatial_preprocess": str(data_cfg.get("spatial_preprocess", "none")),
        "max_samples": args.max_samples,
    }
    data_format = str(data_cfg.get("format", "diffcast_h5"))
    if data_format == "catalog":
        dataset = SEVIRDataset(data_cfg["root"], args.split, **dataset_kwargs)
    elif data_format == "diffcast_h5":
        dataset = DiffCastH5Dataset(data_cfg["root"], args.split, **dataset_kwargs)
    else:
        raise ValueError("data.format must be 'catalog' or 'diffcast_h5'")
    sampler = RankStridedSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_composite_from_config(
        cfg,
        input_frames=int(data_cfg["input_frames"]),
        output_frames=int(data_cfg["output_frames"]),
    ).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "deterministic" in payload:
        model.deterministic.load_state_dict(payload["deterministic"], strict=True)
    elif hasattr(model.deterministic, "select"):
        if args.backbone is None:
            raise ValueError("--backbone is required when evaluating a deterministic_pool config")
        model.select_backbone(args.backbone)
    else:
        raise KeyError("checkpoint does not contain a deterministic state dict")
    model.diffusion.load_state_dict(payload["diffusion"], strict=True)

    frames = int(data_cfg["output_frames"])
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    counts = torch.zeros(3, 6, 4, dtype=torch.float64, device=device)
    ssim_sum = torch.zeros(1, dtype=torch.float64, device=device)
    ssim_n = torch.zeros(1, dtype=torch.float64, device=device)
    lead_abs = torch.zeros(frames, dtype=torch.float64, device=device)
    lead_n = torch.zeros(frames, dtype=torch.float64, device=device)
    crps_sum = torch.zeros(1, dtype=torch.float64, device=device)
    crps_n = torch.zeros(1, dtype=torch.float64, device=device)
    physics_config = dict(cfg.get("physics", {}))
    risk_features: list[torch.Tensor] = []
    risk_targets: dict[str, list[torch.Tensor]] = {}
    if args.risk_artifact is not None and args.split != "val_calib":
        raise ValueError("risk artifacts must be generated from the val_calib split")
    started = time.perf_counter()
    evaluation_context = torch.no_grad() if args.physics_guidance else torch.inference_mode()
    with evaluation_context:
        for batch in loader:
            history, target = _batch_xy(batch, int(data_cfg["input_frames"]), frames)
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            guidance_factory = None
            if args.physics_guidance or args.risk_artifact is not None:
                fields = build_motion_fields(history.float(), output_frames=frames)
                guidance_kwargs = {
                    "robust_scale": physics_config.get("robust_scale", 0.05),
                    "tolerance": physics_config.get("tolerance", 0.1),
                    "gamma_nadv": physics_config.get("gamma_nadv", 1.0),
                    "pool_sizes": tuple(physics_config.get("pool_sizes", (8, 16, 32))),
                    "alpha_mass": physics_config.get("alpha_mass", 0.25),
                    "step_size": args.guidance_step_size,
                    "rho": args.guidance_rho,
                    "lambda_max": args.guidance_lambda_max,
                }

                def make_guidance(trend: torch.Tensor) -> ProximalGuidance:
                    return ProximalGuidance(
                        trend,
                        fields.flow.detach(),
                        fields.c_flow.detach(),
                        fields.m_nadv.detach(),
                        apply_below_timestep=args.guidance_apply_below_timestep,
                        every=args.guidance_every,
                        **guidance_kwargs,
                    )

                guidance_factory = make_guidance
            ensemble = model.sample(
                history,
                ensemble_size=args.ensemble_size,
                sampling_steps=args.sampling_steps,
                guidance_factory=guidance_factory,
            ).clamp(0, 1).float()
            prediction = ensemble.mean(dim=1)
            abs_sum, sq_sum, elements, batch_n = _continuous_metrics(prediction, target)
            sums += torch.stack((abs_sum, sq_sum, elements, batch_n))
            counts += contingency(prediction * 255, target * 255)
            ssim_sum += _ssim(prediction, target).double() * prediction.shape[0] * frames
            ssim_n += prediction.shape[0] * frames
            lead_abs += (prediction - target).abs().sum(dim=(0, 2, 3, 4), dtype=torch.float64)
            lead_n += prediction.shape[0] * prediction.shape[2] * prediction.shape[3] * prediction.shape[4]
            crps_sum += crps_ensemble(ensemble * 255.0, target * 255.0).double() * prediction.shape[0]
            crps_n += prediction.shape[0]
            if args.risk_artifact is not None:
                _, diagnostics = weak_transport_loss(
                    prediction,
                    fields.flow.detach(),
                    fields.c_flow.detach(),
                    fields.m_nadv.detach(),
                    robust_scale=physics_config.get("robust_scale", 0.05),
                    tolerance=physics_config.get("tolerance", 0.1),
                    gamma_nadv=physics_config.get("gamma_nadv", 1.0),
                    pool_sizes=tuple(physics_config.get("pool_sizes", (8, 16, 32))),
                    alpha_mass=physics_config.get("alpha_mass", 0.25),
                )
                batch_features, batch_targets = build_risk_batch(
                    ensemble,
                    prediction,
                    target,
                    history,
                    diagnostics["violation_map"],
                    fields.c_flow,
                    fields.m_nadv,
                    patch_size=args.risk_patch_size,
                    error_threshold=args.risk_error_threshold,
                    strong_threshold=args.risk_strong_threshold,
                )
                risk_features.append(batch_features.cpu())
                for name, values in batch_targets.items():
                    risk_targets.setdefault(name, []).append(values.cpu())

    if distributed:
        for value in (sums, counts, ssim_sum, ssim_n, lead_abs, lead_n, crps_sum, crps_n):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if args.risk_artifact is not None:
        artifact_path = Path(args.risk_artifact)
        if world_size > 1:
            artifact_path = artifact_path.with_suffix(artifact_path.suffix + f".rank{rank}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": torch.cat(risk_features) if risk_features else torch.empty(0),
                "targets": {name: torch.cat(values) for name, values in risk_targets.items()},
                "feature_names": (
                    "log_u_ens",
                    "log_r_phys",
                    "one_minus_c_flow",
                    "m_nadv",
                    "predicted_intensity",
                    "input_intensity",
                    "lead_time",
                    "predicted_gradient",
                    "predicted_object_size",
                ),
                "split": args.split,
                "patch_size": args.risk_patch_size,
                "error_threshold": args.risk_error_threshold,
                "strong_threshold": args.risk_strong_threshold,
            },
            artifact_path,
        )
    if rank == 0:
        metrics = {
            "status": "completed",
            "model": "phyrd_residual_diffusion",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "split": args.split,
            "samples": int(sums[3].item()),
            "sampling_steps": args.sampling_steps,
            "ensemble_size": args.ensemble_size,
            "physics_guidance": args.physics_guidance,
            "epoch": int(payload.get("epoch", -1)),
            "step": int(payload.get("step", -1)),
            "MAE": (sums[0] / sums[2]).item(),
            "MSE": (sums[1] / sums[2]).item(),
            "SSIM": (ssim_sum[0] / ssim_n[0]).item(),
            "lead_mae_vil": (lead_abs / lead_n.clamp_min(1) * 255).tolist(),
            "CRPS": (crps_sum / crps_n.clamp_min(1)).item(),
            "CRPS_domain": "encoded VIL [0,255]",
            "seconds": time.perf_counter() - started,
            "world_size": world_size,
            "per_rank_batch_size": args.batch_size,
        }
        metrics.update(categorical_metrics(counts))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if hasattr(dataset, "close"):
        dataset.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
