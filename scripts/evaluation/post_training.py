from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from phyrd.evaluation.probabilistic import crps_ensemble
try:
    from scripts.evaluation.common import (
        _continuous_metrics,
        _ssim,
        categorical_metrics,
        contingency,
    )
except ModuleNotFoundError:
    from .common import _continuous_metrics, _ssim, categorical_metrics, contingency


class RankStridedSampler(Sampler[int]):
    """Shard without padding, so a full test never evaluates duplicate samples."""

    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - 1 - self.rank) // self.world_size + 1)


@torch.inference_mode()
def evaluate_post_training(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    ensemble_size: int,
    sampling_steps: int,
    checkpoint_epoch: int,
    checkpoint_path: str,
    split: str,
) -> dict[str, Any] | None:
    sampler = RankStridedSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model.eval()
    frames = int(dataset.output_frames)
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    counts = torch.zeros(3, 6, 4, dtype=torch.float64, device=device)
    ssim_sum = torch.zeros(1, dtype=torch.float64, device=device)
    ssim_n = torch.zeros(1, dtype=torch.float64, device=device)
    lead_abs = torch.zeros(frames, dtype=torch.float64, device=device)
    lead_n = torch.zeros(frames, dtype=torch.float64, device=device)
    crps_sum = torch.zeros(1, dtype=torch.float64, device=device)
    crps_n = torch.zeros(1, dtype=torch.float64, device=device)
    started = time.perf_counter()
    processed_local = 0
    total_samples = len(dataset)
    for batch_index, batch in enumerate(loader):
        history = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True).float()
        ensemble = model.sample(
            history,
            ensemble_size=ensemble_size,
            sampling_steps=sampling_steps,
        ).clamp(0, 1).float()
        prediction = ensemble.mean(dim=1)
        abs_sum, sq_sum, elements, batch_n = _continuous_metrics(prediction, target)
        sums += torch.stack((abs_sum, sq_sum, elements, batch_n))
        counts += contingency(prediction * 255, target * 255)
        ssim_sum += _ssim(prediction, target).double() * prediction.shape[0] * frames
        ssim_n += prediction.shape[0] * frames
        lead_abs += (prediction - target).abs().sum(
            dim=(0, 2, 3, 4), dtype=torch.float64
        )
        lead_n += (
            prediction.shape[0]
            * prediction.shape[2]
            * prediction.shape[3]
            * prediction.shape[4]
        )
        crps_sum += (
            crps_ensemble(ensemble * 255.0, target * 255.0).double()
            * prediction.shape[0]
        )
        crps_n += prediction.shape[0]
        processed_local += int(prediction.shape[0])
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
            progress = torch.tensor(
                float(processed_local), dtype=torch.float64, device=device
            )
            if world_size > 1:
                dist.all_reduce(progress, op=dist.ReduceOp.SUM)
            if rank == 0:
                percent = 100.0 * progress.item() / max(1, total_samples)
                elapsed = time.perf_counter() - started
                print(
                    f"[report_test] processed {int(progress.item())}/{total_samples} "
                    f"({percent:.1f}%) elapsed={elapsed / 60.0:.1f} min",
                    flush=True,
                )
    if world_size > 1:
        for value in (
            sums,
            counts,
            ssim_sum,
            ssim_n,
            lead_abs,
            lead_n,
            crps_sum,
            crps_n,
        ):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if rank != 0:
        return None
    metrics: dict[str, Any] = {
        "status": "completed",
        "model": "phyrd_residual_diffusion",
        "checkpoint": checkpoint_path,
        "split": split,
        "samples": int(sums[3].item()),
        "sampling_steps": int(sampling_steps),
        "ensemble_size": int(ensemble_size),
        "epoch": int(checkpoint_epoch),
        "MAE": (sums[0] / sums[2]).item(),
        "MSE": (sums[1] / sums[2]).item(),
        "SSIM": (ssim_sum[0] / ssim_n[0]).item(),
        "lead_mae_vil": (lead_abs / lead_n.clamp_min(1) * 255).tolist(),
        "CRPS": (crps_sum / crps_n.clamp_min(1)).item(),
        "CRPS_domain": "encoded VIL [0,255]",
        "seconds": time.perf_counter() - started,
        "world_size": int(world_size),
        "per_rank_batch_size": int(batch_size),
    }
    metrics.update(categorical_metrics(counts))
    return metrics
