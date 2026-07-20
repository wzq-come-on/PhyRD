from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


SEVIR_THRESHOLDS = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)


def _pool(sequence: torch.Tensor, size: int) -> torch.Tensor:
    if sequence.ndim != 5:
        raise ValueError(f"sequence must have [B,T,C,H,W], got {tuple(sequence.shape)}")
    if size == 1:
        return sequence
    if size <= 0 or sequence.shape[-2] % size or sequence.shape[-1] % size:
        raise ValueError(f"pool={size} must divide spatial shape {tuple(sequence.shape[-2:])}")
    batch, frames, channels, height, width = sequence.shape
    flat = sequence.reshape(batch * frames, channels, height, width)
    pooled = F.max_pool2d(flat, size, stride=size)
    return pooled.reshape(batch, frames, channels, height // size, width // size)


def contingency(
    prediction: torch.Tensor, target: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    predicted_event = prediction >= threshold
    observed_event = target >= threshold
    hits = (predicted_event & observed_event).sum(dtype=torch.float64)
    misses = ((~predicted_event) & observed_event).sum(dtype=torch.float64)
    false_alarms = (predicted_event & (~observed_event)).sum(dtype=torch.float64)
    correct_negatives = ((~predicted_event) & (~observed_event)).sum(dtype=torch.float64)
    return hits, misses, false_alarms, correct_negatives


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator, numerator.new_zeros(()))


def categorical_scores(
    prediction_vil: torch.Tensor,
    target_vil: torch.Tensor,
    *,
    thresholds: Sequence[float] = SEVIR_THRESHOLDS,
    pool_sizes: Sequence[int] = (1, 4, 16),
) -> dict[str, float]:
    """Compute threshold CSI/HSS and mean CSI at registered max-pool scales."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    result: dict[str, float] = {}
    hss_values: list[torch.Tensor] = []
    for pool_size in pool_sizes:
        prediction = _pool(prediction_vil, pool_size)
        target = _pool(target_vil, pool_size)
        csi_values: list[torch.Tensor] = []
        for threshold in thresholds:
            hits, misses, false_alarms, correct_negatives = contingency(
                prediction, target, threshold
            )
            csi = _safe_ratio(hits, hits + misses + false_alarms)
            csi_values.append(csi)
            if pool_size == 1:
                numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
                denominator = (hits + misses) * (misses + correct_negatives) + (
                    hits + false_alarms
                ) * (false_alarms + correct_negatives)
                hss = _safe_ratio(numerator, denominator)
                hss_values.append(hss)
                result[f"CSI_{int(threshold)}"] = float(csi.item())
                result[f"HSS_{int(threshold)}"] = float(hss.item())
        mean_csi = torch.stack(csi_values).mean()
        name = "CSI" if pool_size == 1 else f"CSI_pool{pool_size}"
        result[name] = float(mean_csi.item())
    result["HSS"] = float(torch.stack(hss_values).mean().item())
    return result

