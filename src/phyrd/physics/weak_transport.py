from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.nn import functional as F

from .warp import warp_image


def _validate_prediction_and_flow(
    prediction: torch.Tensor, flow: torch.Tensor
) -> tuple[int, int, int, int, int]:
    if prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError(
            f"prediction must have [B,T,1,H,W], got {tuple(prediction.shape)}"
        )
    batch, frames, channels, height, width = prediction.shape
    expected = (batch, frames - 1, 2, height, width)
    if tuple(flow.shape) != expected:
        raise ValueError(f"flow must have {expected}, got {tuple(flow.shape)}")
    return batch, frames, channels, height, width


def transport_residual(
    prediction: torch.Tensor, flow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local weak-transport residual, advected field, and valid mask."""
    batch, frames, _, height, width = _validate_prediction_and_flow(prediction, flow)
    previous = prediction[:, :-1].reshape(batch * (frames - 1), 1, height, width)
    flat_flow = flow.reshape(batch * (frames - 1), 2, height, width)
    advected, valid = warp_image(previous, flat_flow, return_valid=True)
    advected = advected.reshape(batch, frames - 1, 1, height, width)
    valid = valid.reshape(batch, frames - 1, 1, height, width)
    residual = prediction[:, 1:] - advected
    return residual, advected, valid


def _lead_tensor(
    value: float | Sequence[float] | torch.Tensor,
    transitions: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype).flatten()
    if tensor.numel() == 1:
        tensor = tensor.repeat(transitions)
    if tensor.numel() != transitions:
        raise ValueError(f"expected one or {transitions} lead values, got {tensor.numel()}")
    return tensor.reshape(1, transitions, 1, 1, 1)


def _field(
    value: torch.Tensor,
    batch: int,
    transitions: int,
    height: int,
    width: int,
    name: str,
) -> torch.Tensor:
    if value.ndim == 4:
        value = value.unsqueeze(2)
    expected = (batch, transitions, 1, height, width)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have [B,{transitions},H,W], got {tuple(value.shape)}")
    return value


def _weighted_smooth_l1(violation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = F.smooth_l1_loss(violation, torch.zeros_like(violation), reduction="none")
    expanded = weight.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def weak_transport_loss(
    prediction: torch.Tensor,
    flow: torch.Tensor,
    c_flow: torch.Tensor,
    m_nadv: torch.Tensor,
    *,
    robust_scale: float | Sequence[float] | torch.Tensor = 0.05,
    tolerance: float | Sequence[float] | torch.Tensor = 0.1,
    gamma_nadv: float = 1.0,
    pool_sizes: Sequence[int] = (8, 16, 32),
    alpha_mass: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reliability-gated weak transport with non-advection tolerance.

    VIL is treated as a fixed non-negative proxy, never as exact conserved mass.
    """
    batch, frames, _, height, width = _validate_prediction_and_flow(prediction, flow)
    transitions = frames - 1
    confidence = _field(c_flow, batch, transitions, height, width, "c_flow").clamp_min(0.0)
    nonadvective = _field(m_nadv, batch, transitions, height, width, "m_nadv").clamp(0.0, 1.0)
    proxy = prediction.clamp_min(0.0)
    residual, advected, valid = transport_residual(proxy, flow)
    scale = _lead_tensor(robust_scale, transitions, residual).clamp_min(1e-6)
    base_tolerance = _lead_tensor(tolerance, transitions, residual)
    allowed = base_tolerance * (1.0 + gamma_nadv * nonadvective)
    normalized = residual.abs() / scale
    violation = F.relu(normalized - allowed)
    weight = confidence * valid
    loss_local = _weighted_smooth_l1(violation, weight)

    mass_losses: dict[int, torch.Tensor] = {}
    flat_residual = residual.reshape(batch * transitions, 1, height, width)
    flat_weight = weight.reshape(batch * transitions, 1, height, width)
    flat_allowed = allowed.reshape(batch * transitions, 1, height, width)
    for size in pool_sizes:
        if size <= 0 or height % size or width % size:
            raise ValueError(f"pool size {size} must divide spatial shape {(height, width)}")
        pooled_residual = F.avg_pool2d(flat_residual, size, stride=size).abs()
        pooled_weight = F.avg_pool2d(flat_weight, size, stride=size)
        pooled_allowed = F.avg_pool2d(flat_allowed, size, stride=size)
        flat_scale = scale.expand(batch, -1, -1, -1, -1).reshape(
            batch * transitions, 1, 1, 1
        )
        pooled_violation = F.relu(pooled_residual / flat_scale - pooled_allowed)
        mass_losses[size] = _weighted_smooth_l1(pooled_violation, pooled_weight)
    loss_mass = (
        torch.stack(list(mass_losses.values())).mean()
        if mass_losses
        else residual.new_zeros(())
    )
    loss = loss_local + alpha_mass * loss_mass
    diagnostics: dict[str, Any] = {
        "loss_local": loss_local,
        "loss_mass": loss_mass,
        "mass_losses": mass_losses,
        "residual": residual,
        "advected": advected,
        "valid_mask": valid,
        "violation_map": violation[:, :, 0],
        "mean_confidence": confidence.mean(),
        "mean_nonadvective": nonadvective.mean(),
    }
    return loss, diagnostics
