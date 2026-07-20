from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .weak_transport import weak_transport_loss


@dataclass
class ProximalResult:
    corrected: torch.Tensor
    lambda_map: torch.Tensor
    energy_before: float
    energy_after: float
    accepted: bool
    step_size: float
    backtracks: int


def proximal_correct(
    clean_residual: torch.Tensor,
    deterministic: torch.Tensor,
    flow: torch.Tensor,
    c_flow: torch.Tensor,
    m_nadv: torch.Tensor,
    *,
    lambda_map: torch.Tensor | None = None,
    step_size: float = 0.1,
    rho: float = 0.1,
    lambda_max: float = 5.0,
    max_grad_norm: float = 1.0,
    max_backtracks: int = 6,
    robust_scale: float | Sequence[float] | torch.Tensor = 0.05,
    tolerance: float | Sequence[float] | torch.Tensor = 0.1,
    gamma_nadv: float = 1.0,
    pool_sizes: Sequence[int] = (8, 16, 32),
    alpha_mass: float = 0.25,
) -> ProximalResult:
    """Take one backtracked gradient step in clean-residual space."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    transitions = clean_residual.shape[1] - 1
    if lambda_map is None:
        lambda_map = torch.zeros(
            clean_residual.shape[0],
            transitions,
            clean_residual.shape[-2],
            clean_residual.shape[-1],
            device=clean_residual.device,
            dtype=clean_residual.dtype,
        )
    if lambda_map.shape != c_flow.shape:
        raise ValueError("lambda_map and c_flow must have the same shape")
    variable = clean_residual.detach().requires_grad_(True)
    effective_confidence = c_flow.clamp(0.0, 1.0) * (1.0 + lambda_map.detach())
    energy, diagnostics = weak_transport_loss(
        deterministic + variable,
        flow,
        effective_confidence,
        m_nadv,
        robust_scale=robust_scale,
        tolerance=tolerance,
        gamma_nadv=gamma_nadv,
        pool_sizes=pool_sizes,
        alpha_mass=alpha_mass,
    )
    gradient = torch.autograd.grad(energy, variable, create_graph=False)[0]
    flat = gradient.flatten(1)
    norms = flat.norm(dim=1).clamp_min(1e-12)
    factors = (max_grad_norm / norms).clamp_max(1.0)
    gradient = gradient * factors.reshape(-1, 1, 1, 1, 1)
    before = float(energy.detach().item())
    accepted = False
    after = before
    used_step = step_size
    candidate = variable.detach()
    used_backtracks = max_backtracks
    for attempt in range(max_backtracks + 1):
        proposal = variable.detach() - used_step * gradient.detach()
        proposal_energy, _ = weak_transport_loss(
            deterministic.detach() + proposal,
            flow,
            effective_confidence,
            m_nadv,
            robust_scale=robust_scale,
            tolerance=tolerance,
            gamma_nadv=gamma_nadv,
            pool_sizes=pool_sizes,
            alpha_mass=alpha_mass,
        )
        value = float(proposal_energy.detach().item())
        if value <= before + 1e-8:
            candidate = proposal
            after = value
            accepted = True
            used_backtracks = attempt
            break
        used_step *= 0.5
    violation = diagnostics["violation_map"].detach()
    updated_lambda = (
        lambda_map.detach() + rho * c_flow.detach() * violation
    ).clamp(0.0, lambda_max)
    return ProximalResult(
        corrected=candidate,
        lambda_map=updated_lambda,
        energy_before=before,
        energy_after=after,
        accepted=accepted,
        step_size=used_step,
        backtracks=used_backtracks,
    )


class ProximalGuidance:
    """Stateful violation-feedback adapter for the DDIM clean prediction hook."""

    def __init__(
        self,
        deterministic: torch.Tensor,
        flow: torch.Tensor,
        c_flow: torch.Tensor,
        m_nadv: torch.Tensor,
        *,
        apply_below_timestep: int,
        every: int = 1,
        **kwargs: Any,
    ) -> None:
        self.deterministic = deterministic
        self.flow = flow
        self.c_flow = c_flow
        self.m_nadv = m_nadv
        self.apply_below_timestep = apply_below_timestep
        self.every = every
        self.kwargs = kwargs
        self.lambda_map: torch.Tensor | None = None
        self.calls = 0
        self.accepted = 0

    def __call__(self, clean_residual: torch.Tensor, timestep: int) -> torch.Tensor:
        if timestep > self.apply_below_timestep or self.calls % self.every:
            self.calls += 1
            return clean_residual
        result = proximal_correct(
            clean_residual,
            self.deterministic,
            self.flow,
            self.c_flow,
            self.m_nadv,
            lambda_map=self.lambda_map,
            **self.kwargs,
        )
        self.calls += 1
        self.accepted += int(result.accepted)
        self.lambda_map = result.lambda_map
        return result.corrected
