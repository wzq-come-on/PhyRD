from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class ProbabilisticModel(nn.Module):
    """Small interface shared by every probabilistic forecast model.

    A probabilistic model receives the deterministic trend explicitly. This
    keeps the trainer and evaluator independent of whether the stochastic
    component is a diffusion model, JDIR, flow matching, or another sampler.
    """

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        trend: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        trend: torch.Tensor,
        *,
        ensemble_size: int = 1,
        sampling_steps: int = 20,
        guidance_factory: Callable[[torch.Tensor], Callable[[torch.Tensor, int], torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
