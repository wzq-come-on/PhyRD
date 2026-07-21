from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .probabilistic.base import ProbabilisticModel


class ForecastComposer(nn.Module):
    """Compose one deterministic backbone with one probabilistic model."""

    def __init__(
        self,
        deterministic: nn.Module,
        probabilistic: ProbabilisticModel,
        *,
        freeze_deterministic: bool = True,
        deterministic_name: str = "unknown",
        deterministic_params: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.probabilistic = probabilistic
        self.freeze_deterministic = bool(freeze_deterministic)
        self.deterministic_name = deterministic_name
        self.deterministic_params = dict(deterministic_params or {})
        if self.freeze_deterministic:
            self.deterministic.requires_grad_(False)

    @property
    def diffusion(self) -> nn.Module:
        """Compatibility alias while callers migrate to ``probabilistic``."""
        diffusion = getattr(self.probabilistic, "diffusion", None)
        if diffusion is None:
            raise AttributeError("the selected probabilistic model has no diffusion module")
        return diffusion

    @property
    def diffusion_config(self) -> dict[str, object]:
        return dict(getattr(self.probabilistic, "diffusion_config", {}))

    def predict_trend(self, history: torch.Tensor) -> torch.Tensor:
        trend = self.deterministic(history)
        return trend.detach() if self.freeze_deterministic else trend

    def training_loss(self, history: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        trend = self.predict_trend(history)
        result = self.probabilistic.training_loss(history, target, trend)
        result.setdefault("trend", trend)
        result.setdefault("prediction_x0", trend + result["clean_prediction"])
        return result

    def forward(
        self,
        history: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        stage: str = "deterministic",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compatibility entry point for the current two-stage trainer."""
        if stage == "deterministic":
            if target is None:
                return self.deterministic(history)
            result = self.deterministic.training_loss(history, target)
            output = {"loss_gen": result.loss, "trend": result.prediction}
            output.update(result.metrics)
            return output
        if stage == "residual":
            if target is None:
                raise ValueError("residual forward requires target")
            return self.training_loss(history, target)
        raise ValueError("stage must be 'deterministic' or 'residual'")

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        *,
        ensemble_size: int = 1,
        sampling_steps: int = 20,
        guidance_factory: Callable[[torch.Tensor], Callable[[torch.Tensor, int], torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        trend = self.predict_trend(history)
        return self.probabilistic.sample(
            history,
            trend,
            ensemble_size=ensemble_size,
            sampling_steps=sampling_steps,
            guidance_factory=guidance_factory,
        )
