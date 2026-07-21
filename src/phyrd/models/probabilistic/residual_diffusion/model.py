from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from ..base import ProbabilisticModel
from .diffusion import GaussianResidualDiffusion
from .diffusion import ResidualDenoiser


class ResidualDiffusionModel(ProbabilisticModel):
    """Residual diffusion packaged behind the common probabilistic interface."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        base_channels: int = 32,
        diffusion_steps: int = 100,
        extensions: list[dict[str, object]] | None = None,
        **diffusion_config: object,
    ) -> None:
        super().__init__()
        residual_stats_path = diffusion_config.pop("residual_stats_path", None)
        if residual_stats_path is not None:
            with Path(str(residual_stats_path)).open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            if not isinstance(statistics, dict):
                raise TypeError("residual statistics file must contain a JSON object")
            diffusion_config.setdefault("residual_center", statistics.get("center"))
            diffusion_config.setdefault("residual_scale", statistics.get("scale"))
        self.extensions = list(extensions or [])
        self.diffusion_config = dict(diffusion_config)
        if residual_stats_path is not None:
            self.diffusion_config["residual_stats_path"] = str(residual_stats_path)
        self.diffusion = GaussianResidualDiffusion(
            ResidualDenoiser(input_frames, output_frames, base_channels),
            diffusion_steps,
            **diffusion_config,
        )

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        trend: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = self.diffusion.training_loss(target - trend, history, trend)
        result["trend"] = trend
        result["prediction_x0"] = trend + result["clean_prediction"]
        return result

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
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        members = []
        for _ in range(ensemble_size):
            guidance = guidance_factory(trend) if guidance_factory is not None else None
            residual = self.diffusion.ddim_sample(
                history,
                trend,
                sampling_steps=sampling_steps,
                guidance=guidance,
            )
            members.append((trend + residual).clamp(0.0, 1.0))
        return torch.stack(members, dim=1)
