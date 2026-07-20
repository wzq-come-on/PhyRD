from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .deterministic import build_backbone
from .diffusion import GaussianResidualDiffusion, ResidualDenoiser


class PhyRDModel(nn.Module):
    def __init__(
        self,
        input_frames: int = 13,
        output_frames: int = 12,
        base_channels: int = 32,
        diffusion_steps: int = 100,
        freeze_deterministic: bool = True,
        deterministic: dict[str, object] | None = None,
        diffusion: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        deterministic_config = dict(deterministic or {})
        self.deterministic_name = str(deterministic_config.get("name", "sdir_official"))
        params = deterministic_config.get("params", {})
        if not isinstance(params, dict):
            raise TypeError("model.deterministic.params must be a mapping")
        self.deterministic_params = dict(params)
        self.deterministic = build_backbone(
            self.deterministic_name,
            input_frames=input_frames,
            output_frames=output_frames,
            params=self.deterministic_params,
        )
        diffusion_config = dict(diffusion or {})
        residual_stats_path = diffusion_config.pop("residual_stats_path", None)
        if residual_stats_path is not None:
            with Path(str(residual_stats_path)).open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            if not isinstance(statistics, dict):
                raise TypeError("residual statistics file must contain a JSON object")
            diffusion_config.setdefault("residual_center", statistics.get("center"))
            diffusion_config.setdefault("residual_scale", statistics.get("scale"))
        self.diffusion_config = dict(diffusion_config)
        if residual_stats_path is not None:
            self.diffusion_config["residual_stats_path"] = str(residual_stats_path)
        self.diffusion = GaussianResidualDiffusion(
            ResidualDenoiser(input_frames, output_frames, base_channels),
            diffusion_steps,
            **diffusion_config,
        )
        self.freeze_deterministic = freeze_deterministic
        if freeze_deterministic:
            self.deterministic.requires_grad_(False)

    def forward(
        self,
        history: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        stage: str = "deterministic",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """DDP-compatible entry point for both training stages."""
        if stage == "deterministic":
            if target is None:
                return self.deterministic(history)
            result = self.deterministic.training_loss(history, target)
            output = {
                "loss_gen": result.loss,
                "trend": result.prediction,
            }
            output.update(result.metrics)
            return output
        if stage == "residual":
            if target is None:
                raise ValueError("residual forward requires target")
            return self.diffusion_loss(history, target)
        raise ValueError("stage must be 'deterministic' or 'residual'")

    def predict_trend(self, history: torch.Tensor) -> torch.Tensor:
        trend = self.deterministic(history)
        return trend.detach() if self.freeze_deterministic else trend

    def diffusion_loss(
        self, history: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        trend = self.predict_trend(history)
        result = self.diffusion.training_loss(target - trend, history, trend)
        result["trend"] = trend
        result["prediction_x0"] = trend + result["clean_prediction"]
        return result

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
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        trend = self.predict_trend(history)
        members = []
        for _ in range(ensemble_size):
            guidance = guidance_factory(trend) if guidance_factory is not None else None
            residual = self.diffusion.ddim_sample(
                history, trend, sampling_steps=sampling_steps, guidance=guidance
            )
            members.append((trend + residual).clamp(0.0, 1.0))
        return torch.stack(members, dim=1)
