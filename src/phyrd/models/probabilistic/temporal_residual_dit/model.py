from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from ..base import ProbabilisticModel
from ..residual_diffusion.diffusion import GaussianResidualDiffusion
from .denoiser import TemporalResidualDenoiser, spatial_high_pass


class TemporalResidualDiffusionModel(ProbabilisticModel):
    """DiffCast residual diffusion with an explicit temporal trajectory denoiser."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        image_size: int = 128,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        high_frequency_channels: int = 48,
        gradient_checkpointing: bool = True,
        diffusion_steps: int = 1000,
        prediction_type: str = "v",
        residual_stats_path: str | None = None,
        residual_center: float | list[float] | None = None,
        residual_scale: float | list[float] | None = None,
        x0_clip: float | None = 5.0,
        x0_clip_quantile: float | None = 0.995,
        high_frequency_loss_weight: float = 0.10,
        intensity_loss_weight: float = 0.05,
        intensity_threshold: float = 0.40,
        intensity_temperature: float = 0.08,
        **_: object,
    ) -> None:
        super().__init__()
        if residual_stats_path is not None:
            with Path(residual_stats_path).open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            if not isinstance(statistics, dict):
                raise TypeError("residual statistics file must contain a JSON object")
            residual_center = statistics.get("center", residual_center)
            residual_scale = statistics.get("scale", residual_scale)
        if high_frequency_loss_weight < 0 or intensity_loss_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if intensity_temperature <= 0:
            raise ValueError("intensity_temperature must be positive")

        denoiser = TemporalResidualDenoiser(
            input_frames,
            output_frames,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            high_frequency_channels=high_frequency_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.diffusion = GaussianResidualDiffusion(
            denoiser,
            diffusion_steps,
            prediction_type=prediction_type,
            residual_center=residual_center,
            residual_scale=residual_scale,
            x0_clip=x0_clip,
            x0_clip_quantile=x0_clip_quantile,
        )
        self.high_frequency_loss_weight = float(high_frequency_loss_weight)
        self.intensity_loss_weight = float(intensity_loss_weight)
        self.intensity_threshold = float(intensity_threshold)
        self.intensity_temperature = float(intensity_temperature)
        self.diffusion_config = {
            "name": "temporal_residual_dit",
            "image_size": int(image_size),
            "patch_size": int(patch_size),
            "hidden_size": int(hidden_size),
            "depth": int(depth),
            "num_heads": int(num_heads),
            "mlp_ratio": float(mlp_ratio),
            "high_frequency_channels": int(high_frequency_channels),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "diffusion_steps": int(diffusion_steps),
            "prediction_type": str(prediction_type),
            "x0_clip": x0_clip,
            "x0_clip_quantile": x0_clip_quantile,
            "high_frequency_loss_weight": self.high_frequency_loss_weight,
            "intensity_loss_weight": self.intensity_loss_weight,
            "intensity_threshold": self.intensity_threshold,
            "intensity_temperature": self.intensity_temperature,
        }
        if residual_stats_path is not None:
            self.diffusion_config["residual_stats_path"] = str(residual_stats_path)

    @staticmethod
    def _frame_high_pass(tensor: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = tensor.shape
        high = spatial_high_pass(
            tensor.reshape(batch * frames, channels, height, width)
        )
        return high.reshape(batch, frames, channels, height, width)

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        trend: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual = target - trend
        result = self.diffusion.training_loss(residual, history, trend)
        forecast = trend + result["clean_prediction"]
        high_frequency_loss = F.l1_loss(
            self._frame_high_pass(forecast),
            self._frame_high_pass(target),
        )
        strong_weight = 1.0 + torch.sigmoid(
            (target.detach() - self.intensity_threshold)
            / self.intensity_temperature
        )
        intensity_loss = (
            (forecast - target).abs() * strong_weight
        ).mean()
        diffusion_loss = result["loss_gen"]
        result["loss_diffusion"] = diffusion_loss
        result["loss_high_frequency"] = high_frequency_loss
        result["loss_intensity"] = intensity_loss
        result["loss_gen"] = (
            diffusion_loss
            + self.high_frequency_loss_weight * high_frequency_loss
            + self.intensity_loss_weight * intensity_loss
        )
        # Joint DiffCast-style training consumes this explicit alias while the
        # residual-only trainer uses loss_gen.
        result["loss_diff"] = result["loss_gen"]
        result["trend"] = trend
        result["prediction_x0"] = forecast
        return result

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        trend: torch.Tensor,
        *,
        ensemble_size: int = 1,
        sampling_steps: int = 20,
        guidance_factory: Callable[
            [torch.Tensor],
            Callable[[torch.Tensor, int], torch.Tensor],
        ]
        | None = None,
    ) -> torch.Tensor:
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        members: list[torch.Tensor] = []
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
