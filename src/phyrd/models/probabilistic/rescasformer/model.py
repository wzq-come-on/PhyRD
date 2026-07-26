from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from ..base import ProbabilisticModel
from ..residual_diffusion.diffusion import GaussianResidualDiffusion
from .denoiser import ResidualCasFormerDenoiser


class ResidualCasFormerModel(ProbabilisticModel):
    """DiffCast-style residual diffusion with a CasFormer denoising backbone."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        image_size: int = 128,
        patch_size: int = 4,
        frame_hidden_size: int = 256,
        frame_depth: int = 12,
        frame_heads: int = 4,
        sequence_hidden_size: int = 1152,
        sequence_depth: int = 12,
        sequence_heads: int = 16,
        mlp_ratio: float = 4.0,
        gradient_checkpointing: bool = True,
        diffusion_steps: int = 1000,
        prediction_type: str = "v",
        residual_stats_path: str | None = None,
        residual_center: float | list[float] | None = None,
        residual_scale: float | list[float] | None = None,
        x0_clip: float | None = 5.0,
        x0_clip_quantile: float | None = 0.995,
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

        denoiser = ResidualCasFormerDenoiser(
            input_frames,
            output_frames,
            image_size=image_size,
            patch_size=patch_size,
            frame_hidden_size=frame_hidden_size,
            frame_depth=frame_depth,
            frame_heads=frame_heads,
            sequence_hidden_size=sequence_hidden_size,
            sequence_depth=sequence_depth,
            sequence_heads=sequence_heads,
            mlp_ratio=mlp_ratio,
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
        self.diffusion_config = {
            "name": "rescasformer",
            "image_size": int(image_size),
            "patch_size": int(patch_size),
            "frame_hidden_size": int(frame_hidden_size),
            "frame_depth": int(frame_depth),
            "frame_heads": int(frame_heads),
            "sequence_hidden_size": int(sequence_hidden_size),
            "sequence_depth": int(sequence_depth),
            "sequence_heads": int(sequence_heads),
            "mlp_ratio": float(mlp_ratio),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "diffusion_steps": int(diffusion_steps),
            "prediction_type": str(prediction_type),
            "x0_clip": x0_clip,
            "x0_clip_quantile": x0_clip_quantile,
        }
        if residual_stats_path is not None:
            self.diffusion_config["residual_stats_path"] = str(residual_stats_path)

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

