from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

from ..base import ProbabilisticModel
from .decomposition import DecompositionTargetBuilder, reconstruct
from .denoiser import UDIPDenoiser
from .diffusion import JointGaussianDiffusion


class UDIPModel(ProbabilisticModel):
    """Hot-swappable deformation/intensity probability adapter."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        base_channels: int = 32,
        diffusion_steps: int = 100,
        downsample_factor: int = 4,
        registration_steps: int = 4,
        registration_learning_rate: float = 0.5,
        registration_smoothness_weight: float = 0.05,
        registration_temporal_weight: float = 0.02,
        registration_magnitude_weight: float = 0.001,
        registration_gradient_weight: float = 0.1,
        max_displacement: float = 16.0,
        confidence_scale: float = 0.1,
        reconstruction_weight: float = 0.01,
        reconstruction_timestep_max: int = 50,
        gradient_reconstruction_weight: float = 0.1,
        extensions: list[dict[str, object]] | None = None,
        **diffusion_config: object,
    ) -> None:
        super().__init__()
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.downsample_factor = int(downsample_factor)
        self.reconstruction_weight = float(reconstruction_weight)
        self.reconstruction_timestep_max = int(reconstruction_timestep_max)
        self.gradient_reconstruction_weight = float(gradient_reconstruction_weight)
        self.extensions = list(extensions or [])
        self.target_builder = DecompositionTargetBuilder(
            downsample_factor=downsample_factor,
            steps=registration_steps,
            learning_rate=registration_learning_rate,
            smoothness_weight=registration_smoothness_weight,
            temporal_weight=registration_temporal_weight,
            magnitude_weight=registration_magnitude_weight,
            gradient_weight=registration_gradient_weight,
            max_displacement=max_displacement,
            confidence_scale=confidence_scale,
        )
        denoiser = UDIPDenoiser(
            input_frames,
            output_frames,
            base_channels=base_channels,
            downsample_factor=downsample_factor,
        )
        self.diffusion_config = {
            "model": "udip",
            "downsample_factor": downsample_factor,
            "registration_steps": registration_steps,
            "registration_learning_rate": registration_learning_rate,
            "registration_smoothness_weight": registration_smoothness_weight,
            "registration_temporal_weight": registration_temporal_weight,
            "registration_magnitude_weight": registration_magnitude_weight,
            "registration_gradient_weight": registration_gradient_weight,
            "max_displacement": max_displacement,
            "confidence_scale": confidence_scale,
            "reconstruction_weight": reconstruction_weight,
            "reconstruction_timestep_max": reconstruction_timestep_max,
            "gradient_reconstruction_weight": gradient_reconstruction_weight,
            **diffusion_config,
        }
        self.diffusion = JointGaussianDiffusion(
            denoiser,
            diffusion_steps,
            **diffusion_config,
        )

    @staticmethod
    def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.smooth_l1_loss(prediction_dx, target_dx) + F.smooth_l1_loss(
            prediction_dy, target_dy
        )

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        trend: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        targets = self.target_builder(trend, target)
        result = self.diffusion.training_loss(
            targets.deformation,
            targets.intensity,
            targets.confidence,
            history,
            trend,
        )
        prediction = reconstruct(
            trend, result["clean_deformation"], result["clean_intensity"]
        )
        sample_mask = result["timestep"] <= self.reconstruction_timestep_max
        if sample_mask.any():
            reconstruction_loss = F.smooth_l1_loss(
                prediction[sample_mask], target[sample_mask]
            )
            reconstruction_loss = reconstruction_loss + self.gradient_reconstruction_weight * (
                self._gradient_loss(prediction[sample_mask], target[sample_mask])
            )
        else:
            reconstruction_loss = prediction.new_zeros(())
        result["loss_reconstruction"] = reconstruction_loss
        result["loss_gen"] = (
            result["loss_gen"] + self.reconstruction_weight * reconstruction_loss
        )
        result["trend"] = trend
        result["prediction_x0"] = prediction
        result["clean_prediction"] = prediction - trend
        result["target_deformation"] = targets.deformation
        result["target_intensity"] = targets.intensity
        result["registration_confidence"] = targets.confidence
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
        deformation_size = (
            max(1, trend.shape[-2] // self.downsample_factor),
            max(1, trend.shape[-1] // self.downsample_factor),
        )
        members = []
        for _ in range(ensemble_size):
            deformation, intensity = self.diffusion.ddim_sample(
                history,
                trend,
                deformation_size=deformation_size,
                sampling_steps=sampling_steps,
            )
            prediction = reconstruct(trend, deformation, intensity)
            if guidance_factory is not None:
                guidance = guidance_factory(trend)
                correction = guidance(prediction - trend, 0)
                prediction = trend + correction
            members.append(prediction.clamp(0.0, 1.0))
        return torch.stack(members, dim=1)
