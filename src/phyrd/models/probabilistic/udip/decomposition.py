from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _check_video(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 5 or tensor.shape[2] != 1:
        raise ValueError(f"{name} must have [B,T,1,H,W], got {tuple(tensor.shape)}")


def _check_deformation(deformation: torch.Tensor, reference: torch.Tensor) -> None:
    if deformation.ndim != 5 or deformation.shape[2] != 2:
        raise ValueError(
            "deformation must have [B,T,2,h,w], "
            f"got {tuple(deformation.shape)}"
        )
    if deformation.shape[:2] != reference.shape[:2]:
        raise ValueError("deformation and reference must share batch and time dimensions")


def warp_video(reference: torch.Tensor, deformation: torch.Tensor) -> torch.Tensor:
    """Warp a video with low-resolution ``(dx, dy)`` displacements in pixel units."""
    _check_video(reference, "reference")
    _check_deformation(deformation, reference)
    batch, frames, _, height, width = reference.shape
    output_dtype = reference.dtype
    compute_dtype = (
        torch.float32
        if output_dtype in {torch.float16, torch.bfloat16}
        else output_dtype
    )
    displacement = F.interpolate(
        deformation.flatten(0, 1).to(compute_dtype),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).unflatten(0, (batch, frames))
    y = torch.linspace(
        -1.0, 1.0, height, device=reference.device, dtype=compute_dtype
    )
    x = torch.linspace(
        -1.0, 1.0, width, device=reference.device, dtype=compute_dtype
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch * frames, -1, -1, -1).clone()
    flat_displacement = displacement.flatten(0, 1)
    grid[..., 0] += 2.0 * flat_displacement[:, 0] / max(width - 1, 1)
    grid[..., 1] += 2.0 * flat_displacement[:, 1] / max(height - 1, 1)
    warped = F.grid_sample(
        reference.flatten(0, 1).to(compute_dtype),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.unflatten(0, (batch, frames)).to(output_dtype)


def reconstruct(
    trend: torch.Tensor,
    deformation: torch.Tensor,
    intensity: torch.Tensor,
    *,
    clamp: bool = False,
) -> torch.Tensor:
    _check_video(intensity, "intensity")
    if intensity.shape != trend.shape:
        raise ValueError("trend and intensity must have identical shapes")
    prediction = warp_video(trend, deformation) + intensity
    return prediction.clamp(0.0, 1.0) if clamp else prediction


def spatial_gradient(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = video[..., :, 1:] - video[..., :, :-1]
    dy = video[..., 1:, :] - video[..., :-1, :]
    return dx, dy


@dataclass(frozen=True)
class DecompositionTargets:
    deformation: torch.Tensor
    intensity: torch.Tensor
    confidence: torch.Tensor
    warped_trend: torch.Tensor


class DecompositionTargetBuilder:
    """Build an operational deformation/intensity coordinate by soft registration.

    The returned deformation is a training coordinate, not a physical wind field.
    Optimization is detached from the forecasting network and can later be replaced
    by an offline target bank without changing the U-DIP model interface.
    """

    def __init__(
        self,
        *,
        downsample_factor: int = 4,
        steps: int = 4,
        learning_rate: float = 0.5,
        smoothness_weight: float = 0.05,
        temporal_weight: float = 0.02,
        magnitude_weight: float = 0.001,
        gradient_weight: float = 0.1,
        max_displacement: float = 16.0,
        confidence_scale: float = 0.1,
    ) -> None:
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be positive")
        if steps < 0:
            raise ValueError("steps cannot be negative")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_displacement <= 0 or confidence_scale <= 0:
            raise ValueError("max_displacement and confidence_scale must be positive")
        self.downsample_factor = int(downsample_factor)
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.smoothness_weight = float(smoothness_weight)
        self.temporal_weight = float(temporal_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.gradient_weight = float(gradient_weight)
        self.max_displacement = float(max_displacement)
        self.confidence_scale = float(confidence_scale)

    def _registration_loss(
        self,
        warped: torch.Tensor,
        target: torch.Tensor,
        deformation: torch.Tensor,
    ) -> torch.Tensor:
        photometric = torch.sqrt((warped - target).square() + 1e-6).mean()
        warped_dx, warped_dy = spatial_gradient(warped)
        target_dx, target_dy = spatial_gradient(target)
        gradient = (warped_dx - target_dx).abs().mean()
        gradient = gradient + (warped_dy - target_dy).abs().mean()
        deformation_dx = deformation[..., :, 1:] - deformation[..., :, :-1]
        deformation_dy = deformation[..., 1:, :] - deformation[..., :-1, :]
        smoothness = deformation_dx.abs().mean() + deformation_dy.abs().mean()
        temporal = deformation.new_zeros(())
        if deformation.shape[1] > 1:
            temporal = (deformation[:, 1:] - deformation[:, :-1]).abs().mean()
        magnitude = deformation.abs().mean()
        return (
            photometric
            + self.gradient_weight * gradient
            + self.smoothness_weight * smoothness
            + self.temporal_weight * temporal
            + self.magnitude_weight * magnitude
        )

    def __call__(self, trend: torch.Tensor, target: torch.Tensor) -> DecompositionTargets:
        _check_video(trend, "trend")
        _check_video(target, "target")
        if trend.shape != target.shape:
            raise ValueError("trend and target must have identical shapes")
        trend = trend.detach().float()
        target = target.detach().float()
        batch, frames, _, height, width = trend.shape
        low_height = max(1, height // self.downsample_factor)
        low_width = max(1, width // self.downsample_factor)
        deformation = trend.new_zeros(batch, frames, 2, low_height, low_width)
        with torch.enable_grad():
            for _ in range(self.steps):
                deformation.requires_grad_(True)
                warped = warp_video(trend, deformation)
                loss = self._registration_loss(warped, target, deformation)
                (gradient,) = torch.autograd.grad(loss, deformation)
                # The image loss is averaged over many pixels, so its raw
                # displacement gradient shrinks with resolution and batch size.
                # RMS normalization gives ``learning_rate`` a stable pixel-unit
                # meaning without keeping optimizer state inside the target builder.
                gradient_scale = gradient.square().mean(
                    dim=(2, 3, 4), keepdim=True
                ).sqrt().clamp_min(1e-6)
                deformation = (
                    deformation - self.learning_rate * gradient / gradient_scale
                ).detach()
                deformation.clamp_(-self.max_displacement, self.max_displacement)
        warped = warp_video(trend, deformation)
        intensity = target - warped
        error = (target - warped).abs().flatten(0, 1)
        confidence = torch.exp(-error / self.confidence_scale)
        confidence = F.adaptive_avg_pool2d(confidence, (low_height, low_width))
        confidence = confidence.unflatten(0, (batch, frames)).detach()
        return DecompositionTargets(
            deformation=deformation.detach(),
            intensity=intensity.detach(),
            confidence=confidence,
            warped_trend=warped.detach(),
        )
