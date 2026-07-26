from __future__ import annotations

import math

import torch
from torch import nn

from .denoiser import UDIPDenoiser


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((points / steps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(0.0001, 0.999).float()


class JointGaussianDiffusion(nn.Module):
    """Joint v-prediction diffusion over deformation and intensity coordinates."""

    def __init__(
        self,
        denoiser: UDIPDenoiser,
        diffusion_steps: int = 100,
        *,
        deformation_scale: float = 8.0,
        intensity_scale: float = 0.1,
        deformation_loss_weight: float = 1.0,
        intensity_loss_weight: float = 1.0,
        confidence_floor: float = 0.1,
        x0_clip: float = 8.0,
        per_frame_timesteps: bool = True,
        history_dropout: float = 0.05,
        anchor_dropout: float = 0.05,
        history_guidance_weight: float = 0.0,
        anchor_guidance_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if diffusion_steps < 4:
            raise ValueError("diffusion_steps must be at least four")
        if deformation_scale <= 0 or intensity_scale <= 0:
            raise ValueError("coordinate scales must be positive")
        if not 0 < confidence_floor <= 1:
            raise ValueError("confidence_floor must be in (0, 1]")
        if x0_clip <= 0:
            raise ValueError("x0_clip must be positive")
        if not 0 <= history_dropout < 1 or not 0 <= anchor_dropout < 1:
            raise ValueError("conditioning dropout probabilities must be in [0, 1)")
        self.denoiser = denoiser
        self.diffusion_steps = int(diffusion_steps)
        self.deformation_scale = float(deformation_scale)
        self.intensity_scale = float(intensity_scale)
        self.deformation_loss_weight = float(deformation_loss_weight)
        self.intensity_loss_weight = float(intensity_loss_weight)
        self.confidence_floor = float(confidence_floor)
        self.x0_clip = float(x0_clip)
        self.per_frame_timesteps = bool(per_frame_timesteps)
        self.history_dropout = float(history_dropout)
        self.anchor_dropout = float(anchor_dropout)
        self.history_guidance_weight = float(history_guidance_weight)
        self.anchor_guidance_weight = float(anchor_guidance_weight)
        betas = cosine_beta_schedule(diffusion_steps)
        cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        selected = values[timestep]
        return selected.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _sample_timesteps(self, batch: int, frames: int, device: torch.device) -> torch.Tensor:
        shape = (batch, frames) if self.per_frame_timesteps else (batch, 1)
        timestep = torch.randint(0, self.diffusion_steps, shape, device=device)
        return timestep.expand(batch, frames)

    def normalize_coordinates(
        self,
        deformation: torch.Tensor,
        intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return deformation / self.deformation_scale, intensity / self.intensity_scale

    def denormalize_coordinates(
        self,
        deformation: torch.Tensor,
        intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return deformation * self.deformation_scale, intensity * self.intensity_scale

    def q_sample(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep)
        return alpha * clean + sigma * noise, noise

    def predict_x0(
        self,
        noisy: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep)
        return (alpha * noisy - sigma * model_output).clamp(-self.x0_clip, self.x0_clip)

    def noise_from_x0(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep)
        return (noisy - alpha * clean) / sigma.clamp_min(1e-8)

    def training_loss(
        self,
        clean_deformation: torch.Tensor,
        clean_intensity: torch.Tensor,
        confidence: torch.Tensor,
        history: torch.Tensor,
        trend: torch.Tensor,
        frame_timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, frames = clean_deformation.shape[:2]
        if frame_timestep is None:
            frame_timestep = self._sample_timesteps(batch, frames, clean_deformation.device)
        if frame_timestep.shape != (batch, frames):
            raise ValueError("frame_timestep must have [B,T]")
        clean_d, clean_a = self.normalize_coordinates(clean_deformation, clean_intensity)
        noisy_d, noise_d = self.q_sample(clean_d, frame_timestep)
        noisy_a, noise_a = self.q_sample(clean_a, frame_timestep)
        history_condition = history
        trend_condition = trend
        if self.training and self.history_dropout > 0:
            keep = torch.rand(batch, 1, 1, 1, 1, device=history.device)
            history_condition = history * (keep >= self.history_dropout)
        if self.training and self.anchor_dropout > 0:
            keep = torch.rand(batch, 1, 1, 1, 1, device=trend.device)
            trend_condition = trend * (keep >= self.anchor_dropout)
        output_d, output_a = self.denoiser(
            noisy_d,
            noisy_a,
            frame_timestep,
            history_condition,
            trend_condition,
        )
        alpha = self._extract(self.sqrt_alphas_cumprod, frame_timestep)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, frame_timestep)
        target_d = alpha * noise_d - sigma * clean_d
        target_a = alpha * noise_a - sigma * clean_a
        confidence_weight = self.confidence_floor + (1.0 - self.confidence_floor) * confidence
        loss_d = (
            confidence_weight * (output_d - target_d).square()
        ).sum() / (confidence_weight.sum() * output_d.shape[2]).clamp_min(1.0)
        loss_a = (output_a - target_a).square().mean()
        predicted_d = self.predict_x0(noisy_d, output_d, frame_timestep)
        predicted_a = self.predict_x0(noisy_a, output_a, frame_timestep)
        predicted_d, predicted_a = self.denormalize_coordinates(predicted_d, predicted_a)
        return {
            "loss_gen": (
                self.deformation_loss_weight * loss_d
                + self.intensity_loss_weight * loss_a
            ),
            "loss_deformation": loss_d,
            "loss_intensity": loss_a,
            "clean_deformation": predicted_d,
            "clean_intensity": predicted_a,
            "frame_timestep": frame_timestep,
            # Compatibility with the existing sample-level physics mask. U-DIP
            # configs keep physics disabled, but the field remains well-defined.
            "timestep": frame_timestep.max(dim=1).values,
        }

    @torch.no_grad()
    def _guided_output(
        self,
        noisy_deformation: torch.Tensor,
        noisy_intensity: torch.Tensor,
        timestep: torch.Tensor,
        history: torch.Tensor,
        trend: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional_d, conditional_a = self.denoiser(
            noisy_deformation, noisy_intensity, timestep, history, trend
        )
        output_d, output_a = conditional_d, conditional_a
        if self.history_guidance_weight != 0:
            without_history_d, without_history_a = self.denoiser(
                noisy_deformation,
                noisy_intensity,
                timestep,
                torch.zeros_like(history),
                trend,
            )
            output_d = output_d + self.history_guidance_weight * (
                conditional_d - without_history_d
            )
            output_a = output_a + self.history_guidance_weight * (
                conditional_a - without_history_a
            )
        if self.anchor_guidance_weight != 0:
            without_anchor_d, without_anchor_a = self.denoiser(
                noisy_deformation,
                noisy_intensity,
                timestep,
                history,
                torch.zeros_like(trend),
            )
            output_d = output_d + self.anchor_guidance_weight * (
                conditional_d - without_anchor_d
            )
            output_a = output_a + self.anchor_guidance_weight * (
                conditional_a - without_anchor_a
            )
        return output_d, output_a

    @torch.no_grad()
    def ddim_sample(
        self,
        history: torch.Tensor,
        trend: torch.Tensor,
        *,
        deformation_size: tuple[int, int],
        sampling_steps: int = 20,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= sampling_steps <= self.diffusion_steps:
            raise ValueError("sampling_steps must be within the diffusion schedule")
        batch, frames, _, height, width = trend.shape
        noisy_d = torch.randn(
            (batch, frames, 2, *deformation_size),
            device=trend.device,
            dtype=trend.dtype,
            generator=generator,
        )
        noisy_a = torch.randn(
            (batch, frames, 1, height, width),
            device=trend.device,
            dtype=trend.dtype,
            generator=generator,
        )
        schedule = torch.linspace(
            self.diffusion_steps - 1, 0, sampling_steps, device=trend.device
        ).round().long()
        schedule = torch.unique_consecutive(schedule)
        for position, scalar_time in enumerate(schedule):
            timestep = scalar_time.expand(batch, frames)
            output_d, output_a = self._guided_output(
                noisy_d, noisy_a, timestep, history, trend
            )
            clean_d = self.predict_x0(noisy_d, output_d, timestep)
            clean_a = self.predict_x0(noisy_a, output_a, timestep)
            if position == len(schedule) - 1:
                noisy_d, noisy_a = clean_d, clean_a
                break
            noise_d = self.noise_from_x0(noisy_d, clean_d, timestep)
            noise_a = self.noise_from_x0(noisy_a, clean_a, timestep)
            next_timestep = schedule[position + 1].expand(batch, frames)
            alpha_next = self._extract(self.sqrt_alphas_cumprod, next_timestep)
            sigma_next = self._extract(
                self.sqrt_one_minus_alphas_cumprod, next_timestep
            )
            noisy_d = alpha_next * clean_d + sigma_next * noise_d
            noisy_a = alpha_next * clean_a + sigma_next * noise_a
        return self.denormalize_coordinates(noisy_d, noisy_a)
