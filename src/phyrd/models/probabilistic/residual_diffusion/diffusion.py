from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .unet import SinusoidalEmbedding, UNet2D


def _as_frame_channels(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 5 or tensor.shape[2] != 1:
        raise ValueError(f"{name} must have [B,T,1,H,W], got {tuple(tensor.shape)}")
    return tensor[:, :, 0]


class ResidualDenoiser(nn.Module):
    def __init__(
        self,
        input_frames: int = 13,
        output_frames: int = 12,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.input_frames = input_frames
        self.output_frames = output_frames
        embedding_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.unet = UNet2D(
            input_frames + output_frames * 2,
            output_frames,
            base_channels,
            embedding_dim,
        )

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        history: torch.Tensor,
        deterministic: torch.Tensor,
    ) -> torch.Tensor:
        noisy = _as_frame_channels(noisy_residual, "noisy_residual")
        observed = _as_frame_channels(history, "history")
        trend = _as_frame_channels(deterministic, "deterministic")
        if observed.shape[1] != self.input_frames or noisy.shape[1] != self.output_frames:
            raise ValueError("frame counts do not match the denoiser protocol")
        combined = torch.cat((noisy, observed, trend), dim=1)
        predicted_noise = self.unet(combined, self.time_embedding(timestep))
        return predicted_noise.unsqueeze(2)


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((points / steps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(0.0001, 0.999).float()


class GaussianResidualDiffusion(nn.Module):
    """Residual diffusion with stable v-prediction and per-lead normalization.

    The legacy implementation predicted epsilon directly in the raw residual
    domain.  At the terminal cosine timestep this made x0 recovery divide by an
    almost-zero alpha and amplified small denoiser errors by orders of
    magnitude.  The default v-parameterization reconstructs x0 without that
    division.  Residual statistics are buffers so training and sampling always
    use the same train-split-only normalization.
    """

    def __init__(
        self,
        denoiser: ResidualDenoiser,
        diffusion_steps: int = 100,
        *,
        prediction_type: str = "v",
        residual_center: float | list[float] | tuple[float, ...] | None = None,
        residual_scale: float | list[float] | tuple[float, ...] | None = None,
        x0_clip: float | None = 5.0,
        x0_clip_quantile: float | None = 0.995,
    ) -> None:
        super().__init__()
        if diffusion_steps < 4:
            raise ValueError("diffusion_steps must be at least four")
        if prediction_type not in {"v", "epsilon"}:
            raise ValueError("prediction_type must be 'v' or 'epsilon'")
        if x0_clip is not None and x0_clip <= 0:
            raise ValueError("x0_clip must be positive or null")
        if x0_clip_quantile is not None and not 0.5 < x0_clip_quantile <= 1.0:
            raise ValueError("x0_clip_quantile must be in (0.5, 1.0] or null")
        self.denoiser = denoiser
        self.diffusion_steps = diffusion_steps
        self.prediction_type = prediction_type
        self.x0_clip = x0_clip
        self.x0_clip_quantile = x0_clip_quantile
        center = self._lead_stat(
            residual_center, denoiser.output_frames, default=0.0, name="residual_center"
        )
        scale = self._lead_stat(
            residual_scale, denoiser.output_frames, default=1.0, name="residual_scale"
        )
        if torch.any(scale <= 0):
            raise ValueError("residual_scale entries must be positive")
        self.register_buffer("residual_center", center)
        self.register_buffer("residual_scale", scale)
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt())

    @staticmethod
    def _lead_stat(
        value: float | list[float] | tuple[float, ...] | None,
        output_frames: int,
        *,
        default: float,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            result = torch.full((output_frames,), default, dtype=torch.float32)
        else:
            result = torch.as_tensor(value, dtype=torch.float32).flatten()
            if result.numel() == 1:
                result = result.repeat(output_frames)
            elif result.numel() != output_frames:
                raise ValueError(
                    f"{name} must be scalar or have {output_frames} entries, "
                    f"got {result.numel()}"
                )
        return result.reshape(1, output_frames, 1, 1, 1)

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        selected = values.gather(0, timestep)
        return selected.reshape(timestep.shape[0], *((1,) * (ndim - 1)))

    def q_sample(
        self, clean: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, clean.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, clean.ndim)
        return alpha * clean + sigma * noise, noise

    def normalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 5 or residual.shape[1] != self.denoiser.output_frames:
            raise ValueError(
                "residual must have "
                f"[B,{self.denoiser.output_frames},1,H,W], got {tuple(residual.shape)}"
            )
        return (residual - self.residual_center) / self.residual_scale

    def denormalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return residual * self.residual_scale + self.residual_center

    def _stabilize_x0(self, clean: torch.Tensor) -> torch.Tensor:
        """Clip standardized x0 outliers without rescaling ordinary samples."""
        if self.x0_clip is None:
            return clean
        if self.x0_clip_quantile is None:
            limit = clean.new_full((clean.shape[0],), float(self.x0_clip))
        else:
            flat = clean.detach().abs().flatten(1).float()
            limit = torch.quantile(flat, self.x0_clip_quantile, dim=1).to(clean.dtype)
            limit = limit.clamp(min=1.0, max=float(self.x0_clip))
        limit = limit.reshape(clean.shape[0], *((1,) * (clean.ndim - 1)))
        return torch.maximum(torch.minimum(clean, limit), -limit)

    def predict_x0(
        self, noisy: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim)
        if self.prediction_type == "v":
            return alpha * noisy - sigma * model_output
        return (noisy - sigma * model_output) / alpha.clamp_min(1e-8)

    def model_output_to_noise(
        self, noisy: torch.Tensor, model_output: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return model_output
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim)
        return sigma * noisy + alpha * model_output

    def noise_from_x0(
        self, noisy: torch.Tensor, clean: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim)
        return (noisy - alpha * clean) / sigma.clamp_min(1e-8)

    def training_loss(
        self,
        clean_residual: torch.Tensor,
        history: torch.Tensor,
        deterministic: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean_residual.shape[0]
        if timestep is None:
            timestep = torch.randint(
                0, self.diffusion_steps, (batch,), device=clean_residual.device
            )
        clean_normalized = self.normalize_residual(clean_residual)
        noisy, noise = self.q_sample(clean_normalized, timestep)
        model_output = self.denoiser(noisy, timestep, history, deterministic)
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
        sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim)
        target = (
            alpha * noise - sigma * clean_normalized
            if self.prediction_type == "v"
            else noise
        )
        clean_prediction_normalized = self._stabilize_x0(
            self.predict_x0(noisy, model_output, timestep)
        )
        clean_prediction = self.denormalize_residual(clean_prediction_normalized)
        predicted_noise = self.model_output_to_noise(noisy, model_output, timestep)
        return {
            "loss_gen": F.mse_loss(model_output, target),
            "clean_prediction": clean_prediction,
            "clean_prediction_normalized": clean_prediction_normalized,
            "predicted_noise": predicted_noise,
            "model_output": model_output,
            "model_target": target,
            "timestep": timestep,
        }

    @torch.no_grad()
    def ddim_sample(
        self,
        history: torch.Tensor,
        deterministic: torch.Tensor,
        *,
        sampling_steps: int = 20,
        guidance: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not 1 <= sampling_steps <= self.diffusion_steps:
            raise ValueError("sampling_steps must be within the diffusion schedule")
        noisy = torch.randn(
            deterministic.shape,
            device=deterministic.device,
            dtype=deterministic.dtype,
            generator=generator,
        )
        schedule = torch.linspace(
            self.diffusion_steps - 1, 0, sampling_steps, device=deterministic.device
        ).round().long()
        schedule = torch.unique_consecutive(schedule)
        for position, scalar_time in enumerate(schedule):
            timestep = scalar_time.expand(history.shape[0])
            model_output = self.denoiser(noisy, timestep, history, deterministic)
            clean = self._stabilize_x0(self.predict_x0(noisy, model_output, timestep))
            if guidance is not None:
                # Physics guidance operates in the original residual domain.
                with torch.enable_grad():
                    guided = guidance(
                        self.denormalize_residual(clean), int(scalar_time.item())
                    )
                clean = self._stabilize_x0(self.normalize_residual(guided))
            predicted_noise = self.noise_from_x0(noisy, clean, timestep)
            if position == len(schedule) - 1:
                noisy = clean
                break
            next_time = schedule[position + 1].expand(history.shape[0])
            alpha_next = self._extract(self.sqrt_alphas_cumprod, next_time, noisy.ndim)
            sigma_next = self._extract(
                self.sqrt_one_minus_alphas_cumprod, next_time, noisy.ndim
            )
            noisy = alpha_next * clean + sigma_next * predicted_noise
        return self.denormalize_residual(noisy)

    def reparameterization_error(
        self, noisy: torch.Tensor, clean: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        noise = self.noise_from_x0(noisy, clean, timestep)
        if self.prediction_type == "v":
            alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
            sigma = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim)
            model_output = alpha * noise - sigma * clean
        else:
            model_output = noise
        recovered = self.predict_x0(noisy, model_output, timestep)
        return (recovered - clean).abs().max()
