from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..base import ProbabilisticModel


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _time_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(
        torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
    )
    angles = timestep.float()[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, width - embedding.shape[1]))


class FrameEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, _, height, width = sequence.shape
        feature = self.net(sequence.reshape(batch * frames, 1, height, width))
        feature = feature.reshape(batch, frames, *feature.shape[1:])
        return feature, feature.mean(dim=(-1, -2))


class TrajectoryContextAdapter(nn.Module):
    """Cross-attend each generated lead to the complete deterministic trajectory."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.history_encoder = FrameEncoder(channels)
        self.trend_encoder = FrameEncoder(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, batch_first=True, dropout=0.0
        )

    def forward(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        trend: torch.Tensor,
    ) -> torch.Tensor:
        _, history_tokens = self.history_encoder(history)
        _, trend_tokens = self.trend_encoder(trend)
        context = self.context_norm(torch.cat((history_tokens, trend_tokens), dim=1))
        attended, _ = self.attention(self.query_norm(query), context, context)
        return attended


class TemporalResidualDenoiser(nn.Module):
    """2D spatial encoder plus explicit low-resolution temporal processing."""

    def __init__(
        self,
        segment_frames: int,
        *,
        base_channels: int,
        attention_heads: int,
        diffusion_steps: int,
        num_segments: int,
    ) -> None:
        super().__init__()
        self.segment_frames = int(segment_frames)
        channels = int(base_channels)
        self.input_projection = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.trajectory_adapter = TrajectoryContextAdapter(channels, attention_heads)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.segment_embedding = nn.Embedding(num_segments, channels)
        self.temporal = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                (3, 3, 3),
                padding=1,
                groups=channels,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv3d(
                channels,
                channels,
                (3, 3, 3),
                padding=1,
                groups=channels,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        history: torch.Tensor,
        trend: torch.Tensor,
        trend_segment: torch.Tensor,
        prefix: torch.Tensor,
        timestep: torch.Tensor,
        segment_index: int,
    ) -> torch.Tensor:
        batch, frames, _, height, width = noisy.shape
        last_observation = history[:, -1:].expand(-1, frames, -1, -1, -1)
        inputs = torch.cat((noisy, trend_segment, prefix, last_observation), dim=2)
        encoded = self.input_projection(inputs.reshape(batch * frames, 4, height, width))
        low_h, low_w = encoded.shape[-2:]
        encoded = encoded.reshape(batch, frames, -1, low_h, low_w)
        queries = encoded.mean(dim=(-1, -2))
        context = self.trajectory_adapter(queries, history, trend)
        time_condition = self.time_mlp(
            _time_embedding(timestep, encoded.shape[2]).to(encoded.dtype)
        )
        segment_ids = torch.full(
            (batch,), int(segment_index), device=noisy.device, dtype=torch.long
        )
        condition = (
            context
            + time_condition[:, None]
            + self.segment_embedding(segment_ids)[:, None]
        )
        encoded = encoded + condition[..., None, None]
        temporal = self.temporal(encoded.permute(0, 2, 1, 3, 4))
        temporal = temporal.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, -1, low_h, low_w
        )
        temporal = F.interpolate(
            temporal, size=(height, width), mode="bilinear", align_corners=False
        )
        return self.output_projection(temporal).reshape(batch, frames, 1, height, width)


class TrajectoryResidualDiffusionModel(ProbabilisticModel):
    """Segment-autoregressive residual diffusion for deterministic sharpening."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        segment_frames: int = 5,
        base_channels: int = 64,
        attention_heads: int = 4,
        diffusion_steps: int = 100,
        prediction_type: str = "v",
        residual_stats_path: str | None = None,
        residual_center: Sequence[float] | float = 0.0,
        residual_scale: Sequence[float] | float = 0.15,
        prefix_clean_probability: float = 0.25,
        prefix_noisy_probability: float = 0.65,
        prefix_zero_probability: float = 0.10,
        prefix_noise_max: float = 0.35,
        **_: object,
    ) -> None:
        super().__init__()
        if output_frames % segment_frames:
            raise ValueError("output_frames must be divisible by segment_frames")
        if prediction_type != "v":
            raise ValueError("trajres_diffusion currently uses v-prediction only")
        probabilities = (
            float(prefix_clean_probability),
            float(prefix_noisy_probability),
            float(prefix_zero_probability),
        )
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-6
        ):
            raise ValueError("prefix probabilities must be non-negative and sum to one")
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.segment_frames = int(segment_frames)
        self.num_segments = self.output_frames // self.segment_frames
        self.diffusion_steps = int(diffusion_steps)
        self.prediction_type = prediction_type
        self.prefix_probabilities = probabilities
        self.prefix_noise_max = float(prefix_noise_max)
        if residual_stats_path is not None:
            statistics = json.loads(
                Path(residual_stats_path).read_text(encoding="utf-8")
            )
            residual_center = statistics["center"]
            residual_scale = statistics["scale"]
        center = self._lead_tensor(residual_center, "residual_center")
        scale = self._lead_tensor(residual_scale, "residual_scale")
        if torch.any(scale <= 0):
            raise ValueError("residual_scale values must be positive")
        self.register_buffer("residual_center", center)
        self.register_buffer("residual_scale", scale)
        betas = torch.linspace(1e-4, 0.02, self.diffusion_steps, dtype=torch.float32)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )
        self.denoiser = TemporalResidualDenoiser(
            self.segment_frames,
            base_channels=base_channels,
            attention_heads=attention_heads,
            diffusion_steps=self.diffusion_steps,
            num_segments=self.num_segments,
        )
        # ForecastComposer maps its legacy ``diffusion`` checkpoint key to the
        # complete probabilistic module when this marker is present.
        self.checkpoint_as_diffusion = True
        self.diffusion_config = {
            "name": "trajres_diffusion",
            "segment_frames": self.segment_frames,
            "num_segments": self.num_segments,
            "prediction_type": self.prediction_type,
            "diffusion_steps": self.diffusion_steps,
            "prefix_clean_probability": probabilities[0],
            "prefix_noisy_probability": probabilities[1],
            "prefix_zero_probability": probabilities[2],
            "prefix_noise_max": self.prefix_noise_max,
        }
        if residual_stats_path is not None:
            self.diffusion_config["residual_stats_path"] = str(residual_stats_path)

    def _lead_tensor(self, value: Sequence[float] | float, name: str) -> torch.Tensor:
        values = torch.as_tensor(value, dtype=torch.float32).flatten()
        if values.numel() == 1:
            values = values.repeat(self.output_frames)
        if values.numel() != self.output_frames:
            raise ValueError(f"{name} must be scalar or have {self.output_frames} values")
        return values.reshape(1, self.output_frames, 1, 1, 1)

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        return values.gather(0, timestep).reshape(timestep.shape[0], *((1,) * (ndim - 1)))

    def _slice_stats(self, segment_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = segment_index * self.segment_frames
        stop = start + self.segment_frames
        return self.residual_center[:, start:stop], self.residual_scale[:, start:stop]

    def _normalize(self, residual: torch.Tensor, segment_index: int) -> torch.Tensor:
        center, scale = self._slice_stats(segment_index)
        return (residual - center) / scale

    def _denormalize(self, residual: torch.Tensor, segment_index: int) -> torch.Tensor:
        center, scale = self._slice_stats(segment_index)
        return residual * scale + center

    def _training_prefix(
        self, residual: torch.Tensor, segment_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if segment_index == 0:
            return residual.new_zeros(
                residual.shape[0], self.segment_frames, *residual.shape[2:]
            ), residual.new_zeros((), dtype=torch.long)
        start = (segment_index - 1) * self.segment_frames
        clean = self._normalize(residual[:, start : start + self.segment_frames], segment_index - 1)
        draw = torch.rand((), device=residual.device)
        clean_cutoff = self.prefix_probabilities[0]
        noisy_cutoff = clean_cutoff + self.prefix_probabilities[1]
        if draw < clean_cutoff:
            return clean.detach(), residual.new_zeros((), dtype=torch.long)
        if draw < noisy_cutoff:
            strength = torch.rand((), device=residual.device) * self.prefix_noise_max
            noisy = clean + strength * torch.randn_like(clean)
            return noisy.detach(), residual.new_ones((), dtype=torch.long)
        return torch.zeros_like(clean), residual.new_full((), 2, dtype=torch.long)

    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        trend: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual = target - trend
        segment_index = int(torch.randint(self.num_segments, (), device=target.device).item())
        start = segment_index * self.segment_frames
        stop = start + self.segment_frames
        clean = self._normalize(residual[:, start:stop], segment_index)
        prefix, prefix_mode = self._training_prefix(residual, segment_index)
        timestep = torch.randint(
            self.diffusion_steps, (target.shape[0],), device=target.device
        )
        alpha = self._extract(self.sqrt_alphas_cumprod, timestep, clean.ndim)
        sigma = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, clean.ndim
        )
        noise = torch.randn_like(clean)
        noisy = alpha * clean + sigma * noise
        velocity = alpha * noise - sigma * clean
        model_output = self.denoiser(
            noisy,
            history,
            trend,
            trend[:, start:stop],
            prefix,
            timestep,
            segment_index,
        )
        predicted_clean = alpha * noisy - sigma * model_output
        loss = F.mse_loss(model_output, velocity)
        clean_full = torch.zeros_like(residual)
        clean_full[:, start:stop] = self._denormalize(predicted_clean, segment_index)
        return {
            "loss_gen": loss,
            "loss_diff": loss,
            "clean_prediction": clean_full,
            "prediction_x0": trend + clean_full,
            "timestep": timestep,
            "segment_index": target.new_tensor(segment_index, dtype=torch.long),
            "prefix_mode": prefix_mode,
            "residual_abs_mean": residual.detach().abs().mean(),
        }

    @torch.no_grad()
    def _sample_segment(
        self,
        history: torch.Tensor,
        trend: torch.Tensor,
        prefix: torch.Tensor,
        segment_index: int,
        sampling_steps: int,
    ) -> torch.Tensor:
        batch, _, _, height, width = history.shape
        noisy = torch.randn(
            batch,
            self.segment_frames,
            1,
            height,
            width,
            device=history.device,
            dtype=history.dtype,
        )
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            min(int(sampling_steps), self.diffusion_steps),
            device=history.device,
        ).round().long().unique_consecutive()
        start = segment_index * self.segment_frames
        stop = start + self.segment_frames
        for position, scalar_time in enumerate(schedule):
            timestep = scalar_time.expand(batch)
            alpha = self._extract(self.sqrt_alphas_cumprod, timestep, noisy.ndim)
            sigma = self._extract(
                self.sqrt_one_minus_alphas_cumprod, timestep, noisy.ndim
            )
            velocity = self.denoiser(
                noisy,
                history,
                trend,
                trend[:, start:stop],
                prefix,
                timestep,
                segment_index,
            )
            clean = alpha * noisy - sigma * velocity
            noise = sigma * noisy + alpha * velocity
            if position + 1 == len(schedule):
                noisy = clean
            else:
                next_timestep = schedule[position + 1].expand(batch)
                next_alpha = self._extract(
                    self.sqrt_alphas_cumprod, next_timestep, noisy.ndim
                )
                next_sigma = self._extract(
                    self.sqrt_one_minus_alphas_cumprod, next_timestep, noisy.ndim
                )
                noisy = next_alpha * clean + next_sigma * noise
        return noisy

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
        if guidance_factory is not None:
            raise ValueError("trajres_diffusion does not yet support proximal guidance")
        members: list[torch.Tensor] = []
        for _ in range(ensemble_size):
            prefix = history.new_zeros(
                history.shape[0], self.segment_frames, 1, *history.shape[-2:]
            )
            residual_segments: list[torch.Tensor] = []
            for segment_index in range(self.num_segments):
                normalized = self._sample_segment(
                    history, trend, prefix, segment_index, sampling_steps
                )
                residual_segments.append(self._denormalize(normalized, segment_index))
                prefix = normalized
            residual = torch.cat(residual_segments, dim=1)
            members.append((trend + residual).clamp(0.0, 1.0))
        return torch.stack(members, dim=1)
