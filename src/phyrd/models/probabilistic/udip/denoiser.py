from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        if dimensions < 4 or dimensions % 2:
            raise ValueError("embedding dimensions must be even and at least four")
        self.dimensions = dimensions

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimensions // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float().unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class VideoBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, (3, 3, 3), padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs) + self.skip(inputs)


class UDIPDenoiser(nn.Module):
    """Coupled deformation/intensity denoiser with explicit temporal mixing."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        base_channels: int = 32,
        downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by eight")
        if downsample_factor != 4:
            raise ValueError("the initial U-DIP denoiser requires downsample_factor=4")
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.downsample_factor = int(downsample_factor)
        embedding_dim = base_channels * 4
        self.noise_embedding = nn.Sequential(
            SinusoidalEmbedding(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, base_channels * 4),
        )
        self.lead_embedding = nn.Embedding(output_frames, base_channels * 4)

        # Inputs: noisy intensity, trend, last observation, history mean.
        self.intensity_in = VideoBlock(4, base_channels)
        self.intensity_down1 = nn.Conv3d(
            base_channels, base_channels * 2, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.intensity_mid1 = VideoBlock(base_channels * 2, base_channels * 2)
        self.intensity_down2 = nn.Conv3d(
            base_channels * 2,
            base_channels * 4,
            (1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        # Inputs: noisy dx/dy and the same three conditions at deformation scale.
        self.deformation_in = VideoBlock(5, base_channels * 2)
        self.deformation_mid = VideoBlock(base_channels * 2, base_channels * 4)
        self.coupling = VideoBlock(base_channels * 8, base_channels * 4)
        self.temporal = VideoBlock(base_channels * 4, base_channels * 4)

        self.deformation_out = nn.Conv3d(base_channels * 4, 2, 1)
        self.intensity_up2 = nn.ConvTranspose3d(
            base_channels * 4,
            base_channels * 2,
            (1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.intensity_dec2 = VideoBlock(base_channels * 4, base_channels * 2)
        self.intensity_up1 = nn.ConvTranspose3d(
            base_channels * 2,
            base_channels,
            (1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.intensity_dec1 = VideoBlock(base_channels * 2, base_channels)
        self.intensity_out = nn.Conv3d(base_channels, 1, 1)
        nn.init.zeros_(self.deformation_out.weight)
        nn.init.zeros_(self.deformation_out.bias)
        nn.init.zeros_(self.intensity_out.weight)
        nn.init.zeros_(self.intensity_out.bias)

    @staticmethod
    def _as_channels_first(video: torch.Tensor) -> torch.Tensor:
        return video.transpose(1, 2)

    def _conditions(
        self,
        history: torch.Tensor,
        trend: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 5 or trend.ndim != 5:
            raise ValueError("history and trend must be five-dimensional videos")
        if history.shape[1] != self.input_frames or trend.shape[1] != self.output_frames:
            raise ValueError("frame counts do not match the U-DIP denoiser protocol")
        last = history[:, -1:].expand(-1, self.output_frames, -1, -1, -1)
        mean = history.mean(dim=1, keepdim=True).expand(-1, self.output_frames, -1, -1, -1)
        return torch.cat((trend, last, mean), dim=2)

    def forward(
        self,
        noisy_deformation: torch.Tensor,
        noisy_intensity: torch.Tensor,
        frame_timestep: torch.Tensor,
        history: torch.Tensor,
        trend: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_timestep.shape != (history.shape[0], self.output_frames):
            raise ValueError("frame_timestep must have [B,T]")
        condition = self._conditions(history, trend)
        intensity_input = torch.cat((noisy_intensity, condition), dim=2)
        intensity_input = self._as_channels_first(intensity_input)
        skip1 = self.intensity_in(intensity_input)
        skip2 = self.intensity_mid1(self.intensity_down1(skip1))
        intensity_low = self.intensity_down2(skip2)

        low_size = noisy_deformation.shape[-2:]
        if intensity_low.shape[-2:] != low_size:
            intensity_low = F.interpolate(
                intensity_low,
                size=(self.output_frames, *low_size),
                mode="trilinear",
                align_corners=False,
            )
        low_condition = F.adaptive_avg_pool3d(
            self._as_channels_first(condition), (self.output_frames, *low_size)
        )
        deformation_input = torch.cat(
            (self._as_channels_first(noisy_deformation), low_condition), dim=1
        )
        deformation_low = self.deformation_mid(self.deformation_in(deformation_input))
        hidden = self.coupling(torch.cat((intensity_low, deformation_low), dim=1))

        noise_embedding = self.noise_embedding(frame_timestep)
        lead_index = torch.arange(self.output_frames, device=history.device)
        embedding = noise_embedding + self.lead_embedding(lead_index).unsqueeze(0)
        hidden = hidden + embedding.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        hidden = self.temporal(hidden)

        predicted_deformation = self.deformation_out(hidden).transpose(1, 2)
        decoded = self.intensity_up2(hidden)
        if decoded.shape[-2:] != skip2.shape[-2:]:
            decoded = F.interpolate(
                decoded,
                size=skip2.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        decoded = self.intensity_dec2(torch.cat((decoded, skip2), dim=1))
        decoded = self.intensity_up1(decoded)
        if decoded.shape[-2:] != skip1.shape[-2:]:
            decoded = F.interpolate(
                decoded,
                size=skip1.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        predicted_intensity = self.intensity_out(
            self.intensity_dec1(torch.cat((decoded, skip1), dim=1))
        ).transpose(1, 2)
        return predicted_deformation, predicted_intensity
