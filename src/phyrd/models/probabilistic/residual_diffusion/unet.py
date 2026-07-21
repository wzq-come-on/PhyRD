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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs) + self.skip(inputs)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        if dimensions < 4 or dimensions % 2:
            raise ValueError("sinusoidal dimensions must be even and at least four")
        self.dimensions = dimensions

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimensions // 2
        scale = math.log(10_000) / (half - 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=time.device, dtype=torch.float32)
        )
        angles = time.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class UNet2D(nn.Module):
    """Compact 2D U-Net used with forecast time represented as channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by eight")
        base = base_channels
        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.embedding = (
            nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, base))
            if embedding_dim is not None
            else None
        )
        self.enc1 = ConvBlock(base, base)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.middle = ConvBlock(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(
        self, inputs: torch.Tensor, embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.in_conv(inputs)
        if self.embedding is not None:
            if embedding is None:
                raise ValueError("this U-Net requires a timestep embedding")
            hidden = hidden + self.embedding(embedding)[:, :, None, None]
        skip1 = self.enc1(hidden)
        skip2 = self.enc2(self.down1(skip1))
        hidden = self.middle(self.down2(skip2))
        hidden = self.up2(hidden)
        if hidden.shape[-2:] != skip2.shape[-2:]:
            hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(torch.cat((hidden, skip2), dim=1))
        hidden = self.up1(hidden)
        if hidden.shape[-2:] != skip1.shape[-2:]:
            hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec1(torch.cat((hidden, skip1), dim=1)))

