from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def modulate(
    tensor: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return tensor * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    """Sinusoidal diffusion-timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = int(frequency_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(
        timestep: torch.Tensor,
        width: int,
        *,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        half = width // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if width % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.sinusoidal_embedding(timestep, self.frequency_size)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def _sincos_1d(position: torch.Tensor, width: int) -> torch.Tensor:
    if width % 2:
        raise ValueError("one-dimensional positional width must be even")
    half = width // 2
    omega = torch.arange(half, dtype=torch.float64, device=position.device)
    omega = 1.0 / (10_000 ** (omega / max(half, 1)))
    angles = position.reshape(-1, 1).double() * omega.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1).float()


def sincos_2d_position_embedding(
    hidden_size: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Return a fixed [1, H*W, D] two-dimensional sine/cosine embedding."""

    if hidden_size % 4:
        raise ValueError("2D positional hidden_size must be divisible by four")
    rows = torch.arange(grid_height, dtype=torch.float64)
    columns = torch.arange(grid_width, dtype=torch.float64)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    row_embedding = _sincos_1d(row_grid.flatten(), hidden_size // 2)
    column_embedding = _sincos_1d(column_grid.flatten(), hidden_size // 2)
    return torch.cat((row_embedding, column_embedding), dim=1).unsqueeze(0)


class PatchEmbed2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.projection = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[1] != self.in_channels:
            raise ValueError(
                f"patch input must be [B,{self.in_channels},H,W], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2] % self.patch_size or tensor.shape[-1] % self.patch_size:
            raise ValueError("spatial dimensions must be divisible by patch_size")
        tokens = self.projection(tensor)
        return tokens.flatten(2).transpose(1, 2)


def patchify_pixels(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("pixel tensor must have [B,C,H,W]")
    batch, channels, height, width = tensor.shape
    if height % patch_size or width % patch_size:
        raise ValueError("spatial dimensions must be divisible by patch_size")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = tensor.reshape(
        batch,
        channels,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    return (
        patches.permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid_h * grid_w, patch_size * patch_size * channels)
        .contiguous()
    )


def unpatchify_pixels(
    patches: torch.Tensor,
    *,
    patch_size: int,
    channels: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patch tensor must have [B,N,patch_area*C]")
    batch, count, features = patches.shape
    expected_features = patch_size * patch_size * channels
    if count != grid_height * grid_width or features != expected_features:
        raise ValueError(
            "patch shape is incompatible with the requested output grid: "
            f"got {tuple(patches.shape)}"
        )
    tensor = patches.reshape(
        batch,
        grid_height,
        grid_width,
        patch_size,
        patch_size,
        channels,
    )
    return (
        tensor.permute(0, 5, 1, 3, 2, 4)
        .reshape(
            batch,
            channels,
            grid_height * patch_size,
            grid_width * patch_size,
        )
        .contiguous()
    )


class SelfAttention(nn.Module):
    """Multi-head self-attention backed by PyTorch scaled-dot-product attention."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_size = self.hidden_size // self.num_heads
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        qkv = self.qkv(tensor).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_size,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        return self.projection(attended)


class AdaLNZeroDiTBlock(nn.Module):
    """DiT block with zero-initialized adaptive LayerNorm residual gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm_attention = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attention = SelfAttention(hidden_size, num_heads)
        self.norm_mlp = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

    def forward(
        self,
        tensor: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.modulation(condition).chunk(6, dim=-1)
        tensor = tensor + attention_gate[:, None] * self.attention(
            modulate(self.norm_attention(tensor), attention_shift, attention_scale)
        )
        tensor = tensor + mlp_gate[:, None] * self.mlp(
            modulate(self.norm_mlp(tensor), mlp_shift, mlp_scale)
        )
        return tensor


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        self.projection = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
        )

    def forward(
        self,
        tensor: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.projection(modulate(self.norm(tensor), shift, scale))

