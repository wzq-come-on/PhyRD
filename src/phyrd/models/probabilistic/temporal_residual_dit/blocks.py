from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..rescasformer.blocks import AdaLNZeroDiTBlock, modulate


class CrossAttention(nn.Module):
    """Scaled dot-product cross attention with explicit query and context."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_size = self.hidden_size // self.num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key_value = nn.Linear(hidden_size, hidden_size * 2)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_tokens, _ = tensor.shape
        context_tokens = context.shape[1]
        query = self.query(tensor).reshape(
            batch, query_tokens, self.num_heads, self.head_size
        )
        key, value = self.key_value(context).reshape(
            batch, context_tokens, 2, self.num_heads, self.head_size
        ).permute(2, 0, 3, 1, 4).unbind(0)
        query = query.permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, query_tokens, self.hidden_size
        )
        return self.projection(attended)


class AdaLNCrossAttentionBlock(nn.Module):
    """Diffusion-timestep-conditioned cross attention and feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.query_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.context_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = CrossAttention(hidden_size, num_heads)
        self.mlp_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
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
        context: torch.Tensor,
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
        query = modulate(
            self.query_norm(tensor), attention_shift, attention_scale
        )
        tensor = tensor + attention_gate[:, None] * self.attention(
            query, self.context_norm(context)
        )
        tensor = tensor + mlp_gate[:, None] * self.mlp(
            modulate(self.mlp_norm(tensor), mlp_shift, mlp_scale)
        )
        return tensor


class FactorizedSpatiotemporalBlock(nn.Module):
    """Spatial attention, true lead-time attention, then trajectory context."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.spatial = AdaLNZeroDiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio
        )
        self.temporal = AdaLNZeroDiTBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio
        )
        self.context = AdaLNCrossAttentionBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio
        )

    def forward(
        self,
        tensor: torch.Tensor,
        context: torch.Tensor,
        diffusion_condition: torch.Tensor,
        lead_condition: torch.Tensor,
    ) -> torch.Tensor:
        batch, leads, patches, width = tensor.shape
        spatial = tensor.reshape(batch * leads, patches, width)
        spatial_condition = (
            diffusion_condition[:, None] + lead_condition
        ).reshape(batch * leads, width)
        tensor = self.spatial(spatial, spatial_condition).reshape(
            batch, leads, patches, width
        )

        temporal = tensor.permute(0, 2, 1, 3).reshape(
            batch * patches, leads, width
        )
        temporal_condition = (
            diffusion_condition[:, None]
            .expand(batch, patches, width)
            .reshape(batch * patches, width)
        )
        temporal = self.temporal(temporal, temporal_condition)
        temporal = self.context(temporal, context, temporal_condition)
        return temporal.reshape(batch, patches, leads, width).permute(
            0, 2, 1, 3
        ).contiguous()
