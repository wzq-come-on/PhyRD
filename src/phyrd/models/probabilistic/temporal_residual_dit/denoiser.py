from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..rescasformer.blocks import (
    FinalLayer,
    PatchEmbed2D,
    TimestepEmbedder,
    sincos_2d_position_embedding,
    unpatchify_pixels,
)
from .blocks import (
    AdaLNCrossAttentionBlock,
    FactorizedSpatiotemporalBlock,
)


def spatial_high_pass(tensor: torch.Tensor) -> torch.Tensor:
    """Remove the local 3x3 mean while preserving the input tensor shape."""

    if tensor.ndim != 4:
        raise ValueError("high-pass input must have [B,C,H,W]")
    return tensor - F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1)


class HighFrequencyResidualHead(nn.Module):
    """Pixel-space branch that restores details lost by patch tokenization."""

    def __init__(self, input_channels: int, hidden_channels: int = 48) -> None:
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        correction, gate = self.net(tensor).chunk(2, dim=1)
        correction = spatial_high_pass(correction)
        return torch.sigmoid(gate) * correction


class TemporalResidualDenoiser(nn.Module):
    """Factorized spatiotemporal residual denoiser with trajectory context.

    Future leads remain explicit tokens throughout the network.  At every
    spatial patch, future residual queries attend to all observed history
    frames and all deterministic future leads.  A pixel-space high-frequency
    branch restores structures below the patch scale.
    """

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        image_size: int = 128,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        high_frequency_channels: int = 48,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if depth < 1:
            raise ValueError("depth must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.hidden_size = int(hidden_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # noisy residual, deterministic trend, trend high-pass, last-history high-pass
        self.future_patch_embed = PatchEmbed2D(4, hidden_size, patch_size)
        self.history_patch_embed = PatchEmbed2D(1, hidden_size, patch_size)
        self.trend_patch_embed = PatchEmbed2D(1, hidden_size, patch_size)
        self.timestep_embed = TimestepEmbedder(hidden_size)
        self.history_lead_embedding = nn.Parameter(
            torch.zeros(1, self.input_frames, hidden_size)
        )
        self.future_lead_embedding = nn.Parameter(
            torch.zeros(1, self.output_frames, hidden_size)
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 2, hidden_size))
        self.register_buffer(
            "spatial_position",
            sincos_2d_position_embedding(
                hidden_size, self.grid_size, self.grid_size
            ),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                FactorizedSpatiotemporalBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size=patch_size, out_channels=1
        )
        self.high_frequency_head = HighFrequencyResidualHead(
            4, hidden_channels=high_frequency_channels
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        for patch_embed in (
            self.future_patch_embed,
            self.history_patch_embed,
            self.trend_patch_embed,
        ):
            weight = patch_embed.projection.weight.data
            nn.init.xavier_uniform_(weight.reshape(weight.shape[0], -1))
            nn.init.zeros_(patch_embed.projection.bias)
        nn.init.normal_(self.history_lead_embedding, std=0.02)
        nn.init.normal_(self.future_lead_embedding, std=0.02)
        nn.init.normal_(self.context_type_embedding, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)
        for block in self.blocks:
            for conditioned in (block.spatial, block.temporal, block.context):
                nn.init.zeros_(conditioned.modulation[-1].weight)
                nn.init.zeros_(conditioned.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        # Near-zero rather than exactly-zero output lets all gates receive
        # gradients on the first optimization step.
        nn.init.normal_(self.final_layer.projection.weight, std=1e-4)
        nn.init.zeros_(self.final_layer.projection.bias)
        high_output = self.high_frequency_head.net[-1]
        assert isinstance(high_output, nn.Conv2d)
        nn.init.zeros_(high_output.weight)
        nn.init.zeros_(high_output.bias)
        high_output.bias.data[1] = -2.0

    def _context_tokens(
        self,
        history: torch.Tensor,
        deterministic: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _, height, width = history.shape
        patches = self.grid_size * self.grid_size
        history_tokens = self.history_patch_embed(
            history.reshape(batch * self.input_frames, 1, height, width)
        ).reshape(batch, self.input_frames, patches, self.hidden_size)
        trend_tokens = self.trend_patch_embed(
            deterministic.reshape(batch * self.output_frames, 1, height, width)
        ).reshape(batch, self.output_frames, patches, self.hidden_size)
        spatial = self.spatial_position.to(history_tokens.dtype)[:, None]
        history_tokens = (
            history_tokens
            + spatial
            + self.history_lead_embedding[:, :, None].to(history_tokens.dtype)
            + self.context_type_embedding[:, 0:1, None].to(history_tokens.dtype)
        )
        trend_tokens = (
            trend_tokens
            + spatial
            + self.future_lead_embedding[:, :, None].to(trend_tokens.dtype)
            + self.context_type_embedding[:, 1:2, None].to(trend_tokens.dtype)
        )
        return torch.cat((history_tokens, trend_tokens), dim=1).permute(
            0, 2, 1, 3
        ).reshape(
            batch * patches,
            self.input_frames + self.output_frames,
            self.hidden_size,
        ).contiguous()

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        history: torch.Tensor,
        deterministic: torch.Tensor,
    ) -> torch.Tensor:
        expected_residual = (
            noisy_residual.ndim == 5
            and noisy_residual.shape[1] == self.output_frames
            and noisy_residual.shape[2] == 1
        )
        if not expected_residual or deterministic.shape != noisy_residual.shape:
            raise ValueError(
                "noisy residual and deterministic forecast must have "
                f"[B,{self.output_frames},1,H,W]"
            )
        if (
            history.ndim != 5
            or history.shape[1] != self.input_frames
            or history.shape[2] != 1
        ):
            raise ValueError(
                f"history must have [B,{self.input_frames},1,H,W]"
            )
        height, width = noisy_residual.shape[-2:]
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"configured for {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )

        batch = noisy_residual.shape[0]
        leads = self.output_frames
        trend = deterministic.mul(2.0).sub(1.0)
        history_centered = history.mul(2.0).sub(1.0)
        trend_2d = trend.reshape(batch * leads, 1, height, width)
        last_history = history_centered[:, -1:].expand(
            -1, leads, -1, -1, -1
        ).reshape(batch * leads, 1, height, width)
        paired = torch.cat(
            (
                noisy_residual.reshape(batch * leads, 1, height, width),
                trend_2d,
                spatial_high_pass(trend_2d),
                spatial_high_pass(last_history),
            ),
            dim=1,
        )
        patches = self.grid_size * self.grid_size
        tokens = self.future_patch_embed(paired).reshape(
            batch, leads, patches, self.hidden_size
        )
        tokens = (
            tokens
            + self.spatial_position.to(tokens.dtype)[:, None]
            + self.future_lead_embedding[:, :, None].to(tokens.dtype)
        )
        context = self._context_tokens(history_centered, trend)
        diffusion_condition = self.timestep_embed(timestep)
        lead_condition = self.future_lead_embedding.to(diffusion_condition.dtype)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(
                    block,
                    tokens,
                    context,
                    diffusion_condition,
                    lead_condition,
                    use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens,
                    context,
                    diffusion_condition,
                    lead_condition,
                )

        flat_tokens = tokens.reshape(batch * leads, patches, self.hidden_size)
        final_condition = (
            diffusion_condition[:, None] + lead_condition
        ).reshape(batch * leads, self.hidden_size)
        output_patches = self.final_layer(flat_tokens, final_condition)
        base = unpatchify_pixels(
            output_patches,
            patch_size=self.patch_size,
            channels=1,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
        )
        high_frequency = self.high_frequency_head(paired)
        return (base + high_frequency).reshape(
            batch, leads, 1, height, width
        )
