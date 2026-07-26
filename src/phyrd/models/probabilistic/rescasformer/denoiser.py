from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    AdaLNZeroDiTBlock,
    FinalLayer,
    PatchEmbed2D,
    TimestepEmbedder,
    sincos_2d_position_embedding,
    unpatchify_pixels,
)


class ResidualCasFormerDenoiser(nn.Module):
    """Frame-aligned CasFormer denoiser for a complete residual forecast.

    The topology is a clean-room implementation of the frame-wise encoder and
    sequence aggregation described by CasCast (ICML 2024).  Each noisy residual
    lead is paired with the matching deterministic lead before shared
    frame-wise DiT processing.  All future leads are then aggregated and
    denoised jointly.
    """

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        image_size: int = 128,
        patch_size: int = 4,
        frame_hidden_size: int = 256,
        frame_depth: int = 12,
        frame_heads: int = 4,
        sequence_hidden_size: int = 1152,
        sequence_depth: int = 12,
        sequence_heads: int = 16,
        mlp_ratio: float = 4.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if frame_depth < 1 or sequence_depth < 1:
            raise ValueError("frame_depth and sequence_depth must be positive")
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.patch_embed = PatchEmbed2D(2, frame_hidden_size, self.patch_size)
        self.frame_timestep = TimestepEmbedder(frame_hidden_size)
        self.sequence_timestep = TimestepEmbedder(sequence_hidden_size)
        self.register_buffer(
            "frame_position",
            sincos_2d_position_embedding(
                frame_hidden_size,
                self.grid_size,
                self.grid_size,
            ),
            persistent=False,
        )
        self.frame_blocks = nn.ModuleList(
            [
                AdaLNZeroDiTBlock(
                    frame_hidden_size,
                    frame_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(frame_depth)
            ]
        )
        self.sequence_projection = nn.Linear(
            frame_hidden_size * self.output_frames,
            sequence_hidden_size,
        )
        self.sequence_blocks = nn.ModuleList(
            [
                AdaLNZeroDiTBlock(
                    sequence_hidden_size,
                    sequence_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(sequence_depth)
            ]
        )
        self.final_layer = FinalLayer(
            sequence_hidden_size,
            patch_size=self.patch_size,
            out_channels=self.output_frames,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        patch_weight = self.patch_embed.projection.weight.data
        nn.init.xavier_uniform_(patch_weight.reshape(patch_weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.projection.bias)

        for embedder in (self.frame_timestep, self.sequence_timestep):
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

        for block in [*self.frame_blocks, *self.sequence_blocks]:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        tensor: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tensor = checkpoint(
                    block,
                    tensor,
                    condition,
                    use_reentrant=False,
                )
            else:
                tensor = block(tensor, condition)
        return tensor

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timestep: torch.Tensor,
        history: torch.Tensor,
        deterministic: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            noisy_residual.ndim == 5
            and noisy_residual.shape[1] == self.output_frames
            and noisy_residual.shape[2] == 1
        )
        if not expected:
            raise ValueError(
                "noisy_residual must have "
                f"[B,{self.output_frames},1,H,W], got {tuple(noisy_residual.shape)}"
            )
        if deterministic.shape != noisy_residual.shape:
            raise ValueError("deterministic forecast must match noisy_residual")
        if (
            history.ndim != 5
            or history.shape[1] != self.input_frames
            or history.shape[2] != 1
        ):
            raise ValueError(
                f"history must have [B,{self.input_frames},1,H,W], "
                f"got {tuple(history.shape)}"
            )
        height, width = noisy_residual.shape[-2:]
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"ResCasFormer was configured for {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )

        batch = noisy_residual.shape[0]
        frames = self.output_frames
        # The residual is standardized by the diffusion wrapper.  Map the
        # deterministic VIL condition from [0,1] to a zero-centered [-1,1]
        # range before concatenation.
        condition = deterministic.mul(2.0).sub(1.0)
        paired = torch.cat((noisy_residual, condition), dim=2).reshape(
            batch * frames,
            2,
            height,
            width,
        )
        frame_tokens = self.patch_embed(paired)
        frame_tokens = frame_tokens + self.frame_position.to(frame_tokens.dtype)
        frame_time = self.frame_timestep(timestep)
        frame_time = (
            frame_time[:, None]
            .expand(batch, frames, frame_time.shape[-1])
            .reshape(batch * frames, frame_time.shape[-1])
        )
        frame_tokens = self._run_blocks(
            self.frame_blocks,
            frame_tokens,
            frame_time,
        )

        patch_count = frame_tokens.shape[1]
        frame_width = frame_tokens.shape[2]
        sequence_tokens = (
            frame_tokens.reshape(batch, frames, patch_count, frame_width)
            .permute(0, 2, 1, 3)
            .reshape(batch, patch_count, frames * frame_width)
            .contiguous()
        )
        sequence_tokens = self.sequence_projection(sequence_tokens)
        sequence_time = self.sequence_timestep(timestep)
        sequence_tokens = self._run_blocks(
            self.sequence_blocks,
            sequence_tokens,
            sequence_time,
        )
        output_patches = self.final_layer(sequence_tokens, sequence_time)
        output = unpatchify_pixels(
            output_patches,
            patch_size=self.patch_size,
            channels=self.output_frames,
            grid_height=self.grid_size,
            grid_width=self.grid_size,
        )
        return output.unsqueeze(2)

