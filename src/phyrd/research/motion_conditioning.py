"""Small controlled probes for choosing a motion-conditioning representation.

These modules deliberately do not register as production probabilistic models.  They
share one patch-level residual predictor and differ only in the additional temporal
condition.  This makes a short real-data run useful for deciding between a
DiffCast-GlobalNet-style recurrent context and a Tora-style motion AdaLN adapter.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _future_previous(history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
    return torch.cat((history[:, -1:], trend[:, :-1]), dim=1)


def _spatial_gradients(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = frames.flatten(0, 1)
    gx = F.pad(flat[..., 1:] - flat[..., :-1], (0, 1, 0, 0))
    gy = F.pad(flat[..., 1:, :] - flat[..., :-1, :], (0, 0, 0, 1))
    return gx.unflatten(0, frames.shape[:2]), gy.unflatten(0, frames.shape[:2])


class PatchResidualStem(nn.Module):
    """Shared frame-wise predictor used by every probe variant."""

    def __init__(self, hidden_size: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        groups = 8 if hidden_size % 8 == 0 else 1
        self.patch_embed = nn.Conv2d(
            2, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.body = nn.Sequential(
            nn.GroupNorm(groups, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.GroupNorm(groups, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        )
        self.decoder = nn.ConvTranspose2d(
            hidden_size, 1, kernel_size=patch_size, stride=patch_size
        )

    def encode(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        batch, leads, _, height, width = trend.shape
        last = history[:, -1:].expand(-1, leads, -1, -1, -1)
        paired = torch.cat((trend, last), dim=2).reshape(
            batch * leads, 2, height, width
        )
        features = self.patch_embed(paired)
        return (features + self.body(features)).unflatten(0, (batch, leads))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        batch, leads, channels, grid_h, grid_w = features.shape
        residual = self.decoder(
            features.reshape(batch * leads, channels, grid_h, grid_w)
        )
        return residual.unflatten(0, (batch, leads))

    def forward(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(history, trend))


class MotionProbeBase(nn.Module):
    variant = "baseline"

    def __init__(self, hidden_size: int = 32, patch_size: int = 4) -> None:
        super().__init__()
        self.stem = PatchResidualStem(hidden_size, patch_size)

    def forward(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        return self.stem(history, trend)


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, tensor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        reset, update = self.gates(torch.cat((tensor, state), dim=1)).sigmoid().chunk(2, dim=1)
        candidate = self.candidate(torch.cat((tensor, reset * state), dim=1)).tanh()
        return (1.0 - update) * state + update * candidate


class GlobalNetResidualProbe(MotionProbeBase):
    """One-scale approximation of DiffCast GlobalNet at the DiT patch grid."""

    variant = "globalnet"

    def __init__(self, hidden_size: int = 32, patch_size: int = 4) -> None:
        super().__init__(hidden_size, patch_size)
        self.sequence_embed = nn.Conv2d(
            1, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.recurrent = ConvGRUCell(hidden_size)
        self.context_projection = nn.Conv2d(hidden_size, hidden_size, 1)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def forward(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        base = self.stem.encode(history, trend)
        sequence = torch.cat((history, trend), dim=1)
        batch, frames, _, height, width = sequence.shape
        embedded = self.sequence_embed(
            sequence.reshape(batch * frames, 1, height, width)
        ).unflatten(0, (batch, frames))
        state = torch.zeros_like(embedded[:, 0])
        future_states: list[torch.Tensor] = []
        history_frames = history.shape[1]
        for index in range(frames):
            state = self.recurrent(embedded[:, index], state)
            if index >= history_frames:
                future_states.append(state)
        context = torch.stack(future_states, dim=1)
        projected = self.context_projection(context.flatten(0, 1)).unflatten(
            0, (batch, trend.shape[1])
        )
        return self.stem.decode(base + projected)


class ToraAdaLNResidualProbe(MotionProbeBase):
    """Tora-style dense motion patches fused through zero-initialized AdaLN."""

    variant = "tora_adaln"

    def __init__(self, hidden_size: int = 32, patch_size: int = 4) -> None:
        super().__init__(hidden_size, patch_size)
        groups = 8 if hidden_size % 8 == 0 else 1
        self.motion_patch_embed = nn.Conv2d(
            5, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.temporal_mix = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=hidden_size,
        )
        self.motion_body = nn.Sequential(
            nn.GroupNorm(groups, hidden_size),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        )
        self.normalization = nn.GroupNorm(groups, hidden_size, affine=False)
        self.to_scale_shift = nn.Conv2d(hidden_size, hidden_size * 2, 1)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def motion_features(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        previous = _future_previous(history, trend)
        delta = trend - previous
        previous_delta = torch.cat((history[:, -1:] - history[:, -2:-1], delta[:, :-1]), dim=1)
        acceleration = delta - previous_delta
        gx, gy = _spatial_gradients(trend)
        motion = torch.cat((delta, delta.abs(), acceleration, gx, gy), dim=2)
        batch, leads, channels, height, width = motion.shape
        patches = self.motion_patch_embed(
            motion.reshape(batch * leads, channels, height, width)
        ).unflatten(0, (batch, leads))
        mixed = self.temporal_mix(patches.permute(0, 2, 1, 3, 4)).permute(
            0, 2, 1, 3, 4
        )
        flat = (patches + mixed).flatten(0, 1)
        return (flat + self.motion_body(flat)).unflatten(0, (batch, leads))

    def forward(self, history: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        base = self.stem.encode(history, trend)
        motion = self.motion_features(history, trend)
        batch, leads = base.shape[:2]
        scale, shift = self.to_scale_shift(motion.flatten(0, 1)).chunk(2, dim=1)
        normalized = self.normalization(base.flatten(0, 1)).unflatten(
            0, (batch, leads)
        )
        fused = base + scale.unflatten(0, (batch, leads)) * normalized + shift.unflatten(
            0, (batch, leads)
        )
        return self.stem.decode(fused)


def build_motion_probe(
    variant: str, *, hidden_size: int = 32, patch_size: int = 4
) -> MotionProbeBase:
    normalized = variant.strip().lower()
    if normalized == "baseline":
        return MotionProbeBase(hidden_size, patch_size)
    if normalized == "globalnet":
        return GlobalNetResidualProbe(hidden_size, patch_size)
    if normalized in {"tora", "tora_adaln"}:
        return ToraAdaLNResidualProbe(hidden_size, patch_size)
    raise ValueError(f"unknown motion probe variant: {variant}")
