"""Frozen deterministic-backbone pools for backbone-agnostic probability training."""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn

from .registry import build_backbone


class BackbonePool(nn.Module):
    """Select one frozen deterministic forecast backbone per probability batch."""

    def __init__(
        self,
        backbones: Mapping[str, nn.Module],
        member_specs: Sequence[Mapping[str, object]],
        *,
        selection: str = "uniform",
    ) -> None:
        super().__init__()
        if not backbones:
            raise ValueError("deterministic_pool.members cannot be empty")
        if selection not in {"uniform", "weighted_random", "round_robin"}:
            raise ValueError("selection must be uniform, weighted_random, or round_robin")
        self.backbones = nn.ModuleDict(dict(backbones))
        self.member_specs = [dict(item) for item in member_specs]
        self.selection = selection
        self._names = tuple(self.backbones.keys())
        self._active_name = self._names[0]
        for backbone in self.backbones.values():
            backbone.requires_grad_(False)
            backbone.eval()

    @property
    def active_name(self) -> str:
        return self._active_name

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def select(self, name: str) -> str:
        if name not in self.backbones:
            raise ValueError(f"unknown backbone {name!r}; choices: {', '.join(self._names)}")
        self._active_name = name
        return name

    def select_for_step(self, step: int, seed: int) -> str:
        if self.selection == "round_robin":
            return self.select(self._names[step % len(self._names)])
        generator = random.Random((int(seed) << 32) + int(step))
        if self.selection == "weighted_random":
            weights = [float(spec.get("weight", 1.0)) for spec in self.member_specs]
            if any(weight <= 0 for weight in weights):
                raise ValueError("deterministic_pool member weights must be positive")
            return self.select(generator.choices(self._names, weights=weights, k=1)[0])
        return self.select(generator.choice(self._names))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.backbones[self._active_name](history)


def _load_checkpoint(backbone: nn.Module, checkpoint: str | Path) -> None:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"deterministic pool checkpoint not found: {path}")
    external_loader = getattr(backbone, "load_external_checkpoint", None)
    if callable(external_loader):
        external_loader(path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("deterministic", payload.get("model", payload))
    backbone.load_state_dict(state, strict=True)


def build_backbone_pool(
    members: Sequence[Mapping[str, object]],
    *,
    input_frames: int,
    output_frames: int,
    selection: str = "uniform",
) -> BackbonePool:
    backbones: dict[str, nn.Module] = {}
    normalized_specs: list[dict[str, object]] = []
    for raw_spec in members:
        spec = dict(raw_spec)
        name = str(spec.get("name", "")).strip().lower()
        params = dict(spec.get("params", {}))
        checkpoint = spec.get("checkpoint")
        if not name or not isinstance(spec.get("params", {}), Mapping):
            raise ValueError("each deterministic_pool member needs name and params")
        if name in backbones:
            raise ValueError(f"duplicate deterministic_pool member {name!r}")
        if not checkpoint:
            raise ValueError(f"deterministic_pool member {name!r} requires a checkpoint")
        backbone = build_backbone(
            name, input_frames=input_frames, output_frames=output_frames, params=params
        )
        _load_checkpoint(backbone, str(checkpoint))
        backbones[name] = backbone
        normalized_specs.append(
            {"name": name, "params": params, "checkpoint": str(checkpoint), "weight": spec.get("weight", 1.0)}
        )
    return BackbonePool(backbones, normalized_specs, selection=selection)
