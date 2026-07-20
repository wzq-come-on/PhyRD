"""Deterministic forecast backbones and their config-facing registry.

Add future backbones as sibling modules, register them with ``register_backbone``,
and import the module here so it becomes selectable through ``model.deterministic.name``.
"""

from .base import DeterministicLossOutput
from .registry import (
    available_backbones,
    build_backbone,
    checkpoint_backbone_spec,
    register_backbone,
)

register_backbone(
    "sdir_official",
    "phyrd.models.deterministic.sdir_official:OfficialSDIRForecast",
)

__all__ = [
    "DeterministicLossOutput",
    "available_backbones",
    "build_backbone",
    "checkpoint_backbone_spec",
    "register_backbone",
]
