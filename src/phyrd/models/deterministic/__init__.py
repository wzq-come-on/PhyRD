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
from .pool import BackbonePool, build_backbone_pool

register_backbone(
    "sdir_official",
    "phyrd.models.deterministic.sdir_official:OfficialSDIRForecast",
)
register_backbone(
    "phydnet_external",
    "phyrd.models.deterministic.phydnet_external:ExternalPhyDNetForecast",
)

__all__ = [
    "DeterministicLossOutput",
    "BackbonePool",
    "available_backbones",
    "build_backbone",
    "build_backbone_pool",
    "checkpoint_backbone_spec",
    "register_backbone",
]
