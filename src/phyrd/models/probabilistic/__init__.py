"""Config-selectable probabilistic forecast models."""

from .base import ProbabilisticModel
from .registry import (
    available_probabilistic_models,
    build_probabilistic,
    register_probabilistic,
)

register_probabilistic(
    "residual_diffusion",
    "phyrd.models.probabilistic.residual_diffusion:ResidualDiffusionModel",
)

__all__ = [
    "ProbabilisticModel",
    "available_probabilistic_models",
    "build_probabilistic",
    "register_probabilistic",
]
