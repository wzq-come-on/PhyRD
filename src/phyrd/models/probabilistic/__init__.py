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
register_probabilistic(
    "universal_residual_diffusion",
    "phyrd.models.probabilistic.universal_residual_diffusion:UniversalResidualDiffusionModel",
)

__all__ = [
    "ProbabilisticModel",
    "available_probabilistic_models",
    "build_probabilistic",
    "register_probabilistic",
]
