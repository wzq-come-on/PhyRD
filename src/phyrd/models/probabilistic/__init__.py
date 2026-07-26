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
register_probabilistic(
    "udip",
    "phyrd.models.probabilistic.udip:UDIPModel",
)
register_probabilistic(
    "trajres_diffusion",
    "phyrd.models.probabilistic.trajres_diffusion:TrajectoryResidualDiffusionModel",
)
register_probabilistic(
    "rescasformer",
    "phyrd.models.probabilistic.rescasformer:ResidualCasFormerModel",
)
register_probabilistic(
    "temporal_residual_dit",
    "phyrd.models.probabilistic.temporal_residual_dit:TemporalResidualDiffusionModel",
)

__all__ = [
    "ProbabilisticModel",
    "available_probabilistic_models",
    "build_probabilistic",
    "register_probabilistic",
]
