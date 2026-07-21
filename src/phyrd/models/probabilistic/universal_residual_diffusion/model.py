"""Backbone-agnostic residual diffusion in a fixed ``[-1, 1]`` residual space."""
from __future__ import annotations

from ..residual_diffusion.model import ResidualDiffusionModel


class UniversalResidualDiffusionModel(ResidualDiffusionModel):
    """Condition only on normalized history and deterministic trend.

    A fixed residual coordinate avoids storing residual statistics from any one
    backbone.  Universal behavior is obtained by training with a backbone pool,
    not by zero-shot checkpoint swapping.
    """

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.pop("residual_stats_path", None) is not None:
            raise ValueError(
                "universal_residual_diffusion forbids residual_stats_path; "
                "use the fixed target - trend residual coordinate"
            )
        kwargs.setdefault("residual_center", 0.0)
        kwargs.setdefault("residual_scale", 1.0)
        super().__init__(*args, **kwargs)
