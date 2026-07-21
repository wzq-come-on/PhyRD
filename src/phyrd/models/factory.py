from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .composer import ForecastComposer
from .deterministic import build_backbone
from .probabilistic import build_probabilistic


def build_composite_from_config(
    config: Mapping[str, object],
    *,
    input_frames: int,
    output_frames: int,
) -> ForecastComposer:
    """Build a deterministic/probabilistic pair from a resolved YAML config.

    The legacy ``model.diffusion`` fields remain accepted so old configs can
    migrate without changing checkpoint loading in one step.
    """
    model_config = dict(config.get("model", {}))
    deterministic_config = dict(model_config.get("deterministic", {}))
    deterministic_name = str(deterministic_config.get("name", "sdir_official"))
    deterministic_params = dict(deterministic_config.get("params", {}))
    deterministic = build_backbone(
        deterministic_name,
        input_frames=input_frames,
        output_frames=output_frames,
        params=deterministic_params,
    )

    probabilistic_config = dict(model_config.get("probabilistic", {}))
    probabilistic_name = str(
        probabilistic_config.get("name", "residual_diffusion")
    )
    probabilistic_params = dict(probabilistic_config.get("params", {}))
    legacy_diffusion = dict(model_config.get("diffusion", {}))
    for key, value in legacy_diffusion.items():
        probabilistic_params.setdefault(key, value)
    probabilistic_params.setdefault("base_channels", model_config.get("base_channels", 32))
    probabilistic_params.setdefault(
        "diffusion_steps", model_config.get("diffusion_steps", 100)
    )
    if "extensions" in probabilistic_config:
        probabilistic_params.setdefault("extensions", probabilistic_config["extensions"])
    probabilistic = build_probabilistic(
        probabilistic_name,
        input_frames=input_frames,
        output_frames=output_frames,
        params=probabilistic_params,
    )
    return ForecastComposer(
        deterministic,
        probabilistic,
        freeze_deterministic=bool(model_config.get("freeze_deterministic", True)),
        deterministic_name=deterministic_name,
        deterministic_params=deterministic_params,
    )
