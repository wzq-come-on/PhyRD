from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from torch import nn


_PROBABILISTIC_MODELS: dict[str, type[nn.Module] | str] = {}


def register_probabilistic(name: str, model: type[nn.Module] | str) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("probabilistic model name cannot be empty")
    if normalized in _PROBABILISTIC_MODELS:
        raise ValueError(f"probabilistic model {normalized!r} is already registered")
    _PROBABILISTIC_MODELS[normalized] = model


def available_probabilistic_models() -> tuple[str, ...]:
    return tuple(sorted(_PROBABILISTIC_MODELS))


def build_probabilistic(
    name: str,
    *,
    input_frames: int,
    output_frames: int,
    params: Mapping[str, object] | None = None,
) -> nn.Module:
    normalized = name.strip().lower()
    try:
        model = _PROBABILISTIC_MODELS[normalized]
    except KeyError as error:
        choices = ", ".join(available_probabilistic_models()) or "<none>"
        raise ValueError(
            f"unknown probabilistic model {name!r}; available models: {choices}"
        ) from error
    if isinstance(model, str):
        module_name, separator, class_name = model.partition(":")
        if not separator:
            raise ValueError(f"invalid probabilistic model path {model!r}")
        resolved = getattr(import_module(module_name), class_name)
        if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
            raise TypeError(f"registered probabilistic model {model!r} is not an nn.Module")
        _PROBABILISTIC_MODELS[normalized] = resolved
        model = resolved
    return model(input_frames, output_frames, **dict(params or {}))
