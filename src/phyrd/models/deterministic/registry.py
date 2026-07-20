from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from torch import nn


_BACKBONES: dict[str, type[nn.Module] | str] = {}


def register_backbone(name: str, backbone: type[nn.Module] | str) -> None:
    """Register a class or lazy ``module:class`` path under a config name."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("deterministic backbone name cannot be empty")
    if normalized in _BACKBONES:
        raise ValueError(f"deterministic backbone {normalized!r} is already registered")
    _BACKBONES[normalized] = backbone


def available_backbones() -> tuple[str, ...]:
    return tuple(sorted(_BACKBONES))


def checkpoint_backbone_spec(protocol: Mapping[str, object]) -> dict[str, object] | None:
    """Normalize current and official-migration checkpoint metadata.

    The one accepted legacy schema is the already-trained official SDIR run.
    Ambiguous ``sdir`` metadata from the retired native implementation is
    intentionally rejected.
    """
    current = protocol.get("deterministic")
    if isinstance(current, Mapping):
        return {
            "name": str(current.get("name", "")),
            "params": dict(current.get("params", {})),
        }
    if protocol.get("deterministic_backbone") == "sdir_official":
        return {
            "name": "sdir_official",
            "params": dict(protocol.get("deterministic_config", {})),
        }
    return None


def build_backbone(
    name: str,
    *,
    input_frames: int,
    output_frames: int,
    params: Mapping[str, object] | None = None,
) -> nn.Module:
    normalized = name.strip().lower()
    try:
        backbone = _BACKBONES[normalized]
    except KeyError as error:
        choices = ", ".join(available_backbones()) or "<none>"
        raise ValueError(
            f"unknown deterministic backbone {name!r}; available backbones: {choices}"
        ) from error
    if isinstance(backbone, str):
        module_name, separator, class_name = backbone.partition(":")
        if not separator:
            raise ValueError(f"invalid lazy backbone path {backbone!r}")
        resolved = getattr(import_module(module_name), class_name)
        if not isinstance(resolved, type) or not issubclass(resolved, nn.Module):
            raise TypeError(f"registered backbone {backbone!r} is not an nn.Module class")
        _BACKBONES[normalized] = resolved
        backbone = resolved
    return backbone(input_frames, output_frames, **dict(params or {}))
