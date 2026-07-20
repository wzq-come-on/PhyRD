from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without implicit mutation of caller-owned values."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"configuration must be a mapping, got {type(value).__name__}")
    config = deepcopy(dict(value))
    config["_config_path"] = str(config_path)
    return config


def require_keys(mapping: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"missing required configuration keys: {missing}")

