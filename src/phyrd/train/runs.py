from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_experiment_directory(
    root: str | Path,
    deterministic_name: str,
    probabilistic_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Return a non-overwriting ``deterministic_probabilistic/timestamp`` path."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(stamp) != 15 or stamp[8] != "_":
        raise ValueError("timestamp must use YYYYMMDD_HHMMSS")
    return Path(root) / f"{deterministic_name}_{probabilistic_name}" / stamp
