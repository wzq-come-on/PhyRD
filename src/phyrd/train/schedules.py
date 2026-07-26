from __future__ import annotations

import math
from typing import Any


def learning_rate_multiplier(
    optimization: dict[str, Any],
    *,
    step: int,
    max_steps: int,
    steps_per_epoch: int,
) -> float:
    """Return a deterministic constant or warmup-cosine LR multiplier."""

    schedule = str(optimization.get("scheduler", "constant")).lower()
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError("optimization.scheduler must be 'constant' or 'cosine'")
    base_lr = float(optimization["learning_rate"])
    if base_lr <= 0:
        raise ValueError("optimization.learning_rate must be positive")
    warmup_steps = optimization.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = round(
            float(optimization.get("warmup_epochs", 0.0)) * steps_per_epoch
        )
    warmup_steps = max(0, int(warmup_steps))
    warmup_ratio = float(optimization.get("warmup_lr", base_lr)) / base_lr
    minimum_ratio = float(optimization.get("min_lr", 0.0)) / base_lr
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("optimization.warmup_lr must be within [0, learning_rate]")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("optimization.min_lr must be within [0, learning_rate]")
    if warmup_steps and step < warmup_steps:
        fraction = step / warmup_steps
        return warmup_ratio + (1.0 - warmup_ratio) * fraction
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine

