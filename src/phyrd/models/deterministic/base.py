from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DeterministicLossOutput:
    """Common training result returned by every deterministic backbone."""

    loss: torch.Tensor
    prediction: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
