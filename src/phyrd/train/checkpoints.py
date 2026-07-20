from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Maintain exactly the newest and best-validation checkpoints."""

    def __init__(self, directory: str | Path, *, allow_existing: bool = False) -> None:
        self.directory = Path(directory)
        existing = (
            list(self.directory.glob("checkpoint_*.pt"))
            + [path for path in (self.directory / "run_summary.json",) if path.exists()]
            if self.directory.exists()
            else []
        )
        if existing and not allow_existing:
            raise FileExistsError(
                f"artifact directory already contains run outputs: {self.directory}; "
                "choose a new directory or set artifacts.allow_existing=true explicitly"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def _atomic_save(self, payload: dict[str, Any], filename: str) -> None:
        destination = self.directory / filename
        temporary = self.directory / f".{filename}.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination)

    def save(self, payload: dict[str, Any], *, val_loss: float | None) -> bool:
        """Update last, and update best only when validation loss improves."""
        self._atomic_save(payload, "checkpoint_last.pt")
        improved = val_loss is not None and val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self._atomic_save(payload, "checkpoint_best.pt")
        return improved
