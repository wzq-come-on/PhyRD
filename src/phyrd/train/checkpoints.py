from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


class CheckpointManager:
    """Maintain a run's checkpoints under ``checkpoints/``.

    The nested layout is the v11 contract. Existing artifact directories are
    intentionally never migrated or overwritten by this class.
    """

    def __init__(self, directory: str | Path, *, allow_existing: bool = False) -> None:
        self.directory = Path(directory)
        self.checkpoint_directory = self.directory / "checkpoints"
        self.metrics_directory = self.directory / "metrics"
        self.predictions_directory = self.directory / "predictions"
        existing = (
            list(self.checkpoint_directory.glob("checkpoint_*.pt"))
            + list(self.directory.glob("checkpoint_*.pt"))
            + [
                path
                for path in (
                    self.directory / "run_summary.json",
                    self.directory / "config_snapshot.yaml",
                )
                if path.exists()
            ]
            if self.directory.exists()
            else []
        )
        if existing and not allow_existing:
            raise FileExistsError(
                f"artifact directory already contains run outputs: {self.directory}; "
                "choose a new directory or set artifacts.allow_existing=true explicitly"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)
        self.predictions_directory.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        """Write the resolved run configuration once for reproducibility."""
        destination = self.directory / "config_snapshot.yaml"
        if destination.exists():
            raise FileExistsError(
                f"config snapshot already exists: {destination}; choose a new run directory"
            )
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        return destination

    def _atomic_save(self, payload: dict[str, Any], filename: str) -> None:
        destination = self.checkpoint_directory / filename
        temporary = self.checkpoint_directory / f".{filename}.tmp"
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
