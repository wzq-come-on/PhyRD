from pathlib import Path

import pytest
import torch

from phyrd.train import CheckpointManager


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_checkpoint_manager_keeps_only_best_and_last(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save({"step": 10}, val_loss=0.5)
    manager.save({"step": 20}, val_loss=0.7)

    checkpoint_dir = tmp_path / "checkpoints"
    assert load(checkpoint_dir / "checkpoint_best.pt")["step"] == 10
    assert load(checkpoint_dir / "checkpoint_last.pt")["step"] == 20

    manager.save({"step": 30}, val_loss=0.4)
    assert load(checkpoint_dir / "checkpoint_best.pt")["step"] == 30
    assert load(checkpoint_dir / "checkpoint_last.pt")["step"] == 30
    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == [
        "checkpoint_best.pt",
        "checkpoint_last.pt",
    ]


def test_checkpoint_manager_rejects_an_existing_run(tmp_path: Path) -> None:
    CheckpointManager(tmp_path).save({"step": 1}, val_loss=1.0)
    with pytest.raises(FileExistsError):
        CheckpointManager(tmp_path)
