from __future__ import annotations

from pathlib import Path

import pytest

import torch
import h5py
import numpy as np

from phyrd.data.sevir import (
    DiffCastH5Dataset,
    SPLIT_BOUNDS,
    preprocess_spatial,
    resolve_sevir_paths,
)


def test_resolve_nested_opensciencelab_layout(tmp_path: Path) -> None:
    (tmp_path / "CATALOG.csv").touch()
    data = tmp_path / "OpenScienceLab___SEVIR" / "raw" / "SEVIR" / "data"
    data.mkdir(parents=True)
    paths = resolve_sevir_paths(tmp_path)
    assert paths.catalog_path == (tmp_path / "CATALOG.csv").resolve()
    assert paths.data_root == data.resolve()


def test_four_splits_are_contiguous() -> None:
    assert SPLIT_BOUNDS["train"][1] == SPLIT_BOUNDS["val_model"][0]
    assert SPLIT_BOUNDS["val_model"][1] == SPLIT_BOUNDS["val_calib"][0]
    assert SPLIT_BOUNDS["val_calib"][1] == SPLIT_BOUNDS["report_test"][0]


def test_missing_layout_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_sevir_paths(tmp_path)


def test_diffcast_spatial_resize_keeps_time_and_value_range() -> None:
    values = torch.linspace(0, 1, 25 * 384 * 384).reshape(25, 1, 384, 384)
    resized = preprocess_spatial(values, 128, "diffcast_bilinear")
    assert resized.shape == (25, 1, 128, 128)
    assert 0 <= resized.min() <= resized.max() <= 1


def test_area_downsample_preserves_constant_field() -> None:
    values = torch.full((25, 1, 384, 384), 0.37)
    resized = preprocess_spatial(values, 128, "area")
    assert resized.shape == (25, 1, 128, 128)
    assert torch.allclose(resized, torch.full_like(resized, 0.37))


def test_diffcast_h5_supports_exact_5_to_20_protocol(tmp_path: Path) -> None:
    path = tmp_path / "diffcast.h5"
    sequence = np.arange(25, dtype=np.uint8)[:, None, None]
    sequence = np.broadcast_to(sequence, (25, 384, 384))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=sequence[None])
    dataset = DiffCastH5Dataset(path, "train", model_resolution=128)
    sample = dataset[0]
    assert sample["x"].shape == (5, 1, 128, 128)
    assert sample["y"].shape == (20, 1, 128, 128)
    assert torch.allclose(sample["x"][:, 0, 0, 0] * 255.0, torch.arange(5).float())
    assert torch.allclose(sample["y"][:, 0, 0, 0] * 255.0, torch.arange(5, 25).float())
    dataset.close()
