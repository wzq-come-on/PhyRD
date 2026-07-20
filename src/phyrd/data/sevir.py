from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


SPLIT_BOUNDS: dict[str, tuple[str | None, str | None]] = {
    "train": (None, "2019-01-01"),
    "val_model": ("2019-01-01", "2019-06-01"),
    "val_calib": ("2019-06-01", "2019-10-01"),
    "report_test": ("2019-10-01", None),
}

SPATIAL_PREPROCESS_MODES = {"none", "diffcast_bilinear", "area"}


@dataclass(frozen=True)
class SEVIRPaths:
    requested_root: Path
    catalog_path: Path
    data_root: Path


def _first_existing(candidates: list[Path], kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not resolve SEVIR {kind}; checked:\n  - {rendered}")


def resolve_sevir_paths(
    root: str | Path,
    catalog_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> SEVIRPaths:
    """Resolve known public SEVIR layouts without recursively scanning a multi-TB tree."""
    requested = Path(root).expanduser().resolve()
    catalog = (
        Path(catalog_path).expanduser().resolve()
        if catalog_path is not None
        else _first_existing(
            [
                requested / "CATALOG.csv",
                requested / "SEVIR" / "CATALOG.csv",
                requested / "OpenScienceLab___SEVIR" / "raw" / "SEVIR" / "CATALOG.csv",
            ],
            "catalog",
        )
    )
    hdf_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else _first_existing(
            [
                requested / "data",
                requested / "SEVIR" / "data",
                requested / "OpenScienceLab___SEVIR" / "raw" / "SEVIR" / "data",
            ],
            "HDF5 data root",
        )
    )
    if not catalog.is_file():
        raise FileNotFoundError(f"SEVIR catalog is not a file: {catalog}")
    if not hdf_root.is_dir():
        raise NotADirectoryError(f"SEVIR data root is not a directory: {hdf_root}")
    return SEVIRPaths(requested, catalog, hdf_root)


def preprocess_spatial(
    values: torch.Tensor, model_resolution: int, mode: str
) -> torch.Tensor:
    """Map native 384² frames to a registered model grid without changing time.

    `diffcast_bilinear` reproduces the effective SEVIR wrapper choice in DiffCast
    (`transforms.Resize`). `area` is the conservative alternative corresponding
    to the generic average-pooling helper in its SEVIR loader.
    """
    if values.ndim != 4 or values.shape[1:] != (1, 384, 384):
        raise ValueError(f"expected [T,1,384,384], got {tuple(values.shape)}")
    if mode not in SPATIAL_PREPROCESS_MODES:
        raise ValueError(f"unknown spatial preprocess {mode!r}")
    if model_resolution <= 0 or 384 % model_resolution:
        raise ValueError("model_resolution must be a positive divisor of 384")
    if model_resolution == 384:
        if mode != "none":
            raise ValueError("native 384 resolution must use spatial_preprocess='none'")
        return values
    if mode == "none":
        raise ValueError("downsampled model_resolution requires an explicit preprocess mode")
    if mode == "diffcast_bilinear":
        return F.interpolate(
            values,
            size=(model_resolution, model_resolution),
            mode="bilinear",
            align_corners=False,
            # DiffCast is locked to torchvision 0.13.1, where tensor Resize
            # defaults to the legacy non-antialiased bilinear path.
            antialias=False,
        )
    return F.interpolate(values, size=(model_resolution, model_resolution), mode="area")


class SEVIRDataset(Dataset[dict[str, Any]]):
    """Minimal event-safe VIL reader for the fixed 13→12 center window."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        catalog_path: str | Path | None = None,
        data_root: str | Path | None = None,
        input_frames: int = 13,
        output_frames: int = 12,
        window_start_index: int = 12,
        model_resolution: int = 384,
        spatial_preprocess: str = "none",
        max_samples: int | None = None,
    ) -> None:
        self._h5_files: dict[str, h5py.File] = {}
        if split not in SPLIT_BOUNDS:
            raise ValueError(f"unknown split {split!r}; choose one of {sorted(SPLIT_BOUNDS)}")
        if input_frames != 13 or output_frames != 12 or window_start_index != 12:
            raise ValueError("the frozen main protocol requires input=13, output=12, start=12")
        self.paths = resolve_sevir_paths(root, catalog_path, data_root)
        self.split = split
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.window_start_index = window_start_index
        self.native_resolution = 384
        self.model_resolution = int(model_resolution)
        self.spatial_preprocess = spatial_preprocess
        # Validate the spatial contract before reading any HDF5 event.
        preprocess_spatial(
            torch.zeros(1, 1, 384, 384), self.model_resolution, self.spatial_preprocess
        )

        catalog = pd.read_csv(
            self.paths.catalog_path,
            usecols=lambda column: column
            in {"id", "event_id", "file_name", "file_index", "img_type", "time_utc", "pct_missing"},
            parse_dates=["time_utc"],
            low_memory=False,
        )
        rows = catalog[catalog["img_type"].eq("vil")].copy()
        if "pct_missing" in rows:
            rows = rows[rows["pct_missing"].fillna(1.0).eq(0.0)]
        lower, upper = SPLIT_BOUNDS[split]
        if lower is not None:
            rows = rows[rows["time_utc"] >= pd.Timestamp(lower)]
        if upper is not None:
            rows = rows[rows["time_utc"] < pd.Timestamp(upper)]
        rows = rows.sort_values(["time_utc", "id"], kind="stable").reset_index(drop=True)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            rows = rows.iloc[:max_samples].copy()
        if rows.empty:
            raise RuntimeError(f"no valid VIL events found for split={split} in {self.paths.catalog_path}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _file(self, relative_name: str) -> h5py.File:
        file_path = (self.paths.data_root / relative_name).resolve()
        try:
            file_path.relative_to(self.paths.data_root)
        except ValueError as exc:
            raise ValueError(f"catalog path escapes data root: {relative_name}") from exc
        key = str(file_path)
        if key not in self._h5_files:
            if not file_path.is_file():
                raise FileNotFoundError(f"catalog-referenced HDF5 file is missing: {file_path}")
            self._h5_files[key] = h5py.File(file_path, "r")
        return self._h5_files[key]

    @staticmethod
    def _to_time_first(array: np.ndarray) -> np.ndarray:
        if array.ndim != 3:
            raise ValueError(f"expected a 3D VIL event, got shape {array.shape}")
        time_axes = [axis for axis, size in enumerate(array.shape) if size == 49]
        if len(time_axes) != 1:
            raise ValueError(f"expected exactly one 49-frame axis, got shape {array.shape}")
        event = np.moveaxis(array, time_axes[0], 0)
        if event.shape[1:] != (384, 384):
            raise ValueError(f"main protocol requires 384x384 VIL, got {event.shape}")
        return np.ascontiguousarray(event)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        handle = self._file(str(row["file_name"]))
        if "vil" not in handle:
            raise KeyError(f"HDF5 file {handle.filename} does not contain dataset 'vil'")
        raw = np.asarray(handle["vil"][int(row["file_index"])])
        event = self._to_time_first(raw)
        stop = self.window_start_index + self.input_frames + self.output_frames
        window = event[self.window_start_index : stop]
        if window.shape != (25, 384, 384):
            raise ValueError(f"bad frozen window shape: {window.shape}")
        values = torch.from_numpy(window.astype(np.float32) / 255.0).unsqueeze(1)
        values = preprocess_spatial(values, self.model_resolution, self.spatial_preprocess)
        event_id = row.get("event_id", None)
        if pd.isna(event_id) or event_id in (None, ""):
            event_id = row["id"]
        return {
            "x": values[: self.input_frames],
            "y": values[self.input_frames :],
            "sample_id": str(row["id"]),
            "event_id": str(event_id),
            "time_utc": row["time_utc"].isoformat(),
            "file_name": str(row["file_name"]),
            "file_index": int(row["file_index"]),
            "native_resolution": self.native_resolution,
            "model_resolution": self.model_resolution,
            "spatial_preprocess": self.spatial_preprocess,
        }

    def close(self) -> None:
        for handle in getattr(self, "_h5_files", {}).values():
            handle.close()
        getattr(self, "_h5_files", {}).clear()

    def __del__(self) -> None:
        self.close()


class DiffCastH5Dataset(Dataset[dict[str, Any]]):
    """Read frozen DiffCast HDF5 with registered model/calibration partitions."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        *,
        input_frames: int = 5,
        output_frames: int = 20,
        window_start_index: int = 0,
        model_resolution: int = 128,
        spatial_preprocess: str = "diffcast_bilinear",
        max_samples: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"DiffCast HDF5 is not a file: {self.path}")
        split_source = {
            "val_model": ("valid", 0, 2),
            "val_calib": ("valid", 1, 2),
            "report_test": ("test", 0, 1),
        }
        if split in split_source:
            source_split, index_offset, index_stride = split_source[split]
        elif split in {"train", "valid", "test"}:
            source_split, index_offset, index_stride = split, 0, 1
        else:
            raise ValueError(
                "DiffCast split must be 'train', 'valid', 'test', 'val_model', "
                "'val_calib', or 'report_test'"
            )
        if input_frames <= 0 or output_frames <= 0 or window_start_index < 0:
            raise ValueError("frame counts must be positive and window_start_index non-negative")
        self.split = split
        self.source_split = source_split
        self.index_offset = index_offset
        self.index_stride = index_stride
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.window_start_index = int(window_start_index)
        self.model_resolution = int(model_resolution)
        self.spatial_preprocess = spatial_preprocess
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            if source_split not in handle:
                raise KeyError(
                    f"split {source_split!r} not found in {self.path}; keys={list(handle.keys())}"
                )
            shape = tuple(handle[source_split].shape)
        if len(shape) != 4:
            raise ValueError(f"expected [N,T,H,W] DiffCast data, got {shape}")
        _, sequence_frames, height, width = shape
        if height != width:
            raise ValueError(f"DiffCast frames must be square, got {shape}")
        stop = self.window_start_index + self.input_frames + self.output_frames
        if stop > sequence_frames:
            raise ValueError(f"requested frame stop {stop} exceeds HDF5 sequence length {sequence_frames}")
        self.native_resolution = int(height)
        if self.native_resolution != 384:
            raise ValueError(f"registered DiffCast protocol requires 384x384 input, got {shape}")
        preprocess_spatial(
            torch.zeros(1, 1, 384, 384), self.model_resolution, self.spatial_preprocess
        )
        source_length = int(shape[0])
        length = max(0, (source_length - index_offset + index_stride - 1) // index_stride)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            length = min(length, int(max_samples))
        self.length = length

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < self.length:
            raise IndexError(index)
        source_index = self.index_offset + int(index) * self.index_stride
        raw = np.asarray(self.h5[self.source_split][source_index])
        start = self.window_start_index
        stop = start + self.input_frames + self.output_frames
        window = raw[start:stop]
        expected = (self.input_frames + self.output_frames, 384, 384)
        if window.shape != expected:
            raise ValueError(f"bad DiffCast window shape {window.shape}; expected {expected}")
        values = torch.from_numpy(window.astype(np.float32) / 255.0).unsqueeze(1)
        values = preprocess_spatial(values, self.model_resolution, self.spatial_preprocess)
        return {
            "x": values[: self.input_frames],
            "y": values[self.input_frames :],
            "sample_id": f"{self.source_split}:{source_index}",
            "event_id": f"{self.source_split}:{source_index}",
            "native_resolution": self.native_resolution,
            "model_resolution": self.model_resolution,
            "spatial_preprocess": self.spatial_preprocess,
        }
