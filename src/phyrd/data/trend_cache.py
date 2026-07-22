"""Disk-backed deterministic trend caches for frozen-backbone training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedTrendDataset(Dataset[dict[str, Any]]):
    """Attach a read-only ``[T,1,H,W]`` deterministic forecast to each sample."""

    def __init__(self, base: Dataset[dict[str, Any]], cache_path: str | Path) -> None:
        self.base = base
        self.cache_path = Path(cache_path).expanduser().resolve()
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"trend cache is not a file: {self.cache_path}")
        self._cache: np.ndarray | None = None
        cache_shape = tuple(np.load(self.cache_path, mmap_mode="r").shape)
        if len(cache_shape) != 5 or cache_shape[2] != 1:
            raise ValueError(f"expected trend cache [N,T,1,H,W], got {cache_shape}")
        if cache_shape[0] != len(base):
            raise ValueError(
                f"trend cache length {cache_shape[0]} does not match dataset length {len(base)}"
            )
        self.cache_shape = cache_shape
        self.cache_metadata: dict[str, Any] = {}
        metadata_path = self.cache_path.with_suffix(".json")
        if metadata_path.is_file():
            try:
                self.cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid trend cache metadata: {metadata_path}") from exc

    @property
    def cache(self) -> np.ndarray:
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        sample["trend"] = torch.from_numpy(np.asarray(self.cache[index], dtype=np.float32))
        sample["trend_cache"] = str(self.cache_path)
        return sample

    def __getattr__(self, name: str) -> Any:
        if name in {"base", "cache_path", "_cache", "cache_shape", "cache_metadata"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()
        self._cache = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

