"""Precomputed learned feature provider (stub layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gin
import numpy as np

from .base import ContinuousAttributeProvider


@gin.configurable
class LearnedFeatureProvider(ContinuousAttributeProvider):
    """
    Loads per-index features from {db_path}/learned_features/{index:08d}.npy.

    Expected shape (D_cont, T_lat) or (T_lat, D_cont) — auto-transposed if needed.
    Missing files → zero rows.
    """

    def __init__(
        self,
        db_path: str,
        continuous_attributes: Sequence[str],
        feature_subdir: str = "learned_features",
    ) -> None:
        self.continuous_attributes = list(continuous_attributes)
        self._root = Path(db_path) / feature_subdir
        self._d = len(self.continuous_attributes)

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        path = self._root / f"{index:08d}.npy"
        if not path.is_file():
            return np.zeros((self._d, latent_length), dtype=np.float32)
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = np.tile(arr[:, None], (1, latent_length))
        elif arr.shape[0] == latent_length and arr.shape[1] == self._d:
            arr = arr.T
        if arr.shape[0] != self._d:
            return np.zeros((self._d, latent_length), dtype=np.float32)
        t = arr.shape[1]
        if t != latent_length:
            if t >= latent_length:
                arr = arr[:, :latent_length]
            else:
                arr = np.pad(arr, ((0, 0), (0, latent_length - t)))
        return arr
