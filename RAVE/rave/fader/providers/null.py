"""Placeholder discrete provider when labels come from sidecar."""

from __future__ import annotations

from typing import Sequence

import gin
import numpy as np

from .base import DiscreteAttributeProvider


@gin.configurable
class NullDiscreteProvider(DiscreteAttributeProvider):
    """Zero-filled discrete slice; real values come from sidecar when present."""

    def __init__(self, discrete_attributes: Sequence[str]) -> None:
        self.discrete_attributes = list(discrete_attributes)

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        n = len(self.discrete_attributes)
        if n == 0:
            return np.zeros((0, latent_length), dtype=np.float32)
        return np.zeros((n, latent_length), dtype=np.float32)
