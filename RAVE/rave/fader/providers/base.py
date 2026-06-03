"""ABC interfaces for Fader attribute providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContinuousAttributeProvider(ABC):
    """
    Source for continuous attribute rows.

    Returns raw floats (not normalized) — FaderRAVE._prepare_attributes handles
    min/max normalize (decoder) and quantile bucketize (latent CE).
    """

    @abstractmethod
    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        """Shape (len(continuous_names), T_lat)."""


class DiscreteAttributeProvider(ABC):
    """
    Source for discrete attribute rows.

    Returns integer class indices broadcast across T_lat. Used directly as
    attr_cls in FaderRAVE (no re-bucketing).
    """

    @abstractmethod
    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        """Shape (len(discrete_names), T_lat)."""
