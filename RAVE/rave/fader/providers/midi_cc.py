"""MIDI CC sidecar provider (stub for future host integration)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gin
import numpy as np

from .base import ContinuousAttributeProvider
from .sidecar import SidecarAttributeProvider


@gin.configurable
class MidiCCSidecarProvider(ContinuousAttributeProvider):
    """
    Loads continuous trajectories from {db_path}/midi_cc_sidecar.yaml.

    Same YAML layout as attribute_sidecar.yaml. When the file is missing,
    returns zeros (no-op stub).
    """

    def __init__(
        self,
        db_path: str,
        continuous_attributes: Sequence[str],
        sampling_rate: int,
        sidecar_filename: str = "midi_cc_sidecar.yaml",
    ) -> None:
        self.continuous_attributes = list(continuous_attributes)
        self.sr = sampling_rate
        path = Path(db_path) / sidecar_filename
        kinds = {n: "continuous" for n in self.continuous_attributes}
        self._sidecar = SidecarAttributeProvider(
            sidecar_path=str(path),
            attribute_names=self.continuous_attributes,
            attribute_kinds=kinds,
        )

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        return self._sidecar.load_continuous(
            self.continuous_attributes, index, latent_length)
