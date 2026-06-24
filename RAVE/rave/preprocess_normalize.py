"""Peak normalization for RAVE preprocess (optional --normalize)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizeConfig:
    """Per-chunk peak normalize (same rule as ``rave.dataset.normalize_signal``)."""

    enabled: bool = False
    max_gain_db: float = 30.0

    @classmethod
    def disabled(cls) -> NormalizeConfig:
        return cls(enabled=False)


def normalize_pcm(pcm: np.ndarray, config: NormalizeConfig) -> np.ndarray:
    """
    Boost each chunk toward 0 dBFS peak, capping gain at ``max_gain_db``.

    Args:
        pcm: float32 ``(channels, samples)`` in approximately [-1, 1].
    """
    if not config.enabled or pcm.size == 0:
        return pcm

    peak = float(np.max(np.abs(pcm)))
    if peak <= 0.0:
        return pcm

    log_peak = 20.0 * np.log10(peak)
    log_gain = min(float(config.max_gain_db), -log_peak)
    gain = 10.0 ** (log_gain / 20.0)
    out = pcm.astype(np.float32) * gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)
