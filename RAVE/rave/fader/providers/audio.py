"""On-the-fly librosa + timbral continuous attribute extraction."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import gin
import numpy as np

from ..attributes import compute_descriptor_matrix
from .base import ContinuousAttributeProvider


def _audio_cache_key(index: int, audio: np.ndarray) -> Tuple[int, int]:
    return (index, hash(audio.tobytes()))


@gin.configurable
class CachingAudioDescriptorProvider(ContinuousAttributeProvider):
    """LRU cache over cropped waveforms (index + audio hash)."""

    def __init__(
        self,
        inner: ContinuousAttributeProvider,
        max_entries: int = 4096,
    ) -> None:
        self._inner = inner
        self._max_entries = max(1, max_entries)
        self._cache: OrderedDict[Tuple[int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        key = _audio_cache_key(index, audio)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        out = self._inner.load(index, audio, sr, latent_length)
        self._cache[key] = out
        if len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return out


@gin.configurable
class AudioDescriptorProvider(ContinuousAttributeProvider):
    """
    Default continuous backend: librosa + timbral extractors on mono audio.

    Used at train time (AttributeLoader) and at eval/inference when re-extracting
    attributes from synthesized waveforms.
    """

    def __init__(
        self,
        continuous_attributes: Sequence[str],
        sampling_rate: int,
    ) -> None:
        self.continuous_attributes = list(continuous_attributes)
        self.sr = sampling_rate

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: int,
        latent_length: int,
    ) -> np.ndarray:
        if audio.ndim == 2:
            mono = audio.mean(axis=0)
        else:
            mono = audio.reshape(-1)
        return compute_descriptor_matrix(
            mono,
            sr=sr,
            descriptors=self.continuous_attributes,
            latent_length=latent_length,
        )
