"""AttributeLoader facade and gin factory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import gin
import numpy as np

from ..attributes import ordered_attributes
from .audio import AudioDescriptorProvider
from .base import ContinuousAttributeProvider, DiscreteAttributeProvider
from .null import NullDiscreteProvider
from .sidecar import SidecarAttributeProvider


class AttributeLoader:
    """
    Gin-built facade: dataset calls load(index, audio) -> (D_total, T_lat).

    Composes continuous + discrete providers internally.
    """

    def __init__(
        self,
        continuous_provider: Optional[ContinuousAttributeProvider],
        discrete_provider: Optional[DiscreteAttributeProvider],
        continuous_attributes: Sequence[str],
        discrete_attributes: Sequence[str],
        sampling_rate: int,
        latent_length: int,
        sidecar: Optional[SidecarAttributeProvider] = None,
    ) -> None:
        self.continuous_attributes = list(continuous_attributes)
        self.discrete_attributes = list(discrete_attributes)
        self.attribute_names = ordered_attributes(
            self.continuous_attributes, self.discrete_attributes)
        self.attribute_kinds = {
            name: "continuous" for name in self.continuous_attributes
        }
        self.attribute_kinds.update(
            {name: "discrete" for name in self.discrete_attributes})
        self.sr = sampling_rate
        self.latent_length = latent_length
        self._continuous = continuous_provider
        self._discrete = discrete_provider
        self._sidecar = sidecar

    @property
    def num_attributes(self) -> int:
        return len(self.attribute_names)

    def load(
        self,
        index: int,
        audio: np.ndarray,
        sr: Optional[int] = None,
    ) -> np.ndarray:
        sr = sr or self.sr
        t_lat = self.latent_length
        parts = []

        if self.continuous_attributes:
            if self._sidecar is not None and not self._continuous:
                cont = self._sidecar.load_continuous(
                    self.continuous_attributes, index, t_lat)
            elif self._continuous is not None:
                cont = self._continuous.load(index, audio, sr, t_lat)
            else:
                cont = np.zeros((len(self.continuous_attributes), t_lat),
                                dtype=np.float32)
            parts.append(cont)

        if self.discrete_attributes:
            if self._sidecar is not None and (
                    self._discrete is None
                    or isinstance(self._discrete, NullDiscreteProvider)):
                disc = self._sidecar.load_discrete(
                    self.discrete_attributes, index, t_lat)
            elif self._discrete is not None:
                disc = self._discrete.load(index, audio, sr, t_lat)
            else:
                disc = np.zeros((len(self.discrete_attributes), t_lat),
                                dtype=np.float32)
            parts.append(disc)

        if not parts:
            return np.zeros((0, t_lat), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)


@gin.configurable
def build_attribute_loader(
    continuous_attributes: Sequence[str],
    discrete_attributes: Sequence[str],
    sampling_rate: int,
    latent_length: int,
    db_path: Optional[str] = None,
    sidecar_path: Optional[str] = None,
    use_audio_descriptors: bool = True,
    cache_descriptors: bool = False,
    cache_max_entries: int = 4096,
    use_midi_cc: bool = False,
    use_learned_features: bool = False,
) -> AttributeLoader:
    """Gin factory: wire providers from config."""
    cont_names = list(continuous_attributes)
    disc_names = list(discrete_attributes)
    kinds = {n: "continuous" for n in cont_names}
    kinds.update({n: "discrete" for n in disc_names})

    sidecar = None
    if sidecar_path or db_path:
        path = sidecar_path or str(Path(db_path) / "attribute_sidecar.yaml")
        if Path(path).is_file():
            sidecar = SidecarAttributeProvider(
                sidecar_path=path,
                attribute_names=cont_names + disc_names,
                attribute_kinds=kinds,
            )

    continuous_provider = None
    if cont_names and use_audio_descriptors:
        inner = AudioDescriptorProvider(
            continuous_attributes=cont_names,
            sampling_rate=sampling_rate,
        )
        if cache_descriptors:
            from .audio import CachingAudioDescriptorProvider
            continuous_provider = CachingAudioDescriptorProvider(
                inner=inner,
                max_entries=cache_max_entries,
            )
        else:
            continuous_provider = inner

    if use_midi_cc and db_path:
        from .midi_cc import MidiCCSidecarProvider
        midi = MidiCCSidecarProvider(
            db_path=db_path,
            continuous_attributes=cont_names,
            sampling_rate=sampling_rate,
        )
        if continuous_provider is None and cont_names:
            continuous_provider = midi
        # MidiCC stub: not composed with audio provider in M3

    if use_learned_features and db_path:
        from .learned import LearnedFeatureProvider
        learned = LearnedFeatureProvider(
            db_path=db_path,
            continuous_attributes=cont_names,
        )
        if continuous_provider is None and cont_names:
            continuous_provider = learned

    discrete_provider = None
    if disc_names:
        discrete_provider = NullDiscreteProvider(discrete_attributes=disc_names)

    return AttributeLoader(
        continuous_provider=continuous_provider,
        discrete_provider=discrete_provider,
        continuous_attributes=cont_names,
        discrete_attributes=disc_names,
        sampling_rate=sampling_rate,
        latent_length=latent_length,
        sidecar=sidecar,
    )
