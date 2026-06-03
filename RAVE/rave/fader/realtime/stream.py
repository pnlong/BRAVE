"""
Realtime continuous attribute stream for Fader inference.

Block-wise attribute extraction at latent frame rate for streaming decode.
Uses the same AudioDescriptorProvider + min/max normalize as training.

Typical usage (realtime_fader_demo.py):
  stream = AttributeStream.from_stats_path(stats_path, sr)
  attr_norm = stream.push(audio_block, latent_length=T_lat)
  y = trace_model(x_block, attr_norm.unsqueeze(0))
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..attributes import load_attribute_stats, normalize_attributes
from ..providers import AudioDescriptorProvider, ContinuousAttributeProvider


class AttributeStream:
    """
    Sliding-window attribute extraction normalized for Fader decode.

    push(audio_block) returns attr_norm (D_cont, T_block) for the block.
    """

    def __init__(
        self,
        continuous_attributes: Sequence[str],
        min_max_features: dict,
        sampling_rate: int,
        provider: Optional[ContinuousAttributeProvider] = None,
        discrete_norm: Optional[torch.Tensor] = None,
        attribute_names: Optional[Sequence[str]] = None,
        attribute_kinds: Optional[dict] = None,
    ) -> None:
        self.continuous_attributes = list(continuous_attributes)
        self.min_max_features = min_max_features
        self.sr = sampling_rate
        self._provider = provider or AudioDescriptorProvider(
            continuous_attributes=self.continuous_attributes,
            sampling_rate=sampling_rate,
        )
        self._discrete_norm = discrete_norm
        self.attribute_names = list(attribute_names or continuous_attributes)
        self.attribute_kinds = attribute_kinds or {
            n: "continuous" for n in self.continuous_attributes
        }

    @classmethod
    def from_stats_path(cls, stats_path: str, sampling_rate: int) -> "AttributeStream":
        """Build stream from attribute_stats.yaml."""
        stats = load_attribute_stats(stats_path)
        cont = stats.get("continuous_attributes", [])
        return cls(
            continuous_attributes=cont,
            min_max_features=stats["min_max_features"],
            sampling_rate=sampling_rate,
            attribute_names=stats.get("attribute_names", cont),
            attribute_kinds=stats.get("attribute_kinds", {}),
        )

    def push(
        self,
        audio_block: np.ndarray,
        *,
        index: int = 0,
        latent_length: int,
    ) -> torch.Tensor:
        """
        Extract + normalize attributes for one audio block.

        Returns:
            attr_norm: (D_total, T_lat) tensor ready for decoder concat
        """
        # --- Extract raw continuous trajectories for this block ---
        raw_cont = self._provider.load(
            index, audio_block, self.sr, latent_length)
        raw_t = torch.from_numpy(raw_cont).float().unsqueeze(0)
        # --- Min/max normalize continuous rows to [-1, 1] ---
        attr_norm = normalize_attributes(
            raw_t,
            self.continuous_attributes,
            self.min_max_features,
        ).squeeze(0)

        if self._discrete_norm is not None:
            # --- Append fixed discrete controls (e.g. user-selected class) ---
            attr_norm = torch.cat([attr_norm, self._discrete_norm], dim=0)
        return attr_norm
