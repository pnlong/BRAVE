"""
Fader training dataset: audio + attributes via injected AttributeLoader.

Design
------
The dataset is intentionally dumb: it loads audio from the base RAVE dataset and
delegates all attribute assembly to AttributeLoader.load(). Attribute names,
kinds, and extraction logic live in providers.py + gin config — swap attributes
without touching this file.
"""

from __future__ import annotations

from typing import Sequence, Union

import gin
import numpy as np
import torch
from torch.utils import data

from .attributes import latent_length_from_config
from .providers import AttributeLoader, build_attribute_loader


class FaderAttributeDataset(data.Dataset):
    """
    Wraps a RAVE AudioDataset and appends raw attribute trajectories.

    Returns:
        audio: (C, T) float32 — same as base dataset
        attr: (D_total, T_lat) float32 — raw values (continuous floats + discrete indices)
    """

    def __init__(
        self,
        base_dataset: data.Dataset,
        attribute_loader: AttributeLoader,
    ) -> None:
        super().__init__()
        self._base = base_dataset
        self._loader = attribute_loader

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Load + transform audio (includes RandomCrop when configured) ---
        audio = self._base[index]
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
            audio_t = audio.float()
        else:
            audio_np = np.asarray(audio)
            audio_t = torch.from_numpy(audio_np).float()

        # --- Delegate all attribute assembly to injected loader ---
        attr = self._loader.load(index, audio_np, sr=self._loader.sr)
        return audio_t, torch.from_numpy(attr).float()


@gin.configurable
def wrap_fader_dataset(
    dataset: data.Dataset,
    sampling_rate: int,
    n_signal: int,
    n_bands: int,
    ratios: Sequence[int],
    enabled: bool = True,
    db_path: str = "",
    continuous_attributes: Sequence[str] = (),
    discrete_attributes: Sequence[str] = (),
    attribute_loader: AttributeLoader = None,
) -> Union[data.Dataset, FaderAttributeDataset]:
    """
    Gin hook: wrap train/val sets when FaderRAVE is enabled.

    Wires build_attribute_loader from gin; dataset body stays fixed.
    """
    if not enabled:
        return dataset

    t_lat = latent_length_from_config(n_signal, n_bands, ratios)

    # --- Build loader from gin unless explicitly injected (tests) ---
    if attribute_loader is None:
        loader = build_attribute_loader(
            continuous_attributes=list(continuous_attributes),
            discrete_attributes=list(discrete_attributes),
            sampling_rate=sampling_rate,
            latent_length=t_lat,
            db_path=db_path or None,
        )
    else:
        loader = attribute_loader

    return FaderAttributeDataset(
        base_dataset=dataset,
        attribute_loader=loader,
    )
