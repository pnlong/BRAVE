"""Mixed in-domain + OOD Fader LMDB datasets for canonicalizer Stage-1 training."""

from __future__ import annotations

import random
from typing import Optional

import gin
import numpy as np
import torch
from torch.utils import data

from .dataset import FaderAttributeDataset
from .ir_augmentation import ImpulseResponseAug

# Batch domain tags (generic — not tied to a specific OOD source)
DOMAIN_IN = "in_domain"
DOMAIN_OOD = "ood"


class OodFaderDataset(data.Dataset):
    """
    OOD Fader LMDB corpus with optional IR augmentation and domain tag.

    Uses the same FaderAttributeDataset loader path as in-domain data; IR is
    applied to the cropped waveform before attribute extraction.
    """

    def __init__(
        self,
        fader_dataset: FaderAttributeDataset,
        ir_augment: Optional[ImpulseResponseAug] = None,
    ) -> None:
        if not isinstance(fader_dataset, FaderAttributeDataset):
            raise TypeError("fader_dataset must be FaderAttributeDataset")
        self._fader = fader_dataset
        self._ir_augment = ir_augment if (
            ir_augment is not None and ir_augment.enabled) else None

    def __len__(self) -> int:
        return len(self._fader)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        audio = self._fader._base[index]
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        if self._ir_augment is not None:
            audio_np = self._ir_augment.maybe_apply(audio_np)

        attr = self._fader._loader.load(
            index, audio_np, sr=self._fader._loader.sr)
        return (
            torch.from_numpy(audio_np).float(),
            torch.from_numpy(attr).float(),
            DOMAIN_OOD,
        )


class MixedCanonicalizerDataset(data.Dataset):
    """Sample in_domain_fraction from backbone LMDB, else OOD Fader LMDB."""

    def __init__(
        self,
        in_domain_dataset: data.Dataset,
        ood_dataset: data.Dataset,
        in_domain_fraction: float = 0.8,
        *,
        train_fraction: Optional[float] = None,
    ) -> None:
        if train_fraction is not None:
            in_domain_fraction = train_fraction
        self._in_domain = in_domain_dataset
        self._ood = ood_dataset
        self.in_domain_fraction = in_domain_fraction
        self._len = max(len(in_domain_dataset), len(ood_dataset))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if random.random() < self.in_domain_fraction:
            idx = index % len(self._in_domain)
            audio, attr = self._in_domain[idx]
            return audio, attr, DOMAIN_IN
        idx = index % len(self._ood)
        return self._ood[idx]


@gin.configurable
def make_ir_augment(
    sampling_rate: int,
    ir_path: str = "",
    ir_prob: float = 0.0,
    ir_wet_min: float = 0.15,
    ir_wet_max: float = 0.55,
) -> Optional[ImpulseResponseAug]:
    if ir_prob <= 0.0:
        return None
    ir_aug = ImpulseResponseAug(
        ir_path=ir_path or None,
        sampling_rate=sampling_rate,
        prob=ir_prob,
        wet_min=ir_wet_min,
        wet_max=ir_wet_max,
    )
    return ir_aug if ir_aug.enabled else None


@gin.configurable
def build_canonicalizer_dataset(
    in_domain_dataset: data.Dataset,
    ood_dataset: data.Dataset,
    in_domain_fraction: float = 0.8,
    *,
    train_fraction: Optional[float] = None,
) -> MixedCanonicalizerDataset:
    frac = train_fraction if train_fraction is not None else in_domain_fraction
    return MixedCanonicalizerDataset(
        in_domain_dataset=in_domain_dataset,
        ood_dataset=ood_dataset,
        in_domain_fraction=frac,
    )
