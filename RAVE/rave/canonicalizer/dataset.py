"""Mixed in-domain + OOD datasets for canonicalizer Stage-1 training."""

from __future__ import annotations

import random
from typing import Optional, Union

import gin
import numpy as np
import torch
from torch.utils import data

from ..fader.dataset import FaderAttributeDataset
from .ir_augmentation import ImpulseResponseAug

DOMAIN_IN = "in_domain"
DOMAIN_OOD = "ood"


class OodAudioDataset(data.Dataset):
    """OOD audio corpus with optional IR augmentation (plain or Fader-backed)."""

    def __init__(
        self,
        base_dataset: data.Dataset,
        ir_augment: Optional[ImpulseResponseAug] = None,
        *,
        fader_dataset: Optional[FaderAttributeDataset] = None,
    ) -> None:
        self._base = base_dataset
        self._fader = fader_dataset
        self._ir_augment = ir_augment if (
            ir_augment is not None and ir_augment.enabled) else None

    def __len__(self) -> int:
        return len(self._base)

    def _audio_at(self, index: int) -> np.ndarray:
        item = self._base[index]
        audio = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(audio, torch.Tensor):
            return audio.numpy()
        return np.asarray(audio, dtype=np.float32)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
        audio_np = self._audio_at(index)
        if self._ir_augment is not None:
            audio_np = self._ir_augment.maybe_apply(audio_np)

        audio = torch.from_numpy(audio_np).float()
        if self._fader is not None:
            attr = self._fader._loader.load(
                index, audio_np, sr=self._fader._loader.sr)
            return audio, torch.from_numpy(attr).float(), DOMAIN_OOD
        return audio, None, DOMAIN_OOD


class OodFaderDataset(OodAudioDataset):
    """Backward-compatible alias for Fader LMDB OOD data."""

    def __init__(
        self,
        fader_dataset: FaderAttributeDataset,
        ir_augment: Optional[ImpulseResponseAug] = None,
    ) -> None:
        if not isinstance(fader_dataset, FaderAttributeDataset):
            raise TypeError("fader_dataset must be FaderAttributeDataset")
        super().__init__(
            fader_dataset,
            ir_augment=ir_augment,
            fader_dataset=fader_dataset,
        )


class TaggedAudioDataset(data.Dataset):
    """Wrap plain audio dataset with a domain tag (no attributes)."""

    def __init__(self, base: data.Dataset, domain: str = DOMAIN_IN) -> None:
        self._base = base
        self.domain = domain

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, None, str]:
        item = self._base[index]
        audio = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
        return audio, None, self.domain


class TaggedFaderDataset(data.Dataset):
    """Fader dataset with explicit domain tag for mixed training."""

    def __init__(
        self,
        fader_dataset: FaderAttributeDataset,
        domain: str = DOMAIN_IN,
    ) -> None:
        if not isinstance(fader_dataset, FaderAttributeDataset):
            raise TypeError("fader_dataset must be FaderAttributeDataset")
        self._fader = fader_dataset
        self.domain = domain

    def __len__(self) -> int:
        return len(self._fader)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        audio, attr = self._fader[index]
        return audio, attr, self.domain


def canonicalizer_collate(
    batch: list,
) -> tuple[torch.Tensor, Optional[torch.Tensor], list[str]]:
    """Collate mixed plain / Fader batches into (audio, attr|None, domains)."""
    xs = torch.stack([b[0] for b in batch])
    domains = [b[2] if len(b) == 3 else b[1] for b in batch]
    attrs = None
    if len(batch[0]) == 3 and batch[0][1] is not None:
        attrs = torch.stack([b[1] for b in batch])
    return xs, attrs, domains


class MixedCanonicalizerDataset(data.Dataset):
    """Sample in_domain_fraction from Y LMDB, else X OOD corpus."""

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

    def __getitem__(self, index: int):
        if random.random() < self.in_domain_fraction:
            idx = index % len(self._in_domain)
            return self._in_domain[idx]
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
