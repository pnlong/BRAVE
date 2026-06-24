"""Tests for OOD Fader LMDB dataset wrapper."""

import numpy as np
import torch

from rave.fader.canonicalizer_dataset import DOMAIN_OOD, OodFaderDataset
from rave.fader.dataset import FaderAttributeDataset
from rave.fader.ir_augmentation import ImpulseResponseAug, synthetic_room_ir


class _MockAudioDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> np.ndarray:
        rng = np.random.default_rng(index)
        return rng.standard_normal((1, 4096)).astype(np.float32)


class _MockAttributeLoader:
    sr = 44100

    def load(self, index: int, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        return np.zeros((2, 32), dtype=np.float32)


def _make_fader_dataset() -> FaderAttributeDataset:
    return FaderAttributeDataset(
        base_dataset=_MockAudioDataset(),
        attribute_loader=_MockAttributeLoader(),
    )


def test_ood_fader_dataset_loads():
    ds = OodFaderDataset(_make_fader_dataset())
    audio, attr, domain = ds[0]
    assert domain == DOMAIN_OOD
    assert audio.shape[-1] == 4096
    assert attr.shape == (2, 32)


def test_ood_fader_dataset_ir_augment():
    ir_aug = ImpulseResponseAug(
        ir_path=None,
        sampling_rate=44100,
        prob=1.0,
        use_synthetic_fallback=True,
    )
    base = _make_fader_dataset()
    dry = base[0][0].numpy()
    ds = OodFaderDataset(base, ir_augment=ir_aug)
    wet, _, domain = ds[0]
    assert domain == DOMAIN_OOD
    assert not np.allclose(wet.numpy(), dry)
