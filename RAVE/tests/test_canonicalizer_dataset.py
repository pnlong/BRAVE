"""Tests for OOD Fader LMDB dataset wrapper."""

import numpy as np
import torch

from rave.canonicalizer.dataset import (
    DOMAIN_IN,
    DOMAIN_OOD,
    DualSourceCanonicalizerDataset,
    OodFaderDataset,
    StratifiedCanonicalizerBatchSampler,
    StratifiedCanonicalizerIterableDataset,
    TaggedAudioDataset,
    build_canonicalizer_dataloader,
    canonicalizer_collate,
    ddp_aligned_num_batches,
    ddp_batches_per_rank,
    stratified_domain_counts,
)
from rave.fader.dataset import FaderAttributeDataset
from rave.canonicalizer.ir_augmentation import ImpulseResponseAug, synthetic_room_ir


class _MockAudioDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 2) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

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


def test_stratified_domain_counts_half_split():
    assert stratified_domain_counts(4, 0.5) == (2, 2)
    assert stratified_domain_counts(8, 0.5) == (4, 4)


def test_stratified_batch_sampler_balanced_domains():
    in_ds = TaggedAudioDataset(_MockAudioDataset(), domain=DOMAIN_IN)
    ood_ds = TaggedAudioDataset(_MockAudioDataset(), domain=DOMAIN_OOD)
    dual = DualSourceCanonicalizerDataset(in_ds, ood_ds)
    sampler = StratifiedCanonicalizerBatchSampler(
        len_in_domain=dual.len_in_domain,
        len_ood=dual.len_ood,
        batch_size=4,
        in_domain_fraction=0.5,
        shuffle=False,
    )
    assert sampler.batch_size == 4
    batch_indices = next(iter(sampler))
    batch = canonicalizer_collate([dual[i] for i in batch_indices])
    _, _, domains = batch
    assert domains.count(DOMAIN_IN) == 2
    assert domains.count(DOMAIN_OOD) == 2


def test_ddp_aligned_batch_counts():
    assert ddp_aligned_num_batches(127, 2) == 126
    assert ddp_batches_per_rank(127, 2) == 63
    assert ddp_aligned_num_batches(127, 1) == 127
    assert ddp_batches_per_rank(127, 1) == 127


def test_iterable_dataset_equal_ddp_shard_lengths(monkeypatch):
    in_ds = TaggedAudioDataset(_MockAudioDataset(size=4), domain=DOMAIN_IN)
    ood_ds = TaggedAudioDataset(_MockAudioDataset(size=4), domain=DOMAIN_OOD)
    iterable = StratifiedCanonicalizerIterableDataset(
        in_domain_dataset=in_ds,
        ood_dataset=ood_ds,
        batch_size=4,
        in_domain_fraction=0.5,
        shuffle=False,
    )
    assert iterable._num_batches == 2

    monkeypatch.setattr(
        "rave.canonicalizer.dataset._ddp_rank_world",
        lambda: (0, 2),
    )
    assert len(iterable) == 1
    monkeypatch.setattr(
        "rave.canonicalizer.dataset._ddp_rank_world",
        lambda: (1, 2),
    )
    assert len(iterable) == 1


def test_build_canonicalizer_dataloader_stratified():
    in_ds = TaggedAudioDataset(_MockAudioDataset(), domain=DOMAIN_IN)
    ood_ds = TaggedAudioDataset(_MockAudioDataset(), domain=DOMAIN_OOD)
    loader = build_canonicalizer_dataloader(
        in_domain_dataset=in_ds,
        ood_dataset=ood_ds,
        batch_size=4,
        stratified_batches=True,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    _, _, domains = next(iter(loader))
    assert domains.count(DOMAIN_IN) == 2
    assert domains.count(DOMAIN_OOD) == 2
