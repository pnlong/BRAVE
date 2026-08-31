"""Tests for stratified dual-domain BRAVE audio loader."""

import numpy as np
import torch

from rave.dataset import (
    StratifiedDualAudioIterableDataset,
    _dual_domain_counts,
    build_stratified_dual_dataloader,
)


class _MockAudio(torch.utils.data.Dataset):
    def __init__(self, size: int, marker: float) -> None:
        self._size = size
        self._marker = marker

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> np.ndarray:
        x = np.full((1, 64), self._marker, dtype=np.float32)
        x[0, 0] = float(index)
        return x


def test_dual_domain_counts_balanced():
    assert _dual_domain_counts(8, 0.5) == (4, 4)
    assert _dual_domain_counts(4, 0.25) == (1, 3)


def test_stratified_dual_batch_has_both_domains():
    ds_x = _MockAudio(20, marker=1.0)
    ds_y = _MockAudio(30, marker=2.0)
    it = StratifiedDualAudioIterableDataset(
        ds_x, ds_y, batch_size=8, domain_x_fraction=0.5, shuffle=False)
    batch = next(iter(it))
    assert batch.shape == (8, 1, 64)
    # Last channel sample is the marker (indices only in first sample)
    markers = batch[:, 0, -1].tolist()
    assert markers.count(1.0) == 4
    assert markers.count(2.0) == 4


def test_build_stratified_dual_dataloader_len():
    ds_x = _MockAudio(16, marker=1.0)
    ds_y = _MockAudio(16, marker=2.0)
    loader = build_stratified_dual_dataloader(
        ds_x, ds_y, batch_size=4, domain_x_fraction=0.5, num_workers=0)
    # min(16//2, 16//2) = 8 batches
    assert len(loader) == 8
    batch = next(iter(loader))
    assert batch.shape[0] == 4
