"""Mixed in-domain + OOD datasets for canonicalizer Stage-1 training."""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Union

import gin
import numpy as np
import torch
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import Sampler

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


def stratified_domain_counts(
    batch_size: int,
    in_domain_fraction: float,
) -> tuple[int, int]:
    """Return (n_in_domain, n_ood) per batch with both counts >= 1."""
    if batch_size < 2:
        raise ValueError(
            f"stratified batching requires batch_size >= 2, got {batch_size}")
    n_in = int(round(batch_size * in_domain_fraction))
    n_in = max(1, min(batch_size - 1, n_in))
    n_ood = batch_size - n_in
    if n_ood < 1:
        raise ValueError(
            f"in_domain_fraction={in_domain_fraction} leaves no OOD slots "
            f"in batch_size={batch_size}")
    return n_in, n_ood


def can_stratify_batches(batch_size: int, in_domain_fraction: float) -> bool:
    try:
        stratified_domain_counts(batch_size, in_domain_fraction)
        return True
    except ValueError:
        return False


def _ddp_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def ddp_aligned_num_batches(num_batches: int, world_size: int = 1) -> int:
    """Drop trailing batches so every DDP rank runs the same number of steps."""
    if world_size <= 1 or num_batches <= 0:
        return num_batches
    return (num_batches // world_size) * world_size


def ddp_batches_per_rank(num_batches: int, world_size: int = 1) -> int:
    aligned = ddp_aligned_num_batches(num_batches, world_size)
    if world_size <= 1:
        return aligned
    return aligned // world_size


def _stratified_num_batches(
    len_in_domain: int,
    len_ood: int,
    n_in: int,
    n_ood: int,
    *,
    drop_last: bool,
) -> int:
    n_batches = min(len_in_domain // n_in, len_ood // n_ood)
    if drop_last:
        return n_batches
    return max(
        (len_in_domain + n_in - 1) // n_in,
        (len_ood + n_ood - 1) // n_ood,
    )


def _iter_stratified_batch_indices(
    len_in_domain: int,
    len_ood: int,
    n_in: int,
    n_ood: int,
    num_batches: int,
    *,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> Iterator[List[int]]:
    if num_batches == 0:
        return iter(())

    if shuffle:
        in_perm = torch.randperm(
            len_in_domain, generator=generator).tolist()
        ood_perm = torch.randperm(len_ood, generator=generator).tolist()
    else:
        in_perm = list(range(len_in_domain))
        ood_perm = list(range(len_ood))

    in_ptr = 0
    ood_ptr = 0
    for _ in range(num_batches):
        batch: List[int] = []
        for offset in range(n_in):
            batch.append(in_perm[(in_ptr + offset) % len(in_perm)])
        in_ptr += n_in
        for offset in range(n_ood):
            global_idx = len_in_domain + ood_perm[
                (ood_ptr + offset) % len(ood_perm)]
            batch.append(global_idx)
        ood_ptr += n_ood
        yield batch


class DualSourceCanonicalizerDataset(data.Dataset):
    """In-domain indices [0, len_in); OOD indices [len_in, len_in + len_ood)."""

    def __init__(
        self,
        in_domain_dataset: data.Dataset,
        ood_dataset: data.Dataset,
    ) -> None:
        self._in_domain = in_domain_dataset
        self._ood = ood_dataset
        self._len_in = len(in_domain_dataset)

    @property
    def len_in_domain(self) -> int:
        return self._len_in

    @property
    def len_ood(self) -> int:
        return len(self._ood)

    def __len__(self) -> int:
        return self._len_in + len(self._ood)

    def __getitem__(self, index: int):
        if index < self._len_in:
            return self._in_domain[index]
        return self._ood[index - self._len_in]


class StratifiedCanonicalizerBatchSampler(Sampler[List[int]]):
    """Yield batches with fixed in-domain / OOD counts (e.g. 2/2 for batch=4)."""

    def __init__(
        self,
        len_in_domain: int,
        len_ood: int,
        batch_size: int,
        in_domain_fraction: float = 0.5,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.n_in, self.n_ood = stratified_domain_counts(
            batch_size, in_domain_fraction)
        self.batch_size = batch_size
        self._len_in = len_in_domain
        self._len_ood = len_ood
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
        self._num_batches = _stratified_num_batches(
            len_in_domain,
            len_ood,
            self.n_in,
            self.n_ood,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        return _iter_stratified_batch_indices(
            self._len_in,
            self._len_ood,
            self.n_in,
            self.n_ood,
            self._num_batches,
            shuffle=self.shuffle,
            generator=self.generator,
        )


class StratifiedCanonicalizerIterableDataset(data.IterableDataset):
    """Yield stratified batches; shards across DDP ranks and DataLoader workers."""

    def __init__(
        self,
        in_domain_dataset: data.Dataset,
        ood_dataset: data.Dataset,
        batch_size: int,
        in_domain_fraction: float = 0.5,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self._in_domain = in_domain_dataset
        self._ood = ood_dataset
        self._len_in = len(in_domain_dataset)
        self._len_ood = len(ood_dataset)
        self.n_in, self.n_ood = stratified_domain_counts(
            batch_size, in_domain_fraction)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
        self._num_batches = _stratified_num_batches(
            self._len_in,
            self._len_ood,
            self.n_in,
            self.n_ood,
            drop_last=drop_last,
        )

    def _getitem(self, index: int):
        if index < self._len_in:
            return self._in_domain[index]
        return self._ood[index - self._len_in]

    def __len__(self) -> int:
        rank, world_size = _ddp_rank_world()
        aligned = ddp_aligned_num_batches(self._num_batches, world_size)
        rank_batches = list(range(rank, aligned, world_size))
        worker_info = data.get_worker_info()
        if worker_info is None:
            return len(rank_batches)
        return len(rank_batches[worker_info.id::worker_info.num_workers])

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, Optional[torch.Tensor], list[str]]]:
        rank, world_size = _ddp_rank_world()
        worker_info = data.get_worker_info()

        all_batches = list(
            _iter_stratified_batch_indices(
                self._len_in,
                self._len_ood,
                self.n_in,
                self.n_ood,
                self._num_batches,
                shuffle=self.shuffle,
                generator=self.generator,
            ))

        aligned = ddp_aligned_num_batches(len(all_batches), world_size)
        all_batches = all_batches[:aligned]
        rank_batches = all_batches[rank::world_size]
        if worker_info is not None:
            rank_batches = rank_batches[
                worker_info.id::worker_info.num_workers]

        for batch_indices in rank_batches:
            items = [self._getitem(i) for i in batch_indices]
            yield canonicalizer_collate(items)


class MixedCanonicalizerDataset(data.Dataset):
    """Sample in_domain_fraction from Y LMDB, else X OOD corpus (per sample)."""

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


@gin.configurable
def build_canonicalizer_dataloader(
    in_domain_dataset: data.Dataset,
    ood_dataset: data.Dataset,
    batch_size: int,
    *,
    in_domain_fraction: float = 0.5,
    stratified_batches: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    train_fraction: Optional[float] = None,
) -> data.DataLoader:
    """Build train/val DataLoader with optional stratified in/OOD batches."""
    frac = train_fraction if train_fraction is not None else in_domain_fraction
    use_stratified = stratified_batches and can_stratify_batches(batch_size, frac)

    if use_stratified:
        # IterableDataset avoids PyTorch Lightning re-instantiating a custom
        # batch_sampler when wrapping the train loader for DDP.
        iterable = StratifiedCanonicalizerIterableDataset(
            in_domain_dataset=in_domain_dataset,
            ood_dataset=ood_dataset,
            batch_size=batch_size,
            in_domain_fraction=frac,
            drop_last=drop_last,
            shuffle=shuffle,
        )
        return data.DataLoader(
            iterable,
            batch_size=None,
            num_workers=num_workers,
        )

    mixed = build_canonicalizer_dataset(
        in_domain_dataset=in_domain_dataset,
        ood_dataset=ood_dataset,
        in_domain_fraction=frac,
    )
    return data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=canonicalizer_collate,
    )
