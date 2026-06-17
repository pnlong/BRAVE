"""Mixed backbone LMDB + OOD WAV sidecar for canonicalizer Stage-1 training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Union

import gin
import numpy as np
import torch
import torchaudio
from torch.utils import data

from .attributes import latent_length_from_config
from .ir_augmentation import ImpulseResponseAug
from .providers import AudioDescriptorProvider

# Batch domain tags (generic — not tied to a specific OOD source)
DOMAIN_IN = "in_domain"
DOMAIN_OOD = "ood"


class OodWavDataset(data.Dataset):
    """OOD WAV sidecar; extracts attrs at runtime."""

    def __init__(
        self,
        *,
        ood_path: Union[str, Path],
        sampling_rate: int,
        n_signal: int,
        continuous_attributes: Sequence[str],
        discrete_attributes: Sequence[str],
        latent_length: int,
        ir_augment: Optional[ImpulseResponseAug] = None,
    ) -> None:
        self.ood_path = Path(ood_path)
        self.sr = sampling_rate
        self.n_signal = n_signal
        self.latent_length = latent_length
        self.continuous_attributes = list(continuous_attributes)
        self.discrete_attributes = list(discrete_attributes)
        self._ir_augment = ir_augment if (
            ir_augment is not None and ir_augment.enabled) else None
        self._provider = AudioDescriptorProvider(
            continuous_attributes=self.continuous_attributes,
            sampling_rate=sampling_rate,
        )
        self._files = sorted(self.ood_path.glob("*.wav"))
        if not self._files:
            raise FileNotFoundError(f"No .wav files in ood_path={self.ood_path}")

    def __len__(self) -> int:
        return len(self._files)

    def _load_wav(self, path: Path) -> np.ndarray:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=True)
        x = torch.from_numpy(audio.T).float()
        if sr != self.sr:
            x = torchaudio.functional.resample(x, sr, self.sr)
        mono = x.mean(dim=0, keepdim=True)
        if mono.shape[-1] < self.n_signal:
            reps = int(np.ceil(self.n_signal / mono.shape[-1]))
            mono = mono.repeat(1, reps)
        if mono.shape[-1] > self.n_signal:
            start = random.randint(0, mono.shape[-1] - self.n_signal)
            mono = mono[:, start:start + self.n_signal]
        audio = mono.numpy()
        if self._ir_augment is not None:
            audio = self._ir_augment.maybe_apply(audio)
        return audio

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        audio_np = self._load_wav(self._files[index % len(self._files)])
        audio_t = torch.from_numpy(audio_np).float()
        mono = audio_np[0] if audio_np.ndim == 2 else audio_np.reshape(-1)
        attr_cont = self._provider.load(index, mono, self.sr, self.latent_length)
        parts = [attr_cont]
        if self.discrete_attributes:
            disc = np.zeros(
                (len(self.discrete_attributes), self.latent_length),
                dtype=np.float32,
            )
            parts.append(disc)
        attr = np.concatenate(parts, axis=0)
        return audio_t, torch.from_numpy(attr).float(), DOMAIN_OOD


class MixedCanonicalizerDataset(data.Dataset):
    """Sample in_domain_fraction from backbone LMDB, else OOD WAV sidecar."""

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
def build_canonicalizer_dataset(
    train_dataset: data.Dataset,
    sampling_rate: int,
    n_signal: int,
    n_bands: int,
    ratios: Sequence[int],
    continuous_attributes: Sequence[str],
    discrete_attributes: Sequence[str],
    ood_path: str,
    in_domain_fraction: float = 0.8,
    ir_path: str = "",
    ir_prob: float = 0.0,
    ir_wet_min: float = 0.15,
    ir_wet_max: float = 0.55,
    *,
    train_fraction: Optional[float] = None,
) -> MixedCanonicalizerDataset:
    frac = train_fraction if train_fraction is not None else in_domain_fraction
    t_lat = latent_length_from_config(n_signal, n_bands, ratios)
    ir_aug = None
    if ir_prob > 0.0:
        ir_aug = ImpulseResponseAug(
            ir_path=ir_path or None,
            sampling_rate=sampling_rate,
            prob=ir_prob,
            wet_min=ir_wet_min,
            wet_max=ir_wet_max,
        )
        if not ir_aug.enabled:
            ir_aug = None
    ood_ds = OodWavDataset(
        ood_path=ood_path,
        sampling_rate=sampling_rate,
        n_signal=n_signal,
        continuous_attributes=continuous_attributes,
        discrete_attributes=discrete_attributes,
        latent_length=t_lat,
        ir_augment=ir_aug,
    )
    return MixedCanonicalizerDataset(
        in_domain_dataset=train_dataset,
        ood_dataset=ood_ds,
        in_domain_fraction=frac,
    )
