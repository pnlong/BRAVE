"""PCEN (Per-Channel Energy Normalization) for RAVE preprocess (optional --pcen).

Applies librosa PCEN in the mel domain, then resynthesizes waveform by scaling
the linear STFT magnitude with mel-derived gains (phase preserved). Suited to
field recordings where steady environmental noise should be suppressed and
short transients (e.g. bird calls) emphasized.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class PcenConfig:
    """Per-channel PCEN settings (see librosa.pcen)."""

    enabled: bool = False
    n_fft: int = 2048
    hop_length: int = 0
    n_mels: int = 128
    gain: float = 0.98
    bias: float = 2.0
    power: float = 0.5
    time_constant: float = 0.4
    eps: float = 1e-6
    max_gain: float = 10.0

    @classmethod
    def disabled(cls) -> PcenConfig:
        return cls(enabled=False)

    @property
    def hop(self) -> int:
        return self.hop_length if self.hop_length > 0 else self.n_fft // 4


def _pcen_channel(y: np.ndarray, sr: int, config: PcenConfig) -> np.ndarray:
    n_fft = config.n_fft
    hop = config.hop
    if len(y) < n_fft:
        return y.copy()

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    mag = np.abs(stft).astype(np.float64)

    mel = librosa.feature.melspectrogram(
        S=mag,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=config.n_mels,
        power=1.0,
    )
    mel = np.maximum(mel, 0.0)

    pcen = librosa.pcen(
        mel,
        sr=sr,
        hop_length=hop,
        gain=config.gain,
        bias=config.bias,
        power=config.power,
        time_constant=config.time_constant,
        eps=config.eps,
    )

    ratio = pcen / (mel + config.eps)
    if config.max_gain > 0:
        ratio = np.clip(ratio, 0.0, config.max_gain)

    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=config.n_mels)
    inv_mel = np.linalg.pinv(mel_basis)
    gain_linear = inv_mel @ ratio
    if config.max_gain > 0:
        gain_linear = np.clip(gain_linear, 0.0, config.max_gain)

    mag_out = mag * gain_linear
    spec_out = mag_out * np.exp(1j * np.angle(stft))
    y_out = librosa.istft(
        spec_out,
        hop_length=hop,
        n_fft=n_fft,
        center=True,
        length=len(y),
    )

    peak = float(np.max(np.abs(y_out)) + 1e-12)
    if peak > 1.0:
        y_out = y_out / peak
    return y_out.astype(np.float32)


def pcen_pcm(pcm: np.ndarray, sr: int, config: PcenConfig) -> np.ndarray:
    """
    PCEN enhancement on float32 PCM ``(channels, samples)`` in [-1, 1].
    """
    if not config.enabled or pcm.size == 0:
        return pcm

    out_channels = []
    for ch in pcm:
        y = ch.astype(np.float32)
        if len(y) < config.n_fft:
            out_channels.append(y.copy())
            continue
        out_channels.append(_pcen_channel(y, sr, config))
    return np.stack(out_channels, axis=0)
