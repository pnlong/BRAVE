"""Stationary spectral gate for RAVE preprocess (optional --denoise)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass(frozen=True)
class DenoiseConfig:
    """Per-channel spectral noise gate settings."""

    enabled: bool = False
    strength: float = 0.75
    noise_sec: float = 0.0
    n_fft: int = 2048

    @classmethod
    def disabled(cls) -> DenoiseConfig:
        return cls(enabled=False)


def _denoise_channel(
    y: np.ndarray,
    sr: int,
    *,
    strength: float,
    noise_sec: float,
    n_fft: int,
) -> np.ndarray:
    hop = n_fft // 4
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    mag = np.abs(spec)
    if noise_sec > 0:
        noise_frames = min(
            mag.shape[1],
            max(1, int(noise_sec * sr / hop)),
        )
        mag_est = mag[:, :noise_frames]
    else:
        mag_est = mag
    noise_prof = np.percentile(mag_est, 10, axis=1, keepdims=True)
    cleaned_mag = np.maximum(mag - strength * noise_prof, 0.0)
    spec_out = cleaned_mag * np.exp(1j * np.angle(spec))
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


def denoise_pcm(pcm: np.ndarray, sr: int, config: DenoiseConfig) -> np.ndarray:
    """
    Mild stationary noise reduction on float32 PCM ``(channels, samples)``.

    Noise floor per frequency bin: 10th percentile of STFT magnitudes over the
    clip (or over the first ``noise_sec`` when ``noise_sec > 0``).
    """
    if not config.enabled or pcm.size == 0:
        return pcm

    strength = float(np.clip(config.strength, 0.0, 1.0))
    if strength <= 0.0:
        return pcm

    n_fft = config.n_fft
    out_channels = []
    for ch in pcm:
        y = ch.astype(np.float32)
        if len(y) < n_fft:
            out_channels.append(ch.copy())
            continue
        out_channels.append(
            _denoise_channel(
                y,
                sr,
                strength=strength,
                noise_sec=config.noise_sec,
                n_fft=n_fft,
            ))
    return np.stack(out_channels, axis=0)
