"""
JIT-safe torch audio descriptors for Fader nn~ export.

Approximates librosa/timbral trajectories resampled to T_lat. Used inside
ScriptedFaderRAVE when attr_mode is extract or extract+scale.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


_N_FFT = 2048
_HOP = 512
# Match training crop: 131072 samples @ 44.1 kHz ≈ 2.97 s (see brave.gin n_signal).
_DEFAULT_MAX_HIST = 131072

# Librosa-compatible descriptors with faithful torch proxies. Timbral attrs
# (roughness, brightness, …) must use manual defaults at runtime — see nn_module.
TORCH_EXTRACTABLE = frozenset({"rms", "flatness", "centroid", "zcr", "f0"})


def _mono(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) or (B, 1, T) -> (B, T)."""
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.reshape(x.shape[0], -1)


def _ensure_min_samples(mono: torch.Tensor, min_len: int) -> torch.Tensor:
    """Right-pad mono so STFT / framing work on nn~ block sizes (e.g. 512)."""
    t = mono.shape[-1]
    if t < min_len:
        return F.pad(mono, (0, min_len - t))
    return mono


def _resample_rows(feat: torch.Tensor, target_len: int) -> torch.Tensor:
    """(B, T_feat) -> (B, target_len)."""
    if feat.shape[-1] == target_len:
        return feat
    return F.interpolate(
        feat.unsqueeze(1),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(1)


class TorchDescriptorExtract(nn.Module):
    """
    Extract continuous descriptor trajectories from waveform blocks.

    Maintains a rolling audio history so live nn~ blocks (e.g. 512 samples)
    still have enough context for STFT-based descriptors.

    Output rows align with ``continuous_attributes`` order; caller scatters
    into full (B, D_total, T_lat) using attribute index map.
    """

    def __init__(
        self,
        continuous_attributes: Sequence[str],
        sr: int,
        n_fft: int = _N_FFT,
        hop_length: int = _HOP,
        max_history: int = _DEFAULT_MAX_HIST,
    ) -> None:
        super().__init__()
        self.continuous_attributes = list(continuous_attributes)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_history = int(max_history)
        self.register_buffer(
            "_window",
            torch.hann_window(n_fft),
        )
        freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        self.register_buffer("_freqs", freqs)
        self.register_buffer("_hist", torch.zeros(1, self.max_history))
        self.register_buffer("_hist_len", torch.zeros((), dtype=torch.long))
        n = len(self.continuous_attributes)
        self.register_buffer("_row_scale", torch.ones(n))
        self.register_buffer("_row_bias", torch.zeros(n))

    @torch.jit.export
    def reset_history(self) -> int:
        """Clear rolling audio buffer (e.g. after source switch)."""
        self._hist.zero_()
        self._hist_len.zero_()
        return 0

    def _update_history(self, mono: torch.Tensor) -> torch.Tensor:
        """Append mono block to rolling buffer; return full analysis window."""
        batch = mono.shape[0]
        if batch != self._hist.shape[0]:
            self._hist = torch.zeros(
                batch,
                self.max_history,
                device=mono.device,
                dtype=mono.dtype,
            )
            self._hist_len.zero_()

        old_len = int(self._hist_len.item())
        if old_len > 0:
            mono = torch.cat([self._hist[:, :old_len], mono], dim=-1)

        if mono.shape[-1] > self.max_history:
            mono = mono[:, -self.max_history :]

        new_len = mono.shape[-1]
        self._hist.zero_()
        self._hist[:, :new_len] = mono
        self._hist_len.fill_(new_len)
        return mono

    def _stft_mag(self, mono: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, F, T_frames)."""
        mono = _ensure_min_samples(mono, self.n_fft)
        spec = torch.stft(
            mono,
            self.n_fft,
            self.hop_length,
            window=self._window.to(mono.device),
            return_complex=True,
        )
        return spec.abs()

    def _rms(self, mono: torch.Tensor, t_lat: int) -> torch.Tensor:
        mono = _ensure_min_samples(mono, self.n_fft)
        frames = mono.unfold(-1, self.n_fft, self.hop_length)
        rms = frames.pow(2).mean(dim=-1).sqrt()
        return _resample_rows(rms, t_lat)

    def _spectral(self, mag: torch.Tensor, t_lat: int) -> Dict[str, torch.Tensor]:
        """Shared spectral stats from magnitude STFT."""
        eps = 1e-8
        freq = self._freqs.to(mag.device).view(1, -1, 1)
        power = mag.pow(2) + eps
        total = power.sum(dim=1, keepdim=True)
        total_frames = total.squeeze(1).clamp(min=eps)
        centroid = (freq * mag).sum(dim=1) / mag.sum(dim=1).clamp(min=eps)

        geo_mean = torch.exp(torch.log(power + eps).mean(dim=1))
        arith_mean = power.mean(dim=1) + eps
        flatness = geo_mean / arith_mean

        return {
            "centroid": _resample_rows(centroid, t_lat),
            "flatness": _resample_rows(flatness, t_lat),
        }

    @torch.jit.export
    def forward(self, x: torch.Tensor, t_lat: int) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) audio
            t_lat: target latent frames

        Returns:
            (B, D_cont, T_lat) raw continuous descriptors
        """
        mono = _mono(x)
        mono = self._update_history(mono)
        batch = mono.shape[0]
        d_cont = len(self.continuous_attributes)
        if d_cont == 0:
            return torch.zeros(batch, 0, t_lat, device=x.device, dtype=x.dtype)

        mag = self._stft_mag(mono)
        spectral = self._spectral(mag, t_lat)

        rows: List[torch.Tensor] = []
        for i, name in enumerate(self.continuous_attributes):
            if name == "rms":
                row = self._rms(mono, t_lat)
            elif name in spectral:
                row = spectral[name]
            else:
                row = torch.zeros(batch, t_lat, device=x.device, dtype=x.dtype)
            rows.append(row * self._row_scale[i] + self._row_bias[i])
        return torch.stack(rows, dim=1)


def calibrate_torch_descriptor_extract(
    extractor: TorchDescriptorExtract,
    *,
    sr: int,
    block_size: int = 512,
) -> None:
    """
    Fit per-descriptor scale so torch proxies match librosa units.

    Feeds audio in nn~-sized blocks after filling the rolling buffer.
    Only calibrates ``TORCH_EXTRACTABLE`` names on the extractor.
    """
    import numpy as np

    from rave.fader.attributes import compute_descriptor_matrix

    names = [
        n for n in extractor.continuous_attributes if n in TORCH_EXTRACTABLE
    ]
    if not names:
        return

    scales = np.ones(len(names), dtype=np.float32)
    biases = np.zeros(len(names), dtype=np.float32)
    rng = np.random.default_rng(0)
    probe_len = int(extractor.max_history)
    t = np.linspace(0, probe_len / sr, probe_len, endpoint=False)
    probes = [
        0.5 * np.sin(2 * np.pi * 440 * t),
        0.35 * np.sin(2 * np.pi * 1200 * t),
        0.12 * rng.standard_normal(probe_len),
    ]
    t_lat = 32
    n_blocks = probe_len // block_size

    with torch.no_grad():
        for i, name in enumerate(names):
            lib_means: List[float] = []
            torch_means: List[float] = []
            for audio in probes:
                audio = audio.astype(np.float32)
                lib = compute_descriptor_matrix(
                    audio, sr=sr, descriptors=[name], latent_length=t_lat)
                extractor.reset_history()
                x = torch.from_numpy(audio.reshape(1, 1, -1))
                for b in range(n_blocks):
                    chunk = x[..., b * block_size : (b + 1) * block_size]
                    if chunk.shape[-1] < block_size:
                        chunk = F.pad(chunk, (0, block_size - chunk.shape[-1]))
                    tout = extractor(chunk, t_lat)
                lib_means.append(float(lib[0].mean()))
                torch_means.append(float(tout[0, i].mean()))
            lib_m = float(np.mean(lib_means))
            torch_m = float(np.mean(torch_means))
            if abs(torch_m) > 1e-8 and lib_m * torch_m > 0:
                scale = lib_m / torch_m
            else:
                scale = 1.0
            scales[i] = float(np.clip(abs(scale), 0.5, 2.0))
            biases[i] = 0.0

    idx = {n: j for j, n in enumerate(names)}
    for j, name in enumerate(extractor.continuous_attributes):
        if name in idx:
            k = idx[name]
            extractor._row_scale[j] = float(scales[k])
            extractor._row_bias[j] = float(biases[k])
