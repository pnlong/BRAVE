"""
JIT-safe torch audio descriptors for Fader nn~ export.

Approximates librosa/timbral trajectories resampled to T_lat. Used inside
ScriptedFaderRAVE when attr_mode is extract or extract+scale.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_N_FFT = 2048
_HOP = 512


def _mono(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) or (B, 1, T) -> (B, T)."""
    if x.ndim == 3:
        return x.mean(dim=1)
    return x.reshape(x.shape[0], -1)


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

    Output rows align with ``continuous_attributes`` order; caller scatters
    into full (B, D_total, T_lat) using attribute index map.
    """

    def __init__(
        self,
        continuous_attributes: Sequence[str],
        sr: int,
        n_fft: int = _N_FFT,
        hop_length: int = _HOP,
    ) -> None:
        super().__init__()
        self.continuous_attributes = list(continuous_attributes)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer(
            "_window",
            torch.hann_window(n_fft),
        )
        freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        self.register_buffer("_freqs", freqs)
        # timbral brightness proxy: centroid / Nyquist
        self.register_buffer("_nyquist", torch.tensor(float(sr / 2)))

    def _stft_mag(self, mono: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, F, T_frames)."""
        spec = torch.stft(
            mono,
            self.n_fft,
            self.hop_length,
            window=self._window.to(mono.device),
            return_complex=True,
        )
        return spec.abs()

    def _rms(self, mono: torch.Tensor, t_lat: int) -> torch.Tensor:
        frames = mono.unfold(-1, self.n_fft, self.hop_length)
        rms = frames.pow(2).mean(dim=-1).sqrt()
        return _resample_rows(rms, t_lat)

    def _spectral(self, mag: torch.Tensor, t_lat: int) -> Dict[str, torch.Tensor]:
        """Shared spectral stats from magnitude STFT."""
        eps = 1e-8
        freq = self._freqs.to(mag.device).view(1, -1, 1)
        power = mag.pow(2) + eps
        total = power.sum(dim=1, keepdim=True)
        centroid = (freq * power).sum(dim=1) / total.squeeze(1).sum(dim=1, keepdim=True).clamp(min=eps)
        centroid = centroid / self._nyquist.to(mag.device)

        geo_mean = torch.exp(torch.log(power + eps).mean(dim=1))
        arith_mean = power.mean(dim=1) + eps
        flatness = geo_mean / arith_mean

        flux = mag.diff(dim=-1).abs().mean(dim=1)
        flux = F.pad(flux, (1, 0))

        nyq = self._nyquist.to(mag.device)
        high_mask = self._freqs.to(mag.device) > (0.5 * nyq)
        high = power[:, high_mask, :].sum(dim=1)
        bright = high / (total.squeeze(1).sum(dim=1) + eps)

        out = {
            "centroid": _resample_rows(centroid, t_lat),
            "flatness": _resample_rows(flatness, t_lat),
            "roughness": _resample_rows(flux, t_lat),
            "brightness": _resample_rows(bright, t_lat),
        }
        return out

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
        batch = mono.shape[0]
        d_cont = len(self.continuous_attributes)
        if d_cont == 0:
            return torch.zeros(batch, 0, t_lat, device=x.device, dtype=x.dtype)

        mag = self._stft_mag(mono)
        spectral = self._spectral(mag, t_lat)

        rows: List[torch.Tensor] = []
        for name in self.continuous_attributes:
            if name == "rms":
                rows.append(self._rms(mono, t_lat))
            elif name in spectral:
                rows.append(spectral[name])
            else:
                rows.append(torch.zeros(batch, t_lat, device=x.device, dtype=x.dtype))
        return torch.stack(rows, dim=1)
