"""Causal peaking EQ bank for waveform canonicalization."""

from __future__ import annotations

import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import torchaudio.functional as AF


def _peaking_biquad_coeffs(
    sample_rate: float,
    center_freq: float,
    q: float,
    gain_db: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RBJ peaking EQ coefficients; gain_db may be broadcast per batch."""
    a = 10 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * center_freq / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


class BiquadFilter(nn.Module):
    """Single causal peaking biquad with learnable gain in dB."""

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        q: float = 1.0,
        max_gain_db: float = 12.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.q = q
        self.max_gain_db = max_gain_db
        self.gain_db = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        gain_db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, C, T); gain_db: optional (B,) dB gains from an external encoder
        if gain_db is None:
            gain = torch.tanh(self.gain_db) * self.max_gain_db
            y_filt = self._apply_biquad(x, gain)
            alpha = torch.tanh((gain.abs() / self.max_gain_db) * 5.0)
            return x + alpha * (y_filt - x)

        outs = []
        for b in range(x.shape[0]):
            gain = gain_db[b]
            x_b = x[b:b + 1]
            y_filt = self._apply_biquad(x_b, gain)
            alpha = torch.tanh((gain.abs() / self.max_gain_db) * 5.0)
            outs.append(x_b + alpha * (y_filt - x_b))
        return torch.cat(outs, dim=0)

    def _apply_biquad(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        b, a = _peaking_biquad_coeffs(
            self.sample_rate, self.center_freq, self.q, gain.reshape(()))
        b = b.to(dtype=x.dtype, device=x.device)
        a = a.to(dtype=x.dtype, device=x.device)
        return AF.lfilter(x, a, b, clamp=False)


@gin.configurable
class BiquadBank(nn.Module):
    """Chain of causal peaking biquads with log-spaced center frequencies."""

    def __init__(
        self,
        sample_rate: float = 44100.0,
        n_bands: int = 6,
        min_freq: float = 80.0,
        max_freq: float = 12000.0,
        q: float = 1.0,
        max_gain_db: float = 12.0,
    ) -> None:
        super().__init__()
        self.max_gain_db = max_gain_db
        freqs = torch.logspace(
            math.log10(min_freq),
            math.log10(max_freq),
            n_bands,
        ).tolist()
        self.filters = nn.ModuleList([
            BiquadFilter(sample_rate, fc, q=q, max_gain_db=max_gain_db)
            for fc in freqs
        ])

    @property
    def n_bands(self) -> int:
        return len(self.filters)

    def forward(
        self,
        x: torch.Tensor,
        gains_db: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # gains_db: optional (B, n_bands) dB gains; falls back to internal params
        if gains_db is None:
            for filt in self.filters:
                x = filt(x)
            return x

        if gains_db.shape[-1] != self.n_bands:
            raise ValueError(
                f"expected gains_db with {self.n_bands} bands, got {gains_db.shape[-1]}"
            )
        for i, filt in enumerate(self.filters):
            x = filt(x, gain_db=gains_db[:, i])
        return x
