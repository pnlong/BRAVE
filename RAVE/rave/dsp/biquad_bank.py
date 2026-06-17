"""Causal peaking EQ bank for waveform canonicalization."""

from __future__ import annotations

import math
from typing import Sequence

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        gain = torch.tanh(self.gain_db) * self.max_gain_db
        if gain.abs().item() < 1e-6:
            return x
        b, a = _peaking_biquad_coeffs(
            self.sample_rate, self.center_freq, self.q, gain.squeeze())
        b = b.to(dtype=x.dtype, device=x.device)
        a = a.to(dtype=x.dtype, device=x.device)

        y = x
        x_1 = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        x_2 = torch.zeros_like(x_1)
        y_1 = torch.zeros_like(x_1)
        y_2 = torch.zeros_like(x_1)

        outs = []
        for t in range(x.shape[-1]):
            xt = y[..., t:t + 1]
            yt = (
                b[..., 0:1] * xt
                + b[..., 1:2] * x_1
                + b[..., 2:3] * x_2
                - a[..., 1:2] * y_1
                - a[..., 2:3] * y_2
            )
            x_2, x_1 = x_1, xt
            y_2, y_1 = y_1, yt
            outs.append(yt)
        return torch.cat(outs, dim=-1)


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
        freqs = torch.logspace(
            math.log10(min_freq),
            math.log10(max_freq),
            n_bands,
        ).tolist()
        self.filters = nn.ModuleList([
            BiquadFilter(sample_rate, fc, q=q, max_gain_db=max_gain_db)
            for fc in freqs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for filt in self.filters:
            x = filt(x)
        return x
