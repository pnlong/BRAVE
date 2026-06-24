"""Small causal Schroeder-style reverb for waveform canonicalization."""

from __future__ import annotations

from typing import Sequence

import gin
import torch
import torch.nn as nn
import torchaudio.functional as AF


def _lfilter(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    device = x.device
    return AF.lfilter(x, a.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device), clamp=False)


def _comb_filter(x: torch.Tensor, delay_samples: int, feedback: torch.Tensor) -> torch.Tensor:
    """y[n] = x[n] + fb * y[n - delay]."""
    d = delay_samples
    fb = feedback.reshape(()).to(dtype=x.dtype, device=x.device)
    a = x.new_zeros(d + 1)
    a[0] = 1.0
    a[d] = -fb
    b = x.new_zeros(d + 1)
    b[0] = 1.0
    return _lfilter(x, b, a)


def _allpass_filter(x: torch.Tensor, delay_samples: int, gain: torch.Tensor) -> torch.Tensor:
    """y[n] = x[n-d] + g * (x[n] - x[n-d])."""
    d = delay_samples
    g = gain.reshape(()).to(dtype=x.dtype, device=x.device)
    a = x.new_zeros(d + 1)
    a[0] = 1.0
    b = x.new_zeros(d + 1)
    b[0] = g
    b[d] = 1.0 - g
    return _lfilter(x, b, a)


class _CombFilter(nn.Module):
    def __init__(self, delay_samples: int, feedback: float = 0.0) -> None:
        super().__init__()
        self.delay_samples = delay_samples
        self.feedback = nn.Parameter(torch.tensor([feedback], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fb = torch.sigmoid(self.feedback) * 0.85
        return _comb_filter(x, self.delay_samples, fb)


class _AllpassFilter(nn.Module):
    def __init__(self, delay_samples: int, gain: float = 0.5) -> None:
        super().__init__()
        self.delay_samples = delay_samples
        self.gain = nn.Parameter(torch.tensor([gain], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gain) * 0.7
        return _allpass_filter(x, self.delay_samples, g)


@gin.configurable
class CausalReverb(nn.Module):
    """
    Causal wet/dry reverb. Init with wet=0 → identity.
    """

    def __init__(
        self,
        sample_rate: float = 44100.0,
        comb_delays_ms: Sequence[float] = (29.7, 37.1, 41.1, 43.7),
        allpass_delays_ms: Sequence[float] = (5.0, 1.7),
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        comb_delays = [max(1, int(ms * sample_rate / 1000.0)) for ms in comb_delays_ms]
        allpass_delays = [max(1, int(ms * sample_rate / 1000.0)) for ms in allpass_delays_ms]

        self.combs = nn.ModuleList([_CombFilter(d, feedback=0.0) for d in comb_delays])
        self.allpasses = nn.ModuleList([_AllpassFilter(d) for d in allpass_delays])
        # wet init → ~0 at start (sigmoid(-20) ≈ 2e-9)
        self.wet_logit = nn.Parameter(torch.tensor(-20.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wet = torch.sigmoid(self.wet_logit)
        comb_sum = 0.0
        for comb in self.combs:
            comb_sum = comb_sum + comb(x)
        rev = comb_sum / len(self.combs)
        for ap in self.allpasses:
            rev = ap(rev)
        # Residual form: exact dry pass-through when wet=0, wet_logit stays in the graph.
        return x + wet * (rev - x)
