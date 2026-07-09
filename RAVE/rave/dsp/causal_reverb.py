"""Small causal Schroeder-style reverb for waveform canonicalization."""

from __future__ import annotations

from typing import Optional, Sequence

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

    def forward(
        self,
        x: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feedback is None:
            fb = torch.sigmoid(self.feedback) * 0.85
        else:
            fb = torch.sigmoid(feedback.reshape(())) * 0.85
        return _comb_filter(x, self.delay_samples, fb)


class _AllpassFilter(nn.Module):
    def __init__(self, delay_samples: int, gain: float = 0.5) -> None:
        super().__init__()
        self.delay_samples = delay_samples
        self.gain = nn.Parameter(torch.tensor([gain], dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gain is None:
            g = torch.sigmoid(self.gain) * 0.7
        else:
            g = torch.sigmoid(gain.reshape(())) * 0.7
        return _allpass_filter(x, self.delay_samples, g)


@gin.configurable
class CausalReverb(nn.Module):
    """
    Causal wet/dry reverb. Init with wet=0 → identity.

    External knob layout (``n_knobs`` scalars per batch item):
    ``[wet_logit, comb_fb_0, …, comb_fb_{n-1}, ap_gain_0, …, ap_gain_{m-1}]``
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

    @property
    def n_combs(self) -> int:
        return len(self.combs)

    @property
    def n_allpasses(self) -> int:
        return len(self.allpasses)

    @property
    def n_knobs(self) -> int:
        return 1 + self.n_combs + self.n_allpasses

    def _forward_single(
        self,
        x: torch.Tensor,
        wet_logit: torch.Tensor,
        comb_raw: torch.Tensor,
        ap_raw: torch.Tensor,
    ) -> torch.Tensor:
        wet = torch.sigmoid(wet_logit.reshape(()))
        comb_sum = 0.0
        for i, comb in enumerate(self.combs):
            comb_sum = comb_sum + comb(x, feedback=comb_raw[i])
        rev = comb_sum / len(self.combs)
        for j, ap in enumerate(self.allpasses):
            rev = ap(rev, gain=ap_raw[j])
        return x + wet * (rev - x)

    def forward(
        self,
        x: torch.Tensor,
        knobs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # knobs: optional (B, n_knobs) pre-activation values from an external encoder
        if knobs is None:
            wet = torch.sigmoid(self.wet_logit)
            comb_sum = 0.0
            for comb in self.combs:
                comb_sum = comb_sum + comb(x)
            rev = comb_sum / len(self.combs)
            for ap in self.allpasses:
                rev = ap(rev)
            return x + wet * (rev - x)

        if knobs.shape[-1] != self.n_knobs:
            raise ValueError(
                f"expected knobs with {self.n_knobs} slots, got {knobs.shape[-1]}"
            )

        outs = []
        n_comb = self.n_combs
        for b in range(x.shape[0]):
            row = knobs[b]
            outs.append(self._forward_single(
                x[b:b + 1],
                row[0],
                row[1:1 + n_comb],
                row[1 + n_comb:],
            ))
        return torch.cat(outs, dim=0)
