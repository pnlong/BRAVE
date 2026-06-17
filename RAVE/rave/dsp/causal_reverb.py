"""Small causal Schroeder-style reverb for waveform canonicalization."""

from __future__ import annotations

from typing import Sequence

import gin
import torch
import torch.nn as nn


class _CombFilter(nn.Module):
    def __init__(self, delay_samples: int, feedback: float = 0.0) -> None:
        super().__init__()
        self.delay_samples = delay_samples
        self.feedback = nn.Parameter(torch.tensor([feedback], dtype=torch.float32))
        self.register_buffer("_buf", torch.zeros(1, 1, delay_samples), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, t = x.shape
        if self._buf.shape[0] != b or self._buf.shape[1] != c:
            self._buf = torch.zeros(b, c, self.delay_samples, device=x.device, dtype=x.dtype)

        fb = torch.sigmoid(self.feedback) * 0.85
        out = []
        buf = self._buf
        d = self.delay_samples
        for i in range(t):
            delayed = buf[..., :1]
            sample = x[..., i:i + 1] + fb * delayed
            out.append(sample)
            buf = torch.cat([buf[..., 1:], sample], dim=-1)
        self._buf = buf.detach()
        return torch.cat(out, dim=-1)


class _AllpassFilter(nn.Module):
    def __init__(self, delay_samples: int, gain: float = 0.5) -> None:
        super().__init__()
        self.delay_samples = delay_samples
        self.gain = nn.Parameter(torch.tensor([gain], dtype=torch.float32))
        self.register_buffer("_buf", torch.zeros(1, 1, delay_samples), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if self._buf.shape[0] != b or self._buf.shape[1] != c:
            self._buf = torch.zeros(b, c, self.delay_samples, device=x.device, dtype=x.dtype)

        g = torch.sigmoid(self.gain) * 0.7
        out = []
        buf = self._buf
        for i in range(t):
            delayed = buf[..., :1]
            sample_in = x[..., i:i + 1]
            ap_out = delayed + g * (sample_in - delayed)
            out.append(ap_out)
            buf = torch.cat([buf[..., 1:], sample_in], dim=-1)
        self._buf = buf.detach()
        return torch.cat(out, dim=-1)


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
        if wet.item() < 1e-6:
            return x

        rev = x
        comb_sum = 0.0
        for comb in self.combs:
            comb_sum = comb_sum + comb(x)
        rev = comb_sum / len(self.combs)
        for ap in self.allpasses:
            rev = ap(rev)
        return (1.0 - wet) * x + wet * rev
