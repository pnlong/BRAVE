"""Small causal Schroeder-style reverb for waveform canonicalization."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batch_param(
    value: Union[float, torch.Tensor],
    batch: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast a scalar or (B,) tensor to (B, 1, 1) for audio (B, C, T)."""
    if not torch.is_tensor(value):
        t = torch.full((batch,), float(value), dtype=dtype, device=device)
    else:
        t = value.to(dtype=dtype, device=device).reshape(-1)
        if t.numel() == 1 and batch > 1:
            t = t.expand(batch)
        elif t.numel() != batch:
            raise ValueError(f"expected {batch} params, got {t.numel()}")
    return t.view(batch, 1, 1)


def _feedback_comb(
    x: torch.Tensor,
    delay_samples: int,
    feedback: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Exact feedback comb: ``y[n] = x[n] + fb * y[n - delay]`` (zero initial state).

    Implements the same filter as a dense ``lfilter`` with coeffs of length
    ``delay+1``, but in ``O(T)`` via delay-line blocks instead of ``O(T·delay)``.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
    if delay_samples < 1:
        raise ValueError("delay_samples must be >= 1")

    bsz, _, length = x.shape
    if length <= delay_samples:
        return x

    fb = _as_batch_param(feedback, bsz, dtype=x.dtype, device=x.device)
    blocks = [x[..., :delay_samples]]
    prev = blocks[0]
    pos = delay_samples
    while pos < length:
        end = min(pos + delay_samples, length)
        seg = end - pos
        curr = x[..., pos:end] + fb * prev[..., :seg]
        blocks.append(curr)
        prev = curr
        pos = end
    return torch.cat(blocks, dim=-1)


def _feedforward_allpass(
    x: torch.Tensor,
    delay_samples: int,
    gain: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Exact feedforward allpass matching the historical coeff form:

    ``y[n] = g * x[n] + (1 - g) * x[n - delay]`` (zeros for ``n < delay``).
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
    if delay_samples < 1:
        raise ValueError("delay_samples must be >= 1")

    bsz = x.shape[0]
    g = _as_batch_param(gain, bsz, dtype=x.dtype, device=x.device)
    x_delayed = F.pad(x, (delay_samples, 0))[..., : x.shape[-1]]
    return g * x + (1.0 - g) * x_delayed


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
            fb = torch.sigmoid(feedback) * 0.85
        return _feedback_comb(x, self.delay_samples, fb)


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
            g = torch.sigmoid(gain) * 0.7
        return _feedforward_allpass(x, self.delay_samples, g)


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

        if knobs.dim() != 2 or knobs.shape[0] != x.shape[0]:
            raise ValueError(
                f"knobs must be (B, n_knobs) matching x batch {x.shape[0]}, "
                f"got {tuple(knobs.shape)}"
            )
        if knobs.shape[-1] != self.n_knobs:
            raise ValueError(
                f"expected knobs with {self.n_knobs} slots, got {knobs.shape[-1]}"
            )

        wet = torch.sigmoid(knobs[:, 0]).view(-1, 1, 1)
        n_comb = self.n_combs
        comb_raw = knobs[:, 1:1 + n_comb]
        ap_raw = knobs[:, 1 + n_comb:]

        comb_sum = 0.0
        for i, comb in enumerate(self.combs):
            comb_sum = comb_sum + comb(x, feedback=comb_raw[:, i])
        rev = comb_sum / len(self.combs)
        for j, ap in enumerate(self.allpasses):
            rev = ap(rev, gain=ap_raw[:, j])
        return x + wet * (rev - x)
