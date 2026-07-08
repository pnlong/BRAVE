"""Waveform-domain input canonicalizer: EQ + optional causal reverb."""

from __future__ import annotations

from typing import Optional

import gin
import torch
import torch.nn as nn

from ..dsp import BiquadBank, CausalReverb


@gin.configurable
class WaveformCanonicalizer(nn.Module):
    """
    C(x) applied before PQMF encode. Chains biquad EQ and optional causal reverb.
    Both submodules init to identity.
    """

    def __init__(
        self,
        eq: BiquadBank,
        reverb: Optional[CausalReverb] = None,
        use_reverb: bool = True,
    ) -> None:
        super().__init__()
        self.eq = eq
        self.reverb = reverb
        self.use_reverb = use_reverb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eq(x)
        if self.use_reverb and self.reverb is not None:
            x = self.reverb(x)
        return x
