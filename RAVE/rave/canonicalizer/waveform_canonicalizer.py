"""Waveform-domain input canonicalizer: per-input knob encoder + EQ + reverb."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dsp import BiquadBank, CausalReverb

# Neutral wet logit → sigmoid ≈ 0 (dry / identity pass-through).
_WET_NEUTRAL_LOGIT = -20.0


@gin.configurable
@dataclass(frozen=True)
class WaveformKnobLayout:
    """Knob vector layout: ``[eq_gain_0 … eq_gain_{n-1}, reverb_0 … reverb_{m-1}]``."""

    n_eq_bands: int = 6
    n_reverb_knobs: int = 7

    @property
    def n_knobs(self) -> int:
        return self.n_eq_bands + self.n_reverb_knobs

    def split(self, knobs: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if knobs.shape[-1] != self.n_knobs:
            raise ValueError(
                f"expected knob dim {self.n_knobs}, got {knobs.shape[-1]}"
            )
        eq_k = knobs[..., :self.n_eq_bands]
        if self.n_reverb_knobs == 0:
            return eq_k, None
        rev_k = knobs[..., self.n_eq_bands:]
        return eq_k, rev_k


def layout_from_modules(
    eq: BiquadBank,
    reverb: Optional[CausalReverb],
    *,
    use_reverb: bool,
) -> WaveformKnobLayout:
    n_rev = reverb.n_knobs if (use_reverb and reverb is not None) else 0
    return WaveformKnobLayout(n_eq_bands=eq.n_bands, n_reverb_knobs=n_rev)


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.act(self.conv(x))


@gin.configurable
class WaveformKnobEncoder(nn.Module):
    """
    Causal conv encoder: audio (B, C, T) → knob vector (B, K).

    EQ slots are mapped to dB via ``tanh(raw) * max_gain_db``.
    Reverb slots are pre-activations consumed directly by ``CausalReverb``.
    """

    def __init__(
        self,
        layout: WaveformKnobLayout,
        in_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 4,
        max_gain_db: float = 12.0,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.max_gain_db = max_gain_db

        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(_CausalConvBlock(ch, hidden_channels))
            ch = hidden_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, layout.n_knobs)
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.head.weight)
        bias = torch.zeros(self.layout.n_knobs)
        # EQ bands: tanh(0) → 0 dB
        if self.layout.n_reverb_knobs > 0:
            bias[self.layout.n_eq_bands] = _WET_NEUTRAL_LOGIT
        nn.init.copy_(self.head.bias, bias)

    def _map_knobs(self, raw: torch.Tensor) -> torch.Tensor:
        eq_end = self.layout.n_eq_bands
        eq = torch.tanh(raw[:, :eq_end]) * self.max_gain_db
        if self.layout.n_reverb_knobs == 0:
            return eq
        rev = raw[:, eq_end:]
        return torch.cat([eq, rev], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) audio, got shape {tuple(x.shape)}")
        h = self.backbone(x)
        h = h.mean(dim=-1)
        return self._map_knobs(self.head(h))


@gin.configurable
class WaveformCanonicalizer(nn.Module):
    """
    C(x) applied before PQMF encode.

    Per-input flow: ``x → encoder → knobs (B, K) → EQ → optional reverb``.
    Identity at init via neutral knob predictions.
    """

    def __init__(
        self,
        encoder: WaveformKnobEncoder,
        eq: BiquadBank,
        reverb: Optional[CausalReverb] = None,
        use_reverb: bool = True,
        layout: Optional[WaveformKnobLayout] = None,
        knob_ema_decay: Optional[float] = 0.95,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.eq = eq
        self.reverb = reverb
        self.use_reverb = use_reverb
        self.knob_ema_decay = knob_ema_decay
        self.register_buffer("knob_ema", torch.zeros(0), persistent=True)

        if layout is None:
            layout = layout_from_modules(eq, reverb, use_reverb=use_reverb)
        if layout.n_knobs != encoder.layout.n_knobs:
            raise ValueError(
                f"encoder layout K={encoder.layout.n_knobs} "
                f"!= canonicalizer layout K={layout.n_knobs}"
            )
        if layout.n_eq_bands != eq.n_bands:
            raise ValueError(
                f"layout expects {layout.n_eq_bands} EQ bands, "
                f"eq has {eq.n_bands}"
            )
        if use_reverb and reverb is not None and layout.n_reverb_knobs != reverb.n_knobs:
            raise ValueError(
                f"layout expects {layout.n_reverb_knobs} reverb knobs, "
                f"reverb has {reverb.n_knobs}"
            )
        self.layout = layout

    def predict_knobs(self, x: torch.Tensor) -> torch.Tensor:
        knobs = self.encoder(x)
        return self._smooth_knobs(knobs)

    def _smooth_knobs(self, knobs: torch.Tensor) -> torch.Tensor:
        if self.knob_ema_decay is None:
            return knobs
        if self.training:
            return knobs

        ref = knobs.mean(dim=0, keepdim=True) if knobs.dim() == 2 else knobs.unsqueeze(0)
        if self.knob_ema.numel() == 0 or self.knob_ema.shape != ref.shape:
            self.knob_ema = ref.detach().clone()
        else:
            d = self.knob_ema_decay
            self.knob_ema = d * self.knob_ema + (1.0 - d) * ref.detach()
        if knobs.dim() == 2:
            return self.knob_ema.expand(knobs.shape[0], -1)
        return self.knob_ema.squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        knobs = self.predict_knobs(x)
        eq_k, rev_k = self.layout.split(knobs)
        x = self.eq(x, eq_k)
        if self.use_reverb and self.reverb is not None and rev_k is not None:
            x = self.reverb(x, rev_k)
        return x


def build_waveform_canonicalizer(
    sample_rate: float,
    *,
    n_channels: int = 1,
    use_reverb: bool = True,
) -> WaveformCanonicalizer:
    """Construct waveform canonicalizer from gin bindings (requires parsed canon gin)."""
    eq = gin.get_configurable(BiquadBank)(sample_rate=sample_rate)
    reverb = (
        gin.get_configurable(CausalReverb)(sample_rate=sample_rate)
        if use_reverb else None
    )
    layout = layout_from_modules(eq, reverb, use_reverb=use_reverb)
    encoder = gin.get_configurable(WaveformKnobEncoder)(
        layout=layout,
        in_channels=n_channels,
    )
    return gin.get_configurable(WaveformCanonicalizer)(
        encoder=encoder,
        eq=eq,
        reverb=reverb,
        layout=layout,
        use_reverb=use_reverb,
    )
