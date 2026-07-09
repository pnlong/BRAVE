"""Shared canonicalizer attachment helpers for RAVE and FaderRAVE."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .latent_canonicalizer import LatentCanonicalizer
from .waveform_canonicalizer import build_waveform_canonicalizer


def attach_canonicalizer_modules(
    model: nn.Module,
    state_dict: dict,
    canonicalizer_type: str,
) -> None:
    """Load warp weights onto a backbone with canonicalizer slots."""
    device = next(model.parameters()).device
    if canonicalizer_type == "waveform":
        n_channels = int(getattr(model, "n_channels", 1))
        warp = build_waveform_canonicalizer(
            sample_rate=model.sr,
            n_channels=n_channels,
        )
        warp.load_state_dict(state_dict)
        model.waveform_canonicalizer = warp.to(device)
    elif canonicalizer_type == "latent":
        warp = LatentCanonicalizer(latent_size=model.latent_size)
        warp.load_state_dict(state_dict)
        model.latent_canonicalizer = warp.to(device)
    else:
        raise ValueError(f"unknown canonicalizer_type: {canonicalizer_type}")


def backbone_num_attributes(model: nn.Module) -> int:
    return int(getattr(model, "num_attributes", 0))


def prepare_decode_attributes(
    model: nn.Module,
    attr_raw: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return normalized attrs for decode, or None for unconditional."""
    n_attr = backbone_num_attributes(model)
    if n_attr == 0 or attr_raw is None:
        return None
    attr_norm, _ = model._prepare_attributes(attr_raw)
    return attr_norm
