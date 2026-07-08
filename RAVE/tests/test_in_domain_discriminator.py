import os
from pathlib import Path

import gin
import torch
import torch.nn as nn

from rave.canonicalizer.in_domain_discriminator import (
    InDomainAudioDiscriminator,
    build_in_domain_discriminator,
)

_BRAVE = Path(__file__).resolve().parents[2]


def test_in_domain_discriminator_forward():
    class TinyDisc(nn.Module):
        def forward(self, x):
            return [[x.mean(dim=-1, keepdim=True)], [x[..., ::2].mean(dim=-1, keepdim=True)]]

    disc = InDomainAudioDiscriminator(
        discriminator=lambda n_channels=1: TinyDisc(), n_channels=1)
    x = torch.randn(3, 1, 4096)
    out = disc(x)
    assert len(out) == 2
    assert out[0][-1].shape[0] == 3


def test_build_in_domain_discriminator_from_gin():
    gin.clear_config()
    prev = os.getcwd()
    os.chdir(_BRAVE / "configs")
    try:
        gin.parse_config_file("brave_canonicalizer.gin")
        disc = build_in_domain_discriminator(1)
    finally:
        os.chdir(prev)
    assert sum(p.numel() for p in disc.parameters()) > 500_000
    assert len(disc(torch.randn(1, 1, 4096))) == 1
