"""Binary domain discriminator on VAE content latents (in-domain vs OOD)."""

from __future__ import annotations

import gin
import cached_conv as cc
import torch
import torch.nn as nn


@gin.configurable
class LatentDomainDiscriminator(nn.Module):
    """
    Patch-wise real/fake classifier on z (B, latent_size, T_lat).

  Used in canonicalizer Stage-1 to pull OOD latents toward the in-domain
  distribution without matching raw waveform timbre.
    """

    def __init__(
        self,
        latent_size: int = 128,
        base_channels: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = latent_size
        for _ in range(num_layers):
            layers.extend([
                cc.Conv1d(
                    ch_in,
                    base_channels,
                    7,
                    stride=1,
                    padding=cc.get_padding(7),
                    bias=False,
                ),
                nn.BatchNorm1d(base_channels),
                nn.LeakyReLU(0.2),
            ])
            ch_in = base_channels
        layers.append(
            cc.Conv1d(
                ch_in,
                1,
                7,
                stride=1,
                padding=cc.get_padding(7),
                bias=True,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return per-frame logits (B, 1, T_lat)."""
        return self.net(z)
