"""Latent-domain input canonicalizer: 128→128 warp after encode."""

from __future__ import annotations

import gin
import torch
import torch.nn as nn


@gin.configurable
class LatentCanonicalizer(nn.Module):
    """
    L(z) on content latent (B, latent_size, T_lat). Identity init via residual form.
    """

    def __init__(self, latent_size: int = 128) -> None:
        super().__init__()
        self.conv = nn.Conv1d(latent_size, latent_size, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            for i in range(self.conv.in_channels):
                self.conv.weight[i, i, 0] = 1.0

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha)
        return z + alpha * (self.conv(z) - z)
