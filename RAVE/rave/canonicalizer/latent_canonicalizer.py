"""Latent-domain input canonicalizer: 128→128 warp after encode."""

from __future__ import annotations

from typing import Optional, Sequence

import gin
import torch
import torch.nn as nn

from .attribute_conditioning import AttributeConditioningEmbed, FiLM


@gin.configurable
class LatentCanonicalizer(nn.Module):
    """
    L(z) on content latent (B, latent_size, T_lat). Identity init via residual form.

    Optional ``attr_cls`` FiLM modulates the 1×1 conv output when ``num_attributes > 0``.
    """

    def __init__(
        self,
        latent_size: int = 128,
        num_attributes: int = 0,
        num_classes_per_attribute: Optional[Sequence[int]] = None,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.num_attributes = int(num_attributes)
        self.conv = nn.Conv1d(latent_size, latent_size, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.cond_embed: Optional[AttributeConditioningEmbed] = None
        self.film: Optional[FiLM] = None
        if self.num_attributes > 0:
            if not num_classes_per_attribute:
                raise ValueError(
                    "num_classes_per_attribute required when num_attributes > 0")
            self.cond_embed = AttributeConditioningEmbed(
                num_attributes=self.num_attributes,
                num_classes_per_attribute=num_classes_per_attribute,
                embed_dim=embed_dim,
                condition_on="attr_cls",
            )
            self.film = FiLM(self.cond_embed.out_dim, latent_size)

        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            for i in range(self.conv.in_channels):
                self.conv.weight[i, i, 0] = 1.0

    def forward(
        self,
        z: torch.Tensor,
        attr_cls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        warped = self.conv(z)
        if self.film is not None and self.cond_embed is not None and attr_cls is not None:
            cond = self.cond_embed(attr_cls=attr_cls)
            warped = self.film(warped, cond)
        alpha = torch.sigmoid(self.alpha)
        return z + alpha * (warped - z)
