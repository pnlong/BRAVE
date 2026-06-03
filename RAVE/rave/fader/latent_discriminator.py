"""
Latent discriminator: attribute classifier on VAE latent z.

Purpose
-------
Predict discretized attribute classes from **content** latent z alone.
Used in phase 1 only:
  - lat_dis_step: +CE (train classifier)
  - generator: -CE (push attribute info out of z)

Ported from neurorave latent_discriminator.py; uses causal cached_conv to match
BRAVE brave.gin (cc.get_padding.mode = 'causal').
"""

from typing import List, Sequence, Union

import gin
import cached_conv as cc
import torch
import torch.nn as nn


@gin.configurable
class LatentDiscriminator(nn.Module):
    """
    Predicts discretized attribute classes from latent z.

    One Conv1d head per attribute; each head may have a different num_classes.
    Forward returns list of (B, C_i, T_lat) logits per attribute.
    """

    def __init__(
        self,
        latent_size: int = 128,
        num_attributes: int = 1,
        num_classes: int = 16,
        num_classes_per_attribute: Union[Sequence[int], None] = None,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.num_attributes = num_attributes
        # --- Per-head class counts: nb_bins for continuous, num_classes for discrete ---
        if num_classes_per_attribute is not None:
            self.num_classes_per_attribute = list(num_classes_per_attribute)
        else:
            self.num_classes_per_attribute = [num_classes] * num_attributes
        self.num_classes = max(self.num_classes_per_attribute) if self.num_classes_per_attribute else num_classes

        # --- Shared trunk on z (B, latent_size, T) ---
        net = []
        for _ in range(num_layers):
            net.append(
                cc.Conv1d(
                    latent_size,
                    latent_size,
                    7,
                    stride=1,
                    padding=cc.get_padding(7),
                    bias=False,
                ))
            net.append(nn.BatchNorm1d(latent_size))
            net.append(nn.LeakyReLU(0.2))

        net.append(
            cc.Conv1d(
                latent_size,
                latent_size // 2,
                7,
                stride=1,
                padding=cc.get_padding(7),
                bias=False,
            ))
        net.append(nn.BatchNorm1d(latent_size // 2))
        net.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*net)

        # --- Per-attribute classification heads (variable C per head) ---
        attr_nets = []
        for n_cls in self.num_classes_per_attribute:
            head = nn.Sequential(
                cc.Conv1d(
                    latent_size // 2,
                    latent_size // 4,
                    7,
                    stride=1,
                    padding=cc.get_padding(7),
                    bias=False,
                ),
                nn.BatchNorm1d(latent_size // 4),
                nn.LeakyReLU(0.2),
                cc.Conv1d(
                    latent_size // 4,
                    n_cls,
                    7,
                    stride=1,
                    padding=cc.get_padding(7),
                    bias=False,
                ),
            )
            attr_nets.append(head)
        self.attr_nets = nn.ModuleList(attr_nets)

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            z: (B, latent_size, T_lat) content latent (not cat with attr)

        Returns:
            List of (B, C_i, T_lat) logits, one per attribute head
        """
        # --- Shared trunk → per-attribute conv heads ---
        x = self.net(z)
        return [layer(x) for layer in self.attr_nets]

    def forward_stacked(self, z: torch.Tensor) -> torch.Tensor:
        """Legacy stacked output when all heads share num_classes."""
        outs = self.forward(z)
        if not outs:
            raise RuntimeError("LatentDiscriminator has no attribute heads")
        c0 = outs[0].shape[1]
        if all(o.shape[1] == c0 for o in outs):
            return torch.cat([t.unsqueeze(dim=-2) for t in outs], dim=-2)
        raise ValueError(
            "forward_stacked requires equal num_classes per head; use forward()")
