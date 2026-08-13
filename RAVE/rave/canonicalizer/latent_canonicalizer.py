"""Latent-domain input canonicalizer: 128→128 warp after encode."""

from __future__ import annotations

from typing import Optional, Sequence

import gin
import torch
import torch.nn as nn

from .attribute_conditioning import AttributeConditioningEmbed, FiLM


def infer_latent_warp_hparams(state_dict: dict) -> dict:
    """Recover ``n_layers`` / ``hidden_size`` from a warp state dict."""
    has_conv2 = any(
        key == "conv2.weight" or key.endswith(".conv2.weight")
        for key in state_dict
    )
    if not has_conv2:
        return {"n_layers": 1}
    weight = state_dict.get("conv1.weight")
    if weight is None:
        for key, value in state_dict.items():
            if key.endswith("conv1.weight"):
                weight = value
                break
    if weight is None:
        return {"n_layers": 2}
    return {"n_layers": 2, "hidden_size": int(weight.shape[0])}


@gin.configurable
class LatentCanonicalizer(nn.Module):
    """
    L(z) on content latent (B, latent_size, T_lat).

    ``init_mode="identity"`` (Stage-1 default): gated residual, identity at init.
    ``init_mode="random"`` (CycleGAN): no residual gate; variance-preserving random
    weights (orthogonal 1×1, or Kaiming MLP).

    ``n_layers=1`` (default): 1×1 conv.
    ``n_layers=2``: Conv1d → LeakyReLU → Conv1d.

    Optional ``attr_cls`` FiLM modulates the pre-residual warp when ``num_attributes > 0``.
    """

    def __init__(
        self,
        latent_size: int = 128,
        n_layers: int = 1,
        hidden_size: Optional[int] = None,
        negative_slope: float = 0.2,
        num_attributes: int = 0,
        num_classes_per_attribute: Optional[Sequence[int]] = None,
        embed_dim: int = 32,
        init_mode: str = "identity",
    ) -> None:
        super().__init__()
        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
        if init_mode not in ("identity", "random"):
            raise ValueError("init_mode must be 'identity' or 'random'")
        self.latent_size = latent_size
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size) if hidden_size else latent_size
        self.num_attributes = int(num_attributes)
        self.init_mode = init_mode
        self.negative_slope = float(negative_slope)
        self.alpha = nn.Parameter(torch.zeros(1))

        if self.n_layers == 1:
            self.conv = nn.Conv1d(latent_size, latent_size, kernel_size=1, bias=True)
            self.conv1 = None
            self.conv2 = None
            self.act = None
        else:
            self.conv = None
            self.conv1 = nn.Conv1d(
                latent_size, self.hidden_size, kernel_size=1, bias=True)
            self.act = nn.LeakyReLU(negative_slope)
            self.conv2 = nn.Conv1d(
                self.hidden_size, latent_size, kernel_size=1, bias=True)

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

        if self.init_mode == "identity":
            self._init_identity()
        else:
            self._init_random()

    def _init_identity(self) -> None:
        if self.n_layers == 1:
            assert self.conv is not None
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
            with torch.no_grad():
                for i in range(self.conv.in_channels):
                    self.conv.weight[i, i, 0] = 1.0
            return
        assert self.conv2 is not None
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _init_random(self) -> None:
        if self.n_layers == 1:
            assert self.conv is not None
            with torch.no_grad():
                nn.init.orthogonal_(self.conv.weight[:, :, 0])
            nn.init.zeros_(self.conv.bias)
            return
        assert self.conv1 is not None and self.conv2 is not None
        nn.init.kaiming_uniform_(self.conv1.weight, a=self.negative_slope)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, a=self.negative_slope)
        nn.init.zeros_(self.conv2.bias)

    def _warp(self, z: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 1:
            assert self.conv is not None
            return self.conv(z)
        assert self.conv1 is not None and self.conv2 is not None and self.act is not None
        h = self.conv2(self.act(self.conv1(z)))
        if self.init_mode == "identity":
            return z + h
        return h

    def forward(
        self,
        z: torch.Tensor,
        attr_cls: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        warped = self._warp(z)
        if self.film is not None and self.cond_embed is not None and attr_cls is not None:
            cond = self.cond_embed(attr_cls=attr_cls)
            warped = self.film(warped, cond)
        if self.init_mode == "random":
            return warped
        alpha = torch.sigmoid(self.alpha)
        return z + alpha * (warped - z)
