"""Attribute conditioning helpers for conditional canonicalizer training."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import gin
import torch
import torch.nn as nn


ConditionKey = Literal["attr_cls", "attr_norm"]


@gin.configurable
class AttributeConditioningEmbed(nn.Module):
    """
    Pool batch attributes into a fixed-size conditioning vector.

    Default ``attr_cls``: per-attribute embedding of time-pooled class/bin indices.
    Ablation ``attr_norm``: mean-pool normalized float controls per attribute row.
    """

    def __init__(
        self,
        num_attributes: int,
        num_classes_per_attribute: Sequence[int],
        embed_dim: int = 32,
        condition_on: ConditionKey = "attr_cls",
    ) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be > 0 for AttributeConditioningEmbed")
        if condition_on not in ("attr_cls", "attr_norm"):
            raise ValueError("condition_on must be attr_cls or attr_norm")

        self.num_attributes = num_attributes
        self.num_classes_per_attribute = list(num_classes_per_attribute)
        self.embed_dim = embed_dim
        self.condition_on = condition_on

        if len(self.num_classes_per_attribute) != num_attributes:
            raise ValueError(
                "num_classes_per_attribute length must match num_attributes")

        if condition_on == "attr_cls":
            self.embeddings = nn.ModuleList([
                nn.Embedding(n_cls, embed_dim)
                for n_cls in self.num_classes_per_attribute
            ])
            self.norm_proj = None
            self.out_dim = num_attributes * embed_dim
        else:
            self.embeddings = None
            self.norm_proj = nn.Linear(num_attributes, embed_dim)
            nn.init.zeros_(self.norm_proj.weight)
            nn.init.zeros_(self.norm_proj.bias)
            self.out_dim = embed_dim

    def forward(
        self,
        *,
        attr_cls: Optional[torch.Tensor] = None,
        attr_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_on == "attr_cls":
            if attr_cls is None:
                raise ValueError("attr_cls required when condition_on='attr_cls'")
            return self._embed_cls(attr_cls)
        if attr_norm is None:
            raise ValueError("attr_norm required when condition_on='attr_norm'")
        pooled = attr_norm.mean(dim=-1)
        return self.norm_proj(pooled)

    def _embed_cls(self, attr_cls: torch.Tensor) -> torch.Tensor:
        # attr_cls: (B, D, T_lat) integer bins / class indices
        parts = []
        for i, emb in enumerate(self.embeddings):
            n_cls = self.num_classes_per_attribute[i]
            idx = attr_cls[:, i, :].float().mean(dim=-1).round().long()
            idx = idx.clamp(0, n_cls - 1)
            parts.append(emb(idx))
        return torch.cat(parts, dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``gamma * x + beta`` with identity init."""

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[:feat_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return gamma * x + beta
