"""Shared canonicalizer attachment helpers for RAVE and FaderRAVE."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .latent_canonicalizer import LatentCanonicalizer, infer_latent_warp_hparams
from .waveform_canonicalizer import build_waveform_canonicalizer


def attach_canonicalizer_modules(
    model: nn.Module,
    state_dict: dict,
    canonicalizer_type: str,
) -> None:
    """Load warp weights onto a backbone with canonicalizer slots."""
    device = next(model.parameters()).device
    n_attr = backbone_num_attributes(model)
    cond_kwargs: dict = {}
    if n_attr > 0:
        cond_kwargs = {
            "num_attributes": n_attr,
            "num_classes_per_attribute": model.num_classes_per_attribute,
        }
    if canonicalizer_type == "waveform":
        n_channels = int(getattr(model, "n_channels", 1))
        warp = build_waveform_canonicalizer(
            sample_rate=model.sr,
            n_channels=n_channels,
            **cond_kwargs,
        )
        warp.load_state_dict(state_dict)
        model.waveform_canonicalizer = warp.to(device)
    elif canonicalizer_type == "latent":
        warp = LatentCanonicalizer(
            latent_size=model.latent_size,
            **infer_latent_warp_hparams(state_dict),
            **cond_kwargs,
        )
        warp.load_state_dict(state_dict)
        model.latent_canonicalizer = warp.to(device)
    else:
        raise ValueError(f"unknown canonicalizer_type: {canonicalizer_type}")


def backbone_num_attributes(model: nn.Module) -> int:
    return int(getattr(model, "num_attributes", 0))


def prepare_batch_attributes(
    model: nn.Module,
    attr_raw: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Raw attrs → (attr_norm, attr_cls) for decode + conditional D."""
    if backbone_num_attributes(model) == 0 or attr_raw is None:
        return None, None
    return model._prepare_attributes(attr_raw)


def prepare_decode_attributes(
    model: nn.Module,
    attr_raw: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return normalized attrs for decode, or None for unconditional."""
    attr_norm, _ = prepare_batch_attributes(model, attr_raw)
    return attr_norm


def primary_discrete_attr_index(model: nn.Module) -> Optional[int]:
    """First discrete attribute row index, or None for plain BRAVE."""
    kinds = getattr(model, "attribute_kinds", {})
    for i, name in enumerate(getattr(model, "attribute_names", [])):
        if kinds.get(name) == "discrete":
            return i
    return None


def discrete_class_ids_from_attr_raw(
    attr_raw: torch.Tensor,
    attr_index: int,
) -> torch.Tensor:
    """(B,) long class ids from the first latent frame of one attribute row."""
    return attr_raw[:, attr_index, 0].long()


def _sample_discrete_classes(
    n_samples: int,
    n_cls: int,
    *,
    device: torch.device,
    sampling: str,
    marginal_probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    if sampling == "marginal" and marginal_probs is not None:
        probs = marginal_probs
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs).to(device=device, dtype=torch.float32)
        else:
            probs = probs.to(device=device, dtype=torch.float32)
        if probs.numel() != n_cls:
            probs = torch.full((n_cls,), 1.0 / n_cls, device=device)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum()
        return torch.multinomial(probs, n_samples, replacement=True).float()
    return torch.randint(0, n_cls, (n_samples,), device=device).float()


def assign_ood_target_attrs(
    attr_raw: torch.Tensor,
    model: nn.Module,
    ood_mask: torch.Tensor,
    *,
    discrete_sampling: str = "uniform",
    marginal_probs: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]] = None,
) -> torch.Tensor:
    """
    OOD tap attribute policy: keep tap-extracted continuous rows; sample discrete
    class indices (uniform or Y marginal) for each discrete attribute row.
    """
    if not ood_mask.any() or backbone_num_attributes(model) == 0:
        return attr_raw

    out = attr_raw.clone()
    ood_rows = torch.nonzero(ood_mask, as_tuple=False).squeeze(1)
    kinds = getattr(model, "attribute_kinds", {})
    names = getattr(model, "attribute_names", [])
    marginals = marginal_probs or {}

    for i, name in enumerate(names):
        if kinds.get(name) != "discrete":
            continue
        n_cls = int(getattr(model, "discrete_num_classes", {}).get(name, 2))
        if n_cls <= 1:
            continue
        sampled = _sample_discrete_classes(
            ood_rows.numel(),
            n_cls,
            device=out.device,
            sampling=discrete_sampling,
            marginal_probs=marginals.get(name),
        )
        out[ood_rows, i, :] = sampled.unsqueeze(1)

    return out
