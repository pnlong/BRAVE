"""Shared training metrics for W&B / PyTorch Lightning (train/* and val/* prefixes)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

_RECON_PREFIXES = ("multiband_", "fullband_")


def aggregate_generator_loss(
    loss_gen: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match generator ``backward()``: sum_k loss_gen[k] * weights.get(k, 1.).

    ``loss_recon`` — distance terms only (multiband_* / fullband_*).
    ``loss_latent`` — Fader phase-1 ``latent_adversarial`` term (else 0).
    """
    device = next(iter(loss_gen.values())).device
    dtype = next(iter(loss_gen.values())).dtype
    loss_total = torch.tensor(0.0, device=device, dtype=dtype)
    loss_recon = torch.tensor(0.0, device=device, dtype=dtype)
    loss_latent = torch.tensor(0.0, device=device, dtype=dtype)
    for k, v in loss_gen.items():
        term = v * weights.get(k, 1.0)
        loss_total = loss_total + term
        if k.startswith(_RECON_PREFIXES):
            loss_recon = loss_recon + term
        elif k == "latent_adversarial":
            loss_latent = loss_latent + term
    return loss_total, loss_recon, loss_latent


def log_generator_losses(module, loss_gen: Dict[str, torch.Tensor], is_gen_step: bool) -> None:
    """Log ``loss``, ``loss_recon``, ``loss_latent`` on generator optimizer steps only."""
    if not is_gen_step:
        return
    loss_total, loss_recon, loss_latent = aggregate_generator_loss(
        loss_gen, module.weights)
    module.log("loss", loss_total)
    module.log("loss_recon", loss_recon)
    module.log("loss_latent", loss_latent)
