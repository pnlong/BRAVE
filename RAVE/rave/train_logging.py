"""Shared training metrics for W&B / PyTorch Lightning (explicit train/ and val/ keys)."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple, Union

import torch

TRAIN_PREFIX = "train/"
VAL_PREFIX = "val/"

_RECON_PREFIXES = ("multiband_", "fullband_")

# Explicit flags so W&B receives time series under manual optimization and when
# len(train) < Trainer.log_every_n_steps (common for small LMDBs + large batch).
_LOG_KWARGS = dict(on_step=True, on_epoch=True, logger=True, sync_dist=False)
_VAL_KWARGS = dict(on_step=False, on_epoch=True, logger=True, sync_dist=False)


def _merged_opts(defaults: dict, overrides: dict) -> dict:
    return {**defaults, **overrides}


def train_key(name: str) -> str:
    if name.startswith(TRAIN_PREFIX) or name.startswith(VAL_PREFIX):
        return name
    return f"{TRAIN_PREFIX}{name}"


def val_key(name: str) -> str:
    if name.startswith(VAL_PREFIX) or name.startswith(TRAIN_PREFIX):
        return name
    return f"{VAL_PREFIX}{name}"


def log_train(module, name: str, value, **kwargs) -> None:
    module.log(train_key(name), value, **_merged_opts(_LOG_KWARGS, kwargs))


def log_val(module, name: str, value, **kwargs) -> None:
    module.log(val_key(name), value, **_merged_opts(_VAL_KWARGS, kwargs))


def log_train_dict(
    module,
    metrics: Mapping[str, torch.Tensor],
    **kwargs,
) -> None:
    module.log_dict(
        {train_key(k): v for k, v in metrics.items()},
        **_merged_opts(_LOG_KWARGS, kwargs),
    )


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
    """Log headline generator losses on generator optimizer steps only."""
    if not is_gen_step:
        return
    loss_total, loss_recon, loss_latent = aggregate_generator_loss(
        loss_gen, module.weights)
    log_train(module, "loss", loss_total)
    log_train(module, "loss_recon", loss_recon)
    log_train(module, "loss_latent", loss_latent)
