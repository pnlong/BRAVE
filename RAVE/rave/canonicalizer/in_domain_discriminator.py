"""Audio discriminator: real in-domain vs OOD-translated reconstructions."""

from __future__ import annotations

from typing import Callable, List

import gin
import torch
import torch.nn as nn


@gin.configurable
class InDomainAudioDiscriminator(nn.Module):
    """
    Multi-scale audio discriminator for one-way OOD → in-domain transfer.

    Trained to classify:
      - **real**: reconstructions from in-domain (Y) batches
      - **fake**: reconstructions from out-of-domain (X) batches after the warp

    Wraps the same ``MultiScaleDiscriminator`` stack used in RAVE training.
    ``forward`` returns per-scale feature lists; final maps are GAN logits.
    """

    def __init__(
        self,
        discriminator: Callable[..., nn.Module],
        n_channels: int = 1,
    ) -> None:
        super().__init__()
        self.net = discriminator(n_channels=n_channels)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        return self.net(x)

    @staticmethod
    def gan_losses(
        features_real: List[List[torch.Tensor]],
        features_fake: List[List[torch.Tensor]],
        gan_loss_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate multi-scale GAN loss (D step, G step)."""
        loss_d = torch.tensor(0.0, device=features_real[0][-1].device)
        loss_g = torch.tensor(0.0, device=features_real[0][-1].device)
        n_scales = len(features_real)
        for scale_real, scale_fake in zip(features_real, features_fake):
            score_real = scale_real[-1]
            score_fake = scale_fake[-1]
            if score_real.shape[0] != score_fake.shape[0]:
                # Mixed batches often have unequal in-domain vs OOD counts.
                loss_dis = (
                    torch.relu(1 - score_real).mean()
                    + torch.relu(1 + score_fake).mean()
                )
                loss_gen = -score_fake.mean()
            else:
                loss_dis, loss_gen = gan_loss_fn(score_real, score_fake)
            loss_d = loss_d + loss_dis
            loss_g = loss_g + loss_gen
        return loss_d / n_scales, loss_g / n_scales


_MSD_SCOPE = "discriminator.MultiScaleDiscriminator"


def require_gin_binding(param: str) -> None:
    """Raise if ``param`` is not bound (canonicalizer gin was not parsed)."""
    try:
        gin.query_parameter(param)
    except (ValueError, KeyError) as exc:
        preview = gin.config_str()
        raise RuntimeError(
            f"Missing required gin binding {param!r}. "
            "Parse configs/brave_canonicalizer.gin from the configs/ directory "
            f"before building the in-domain discriminator "
            f"(config_str length={len(preview)}).\n"
            f"Preview:\n{preview[:2000]}"
        ) from exc


def build_in_domain_discriminator(n_channels: int) -> InDomainAudioDiscriminator:
    """Construct MSD from gin bindings in ``brave_canonicalizer.gin``."""
    require_gin_binding(f"{_MSD_SCOPE}.n_discriminators")
    msd = gin.get_configurable(_MSD_SCOPE)(n_channels=n_channels)
    return InDomainAudioDiscriminator(
        discriminator=lambda **_kwargs: msd,
        n_channels=n_channels,
    )
