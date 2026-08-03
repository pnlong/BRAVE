"""Shared GAN / feature-matching helpers for canonicalizer and CycleGAN trainers."""

from __future__ import annotations

from typing import List

import torch

from ..core import mean_difference
from .in_domain_discriminator import InDomainAudioDiscriminator


def audio_gan_d(
    feat_real: List[List[torch.Tensor]],
    feat_fake: List[List[torch.Tensor]],
    gan_loss_fn,
) -> torch.Tensor:
    zero = torch.tensor(0.0, device=feat_real[0][-1].device)
    if not feat_real or not feat_fake:
        return zero
    loss_d, _ = InDomainAudioDiscriminator.gan_losses(
        feat_real, feat_fake, gan_loss_fn)
    return loss_d


def audio_gan_g(
    feat_fake: List[List[torch.Tensor]],
    gan_loss_fn,
) -> torch.Tensor:
    zero = torch.tensor(0.0, device=feat_fake[0][-1].device)
    if not feat_fake:
        return zero
    loss_g = torch.tensor(0.0, device=feat_fake[0][-1].device)
    for scale in feat_fake:
        _, g = gan_loss_fn(scale[-1].detach(), scale[-1])
        loss_g = loss_g + g
    return loss_g / max(len(feat_fake), 1)


def feature_matching_loss(
    feat_real: List[List[torch.Tensor]],
    feat_fake: List[List[torch.Tensor]],
    *,
    num_skipped_features: int = 1,
) -> torch.Tensor:
    zero = torch.tensor(0.0, device=feat_fake[0][-1].device)
    if not feat_fake:
        return zero
    loss_fm = zero
    n_scales = len(feat_real)
    for scale_real, scale_fake in zip(feat_real, feat_fake):
        real_layers = scale_real[num_skipped_features:]
        fake_layers = scale_fake[num_skipped_features:]
        if not real_layers:
            continue
        current = sum(
            mean_difference(r.detach(), f, norm="L1")
            for r, f in zip(real_layers, fake_layers)
        ) / len(real_layers)
        loss_fm = loss_fm + current
    return loss_fm / max(n_scales, 1)


def mean_fake_logit(feat_fake: List[List[torch.Tensor]]) -> torch.Tensor:
    return sum(scale[-1].mean() for scale in feat_fake) / max(len(feat_fake), 1)
