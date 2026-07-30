"""Tests for latent spread loss helpers."""

import torch

from rave.canonicalizer.losses import (
    latent_ood_in_var_ratio,
    latent_per_channel_variance,
    latent_var_floor_loss,
    latent_var_match_loss,
    resolve_latent_spread_loss,
)


def test_var_match_zero_for_matching_distributions():
    z = torch.randn(4, 8, 16)
    v_ref = latent_per_channel_variance(z)
    loss = latent_var_match_loss(z, v_ref)
    assert loss.item() < 1e-5


def test_var_match_positive_when_ood_collapsed():
    z_in = torch.randn(8, 16, 32)
    z_ood = torch.zeros(8, 16, 32)
    v_ref = latent_per_channel_variance(z_in)
    loss = latent_var_match_loss(z_ood, v_ref)
    assert loss.item() > 0.1


def test_var_floor_penalizes_collapsed_batch():
    z_in = torch.randn(8, 16, 32)
    z_ood = torch.zeros(4, 16, 32)
    v_ref = latent_per_channel_variance(z_in)
    assert latent_var_floor_loss(z_ood, v_ref).item() > 0.0
    z_spread = torch.randn(4, 16, 32)
    assert latent_var_floor_loss(z_spread, v_ref).item() == 0.0


def test_ood_in_var_ratio_near_one_for_same_scale():
    z = torch.randn(4, 8, 16)
    v_ref = latent_per_channel_variance(z)
    ratio = latent_ood_in_var_ratio(z, v_ref)
    assert abs(ratio.item() - 1.0) < 0.05


def test_resolve_latent_spread_loss_modes():
    assert resolve_latent_spread_loss("var_match") is latent_var_match_loss
    assert resolve_latent_spread_loss("var_floor") is latent_var_floor_loss


def test_spread_loss_helpers_finite():
    z_ood = torch.randn(2, 4, 8)
    v_ref = torch.ones(4)
    assert torch.isfinite(latent_var_match_loss(z_ood, v_ref))
    assert torch.isfinite(latent_var_floor_loss(z_ood, v_ref))
