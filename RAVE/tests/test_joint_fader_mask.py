"""Tests for joint Fader availability masks and merged stats."""

import numpy as np
import torch

from rave.fader.attributes import (
    attribute_classification_loss,
    merge_attribute_stats,
)


def test_attribute_ce_masked_skips_missing():
    # One head available, one missing → only head 0 contributes.
    pred0 = torch.zeros(2, 3, 4)
    pred0[:, 0, :] = 10.0  # confident class 0
    pred1 = torch.zeros(2, 3, 4)  # uniform / wrong for class 1
    target = torch.zeros(2, 2, 4, dtype=torch.long)
    target[:, 1, :] = 1
    mask = torch.ones(2, 2, 4)
    mask[:, 1, :] = 0.0  # skip head 1
    loss = attribute_classification_loss([pred0, pred1], target, avail_mask=mask)
    loss_head0 = attribute_classification_loss(
        [pred0], target[:, :1], avail_mask=mask[:, :1])
    assert torch.isfinite(loss)
    assert abs(float(loss) - float(loss_head0)) < 1e-5


def test_attribute_ce_all_masked_is_zero():
    pred = [torch.randn(2, 3, 4)]
    target = torch.zeros(2, 1, 4, dtype=torch.long)
    mask = torch.zeros(2, 1, 4)
    loss = attribute_classification_loss(pred, target, avail_mask=mask)
    assert float(loss) == 0.0


def test_merge_attribute_stats_pools_minmax():
    sx = {
        "version": 1,
        "attribute_names": ["rms", "water_scene"],
        "continuous_attributes": ["rms"],
        "discrete_attributes": ["water_scene"],
        "attribute_kinds": {"rms": "continuous", "water_scene": "discrete"},
        "nb_bins": 4,
        "latent_length": 8,
        "sr": 44100,
        "min_max_features": {"rms": (0.0, 1.0)},
        "bin_values": np.zeros((2, 4), dtype=np.float32),
        "discrete_num_classes": {"water_scene": 3},
    }
    sy = {
        "version": 1,
        "attribute_names": ["rms", "water_scene"],
        "continuous_attributes": ["rms"],
        "discrete_attributes": ["water_scene"],
        "attribute_kinds": {"rms": "continuous", "water_scene": "discrete"},
        "nb_bins": 4,
        "latent_length": 8,
        "sr": 44100,
        "min_max_features": {"rms": (0.5, 2.0)},
        "bin_values": np.zeros((2, 4), dtype=np.float32),
        "discrete_num_classes": {"water_scene": 3},
    }
    m = merge_attribute_stats(sx, sy)
    assert m["min_max_features"]["rms"] == (0.0, 2.0)
    assert m["attribute_names"] == ["rms", "water_scene"]
    assert m["bin_values"].shape == (2, 4)
