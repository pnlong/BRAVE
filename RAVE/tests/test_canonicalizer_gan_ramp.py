"""Tests for canonicalizer GAN ramp schedule."""

from rave.canonicalizer.callbacks import compute_gan_ramp_factor


def test_gan_ramp_recon_only_phase():
    assert compute_gan_ramp_factor(0, delay=1000, ramp_duration=5000) == 0.0
    assert compute_gan_ramp_factor(999, delay=1000, ramp_duration=5000) == 0.0


def test_gan_ramp_linear_increase():
    assert compute_gan_ramp_factor(1000, delay=1000, ramp_duration=5000) == 0.0
    assert compute_gan_ramp_factor(3500, delay=1000, ramp_duration=5000) == 0.5
    assert compute_gan_ramp_factor(5999, delay=1000, ramp_duration=5000) == 0.9998


def test_gan_ramp_full_strength():
    assert compute_gan_ramp_factor(6000, delay=1000, ramp_duration=5000) == 1.0
    assert compute_gan_ramp_factor(10000, delay=1000, ramp_duration=5000) == 1.0


def test_gan_ramp_zero_duration_is_hard_gate():
    assert compute_gan_ramp_factor(1000, delay=1000, ramp_duration=0) == 1.0


def test_spread_ramp_follows_same_schedule():
    assert compute_gan_ramp_factor(0, delay=25000, ramp_duration=25000) == 0.0
    assert compute_gan_ramp_factor(37500, delay=25000, ramp_duration=25000) == 0.5
    assert compute_gan_ramp_factor(50000, delay=25000, ramp_duration=25000) == 1.0
