import os
from pathlib import Path

import gin
import numpy as np

import rave  # noqa: F401 — gin search path
from rave.transforms import RandomGain, RandomTargetLoudness

_BRAVE = Path(__file__).resolve().parents[2]


def test_random_gain_applies_constant_db():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 1024).astype(np.float32) * 0.1
    aug = RandomGain(gain_range=(6, 6), prob=1.0, limit=False)
    y = aug(x)
    np.testing.assert_allclose(y, x * (10 ** (6 / 20)), rtol=1e-5)


def test_random_gain_limit_prevents_clip():
    x = np.ones((1, 64), dtype=np.float32) * 0.9
    aug = RandomGain(gain_range=(12, 12), prob=1.0, limit=True)
    y = aug(x)
    assert float(np.abs(y).max()) <= 1.0 + 1e-6


def _rms_db(x, eps=1e-8):
    rms = float(np.sqrt(np.mean(np.square(x)) + eps))
    return 20.0 * np.log10(max(rms, eps))


def test_random_target_loudness_hits_target():
    rng = np.random.RandomState(0)
    # Naturally "loud" crop (~−6 dBFS RMS-ish after scale)
    x = rng.randn(1, 4096).astype(np.float32)
    x = x / (np.abs(x).max() + 1e-8) * 0.9
    assert _rms_db(x) > -20.0
    aug = RandomTargetLoudness(
        loudness_range=(-24, -24),
        prob=1.0,
        hard_clip=False,
        max_gain_db=60.0,
    )
    y = aug(x)
    np.testing.assert_allclose(_rms_db(y), -24.0, atol=0.25)


def test_random_target_loudness_maps_quiet_and_loud_together():
    rng = np.random.RandomState(1)
    loud = rng.randn(1, 4096).astype(np.float32)
    loud = loud / (np.abs(loud).max() + 1e-8) * 0.95
    quiet = loud * 0.05
    assert _rms_db(loud) - _rms_db(quiet) > 20.0
    aug = RandomTargetLoudness(
        loudness_range=(-30, -30),
        prob=1.0,
        hard_clip=False,
        max_gain_db=60.0,
    )
    y_loud = aug(loud)
    y_quiet = aug(quiet)
    np.testing.assert_allclose(_rms_db(y_loud), -30.0, atol=0.25)
    np.testing.assert_allclose(_rms_db(y_quiet), -30.0, atol=0.25)


def test_random_target_loudness_applies_to_near_silence():
    # Tiny non-zero crop: still boosted (capped by max_gain_db), not skipped.
    x = np.full((1, 256), 1e-5, dtype=np.float32)
    aug = RandomTargetLoudness(
        loudness_range=(-20, -20),
        prob=1.0,
        hard_clip=False,
        max_gain_db=40.0,
    )
    y = aug(x)
    assert float(np.abs(y).mean()) > float(np.abs(x).mean())
    # Exact target may be unreachable under max_gain_db; level must rise.
    assert _rms_db(y) > _rms_db(x) + 10.0


def test_random_target_loudness_hard_clips_overdrive():
    x = np.ones((1, 64), dtype=np.float32) * 0.5
    aug = RandomTargetLoudness(
        loudness_range=(0, 0),  # force large boost → would exceed ±1
        prob=1.0,
        hard_clip=True,
        max_gain_db=60.0,
    )
    y = aug(x)
    assert float(np.abs(y).max()) <= 1.0 + 1e-6
    # Saturation: many samples at the rail (not a rescaled soft peak).
    assert float(np.mean(np.abs(y) >= 1.0 - 1e-5)) > 0.5


def test_brave_gin_binds_target_loudness():
    gin.clear_config()
    prev = os.getcwd()
    os.chdir(_BRAVE / "configs")
    try:
        gin.parse_config_file("brave.gin")
    finally:
        os.chdir(prev)
    cfg = gin.config_str()
    assert "RandomGain" in cfg
    assert "dataset.get_dataset.augmentations" in cfg
    assert "(-12, 6)" in cfg or "(-12,6)" in cfg
    assert "RandomTargetLoudness" not in cfg
    assert "silence_db" not in cfg


def test_brave_tap_gin_inherits_target_loudness():
    gin.clear_config()
    prev = os.getcwd()
    os.chdir(_BRAVE / "configs")
    try:
        gin.parse_config_file("brave_tap.gin")
    finally:
        os.chdir(prev)
    cfg = gin.config_str()
    assert "RandomGain" in cfg
    assert "(-12, 6)" in cfg or "(-12,6)" in cfg
    assert "dataset.get_dataset.augmentations" in cfg
