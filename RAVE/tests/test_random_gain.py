import os
from pathlib import Path

import gin
import numpy as np

import rave  # noqa: F401 — gin search path
from rave.transforms import RandomGain

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


def test_brave_tap_gin_binds_random_gain():
    gin.clear_config()
    prev = os.getcwd()
    os.chdir(_BRAVE / "configs")
    try:
        gin.parse_config_file("brave_tap.gin")
    finally:
        os.chdir(prev)
    cfg = gin.config_str()
    assert "RandomGain" in cfg
    assert "dataset.get_dataset.augmentations" in cfg

