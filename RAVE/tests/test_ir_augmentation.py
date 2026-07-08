"""Tests for room IR augmentation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rave.canonicalizer.ir_augmentation import ImpulseResponseAug, synthetic_room_ir


@pytest.fixture
def ir_dir():
    with tempfile.TemporaryDirectory() as td:
        sr = 44100
        ir = synthetic_room_ir(sr, decay_sec=0.2)
        sf.write(str(Path(td) / "room_a.wav"), ir, sr)
        sf.write(str(Path(td) / "room_b.wav"), ir * 0.8, sr)
        yield td


def test_synthetic_ir_convolve_changes_audio():
    sr = 44100
    t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
    x = np.stack([(0.2 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)])
    ir = synthetic_room_ir(sr)
    aug = ImpulseResponseAug(sampling_rate=sr, prob=1.0, ir_path=None)
    y = aug.convolve(x, ir, wet=0.4)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_ir_augment_from_directory(ir_dir):
    sr = 44100
    t = np.linspace(0, 0.25, sr // 4, endpoint=False, dtype=np.float32)
    x = np.stack([(0.3 * np.sign(np.sin(2 * np.pi * 200 * t))).astype(np.float32)])
    aug = ImpulseResponseAug(ir_path=ir_dir, sampling_rate=sr, prob=1.0)
    y = aug.maybe_apply(x)
    assert y.shape == x.shape
    assert not np.allclose(y, x)


def test_ir_augment_prob_zero_is_noop(ir_dir):
    sr = 44100
    x = np.random.randn(1, 4096).astype(np.float32)
    aug = ImpulseResponseAug(ir_path=ir_dir, sampling_rate=sr, prob=0.0)
    assert np.allclose(aug.maybe_apply(x), x)
