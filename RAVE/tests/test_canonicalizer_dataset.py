"""Tests for OOD WAV sidecar dataset."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rave.fader.canonicalizer_dataset import DOMAIN_OOD, OodWavDataset


@pytest.fixture
def ood_dir():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "clip.wav"
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(p), audio, sr)
        yield td


def test_ood_wav_dataset_loads(ood_dir):
    ds = OodWavDataset(
        ood_path=ood_dir,
        sampling_rate=44100,
        n_signal=4096,
        continuous_attributes=["rms", "centroid"],
        discrete_attributes=[],
        latent_length=32,
    )
    audio, attr, domain = ds[0]
    assert domain == DOMAIN_OOD
    assert audio.shape[-1] == 4096
    assert attr.shape[0] == 2
    assert attr.shape[1] == 32
