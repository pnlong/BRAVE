"""Parity tests for JIT-safe torch Fader descriptors."""

import numpy as np
import pytest
import torch

from rave.fader.attributes import compute_descriptor_matrix
from rave.fader.export.torch_descriptors import TorchDescriptorExtract


CONTINUOUS = ["rms", "flatness", "centroid", "roughness", "brightness"]
SR = 44100
T_LAT = 32


@pytest.fixture
def mono_audio():
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * rng.standard_normal(t.shape)
    ).astype(np.float32)
    return audio


def test_torch_descriptors_runs(mono_audio):
    x = torch.from_numpy(mono_audio[None, None, :])
    ext = TorchDescriptorExtract(CONTINUOUS, sr=SR)
    out = ext(x, T_LAT)
    assert out.shape == (1, len(CONTINUOUS), T_LAT)
    assert torch.isfinite(out).all()


def test_torch_descriptors_rms_correlates(mono_audio):
    x = torch.from_numpy(mono_audio[None, None, :])
    ext = TorchDescriptorExtract(["rms"], sr=SR)
    torch_rms = ext(x, T_LAT)[0, 0].detach().numpy()

    librosa_mat = compute_descriptor_matrix(
        mono_audio, sr=SR, descriptors=["rms"], latent_length=T_LAT)
    librosa_rms = librosa_mat[0]

    corr = np.corrcoef(torch_rms, librosa_rms)[0, 1]
    assert corr > 0.85, f"rms correlation too low: {corr}"


def test_torch_descriptors_spectral_correlates(mono_audio):
    x = torch.from_numpy(mono_audio[None, None, :])
    for name in ("centroid", "flatness"):
        ext = TorchDescriptorExtract([name], sr=SR)
        torch_row = ext(x, T_LAT)[0, 0].detach().numpy()
        librosa_row = compute_descriptor_matrix(
            mono_audio, sr=SR, descriptors=[name], latent_length=T_LAT)[0]
        corr = np.corrcoef(torch_row, librosa_row)[0, 1]
        assert corr > 0.5, f"{name} correlation too low: {corr}"


def test_scripted_extractor():
    ext = TorchDescriptorExtract(CONTINUOUS, sr=SR)
    scripted = torch.jit.script(ext)
    x = torch.randn(1, 1, 4096)
    out = scripted(x, 16)
    assert out.shape == (1, len(CONTINUOUS), 16)
