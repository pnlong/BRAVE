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
    ext = TorchDescriptorExtract(["rms", "flatness", "centroid"], sr=SR)
    out = ext(x, T_LAT)
    assert out.shape == (1, 3, T_LAT)
    assert torch.isfinite(out).all()


def test_torch_descriptors_legacy_five_names(mono_audio):
    """Non-librosa names are omitted from torch output rows."""
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


def test_torch_descriptors_flatness_correlates(mono_audio):
    x = torch.from_numpy(mono_audio[None, None, :])
    ext = TorchDescriptorExtract(["flatness"], sr=SR)
    torch_row = ext(x, T_LAT)[0, 0].detach().numpy()
    librosa_row = compute_descriptor_matrix(
        mono_audio, sr=SR, descriptors=["flatness"], latent_length=T_LAT)[0]
    corr = np.corrcoef(torch_row, librosa_row)[0, 1]
    assert corr > 0.5, f"flatness correlation too low: {corr}"


def test_torch_descriptors_centroid_matches_librosa(mono_audio):
    x = torch.from_numpy(mono_audio[None, None, :])
    ext = TorchDescriptorExtract(["centroid"], sr=SR)
    torch_cent = float(ext(x, T_LAT)[0, 0].mean())

    import librosa
    S = np.abs(librosa.stft(mono_audio, n_fft=2048, hop_length=512))
    lib_cent = float(librosa.feature.spectral_centroid(S=S, sr=SR)[0].mean())

    assert abs(torch_cent - lib_cent) / max(lib_cent, 1.0) < 0.05


def test_scripted_extractor():
    ext = TorchDescriptorExtract(CONTINUOUS, sr=SR)
    scripted = torch.jit.script(ext)
    x = torch.randn(1, 1, 4096)
    out = scripted(x, 16)
    assert out.shape == (1, len(CONTINUOUS), 16)


def test_torch_descriptors_streaming_blocks():
    """Rolling history: later blocks use more audio context than the first."""
    ext = TorchDescriptorExtract(CONTINUOUS, sr=SR, max_history=8192)
    first = ext(torch.randn(1, 1, 512), 8)
    for _ in range(15):
        ext(torch.randn(1, 1, 512), 8)
    warmed = ext(torch.randn(1, 1, 512), 8)
    assert first.shape == warmed.shape == (1, len(CONTINUOUS), 8)
    assert torch.isfinite(warmed).all()
    scripted = torch.jit.script(ext)
    out = scripted(torch.randn(1, 1, 512), 8)
    assert torch.isfinite(out).all()
