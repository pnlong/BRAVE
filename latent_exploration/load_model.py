"""Shared RAVE checkpoint loader for latent exploration scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import cached_conv as cc
import gin
import torch

import numpy as np

from paths import rave_root

_RAVE = rave_root()
if _RAVE.is_dir() and str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

import rave  # noqa: E402


def load_rave(model_path: str | Path, use_gpu: bool = False) -> rave.RAVE:
    """Load a RAVE/BRAVE checkpoint from a run directory or .ckpt file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    cc.use_cached_conv(False)
    torch.set_float32_matmul_precision("high")

    config_path = rave.core.search_for_config(str(model_path))
    if config_path is None:
        raise FileNotFoundError(f"config.gin not found near {model_path}")
    gin.parse_config_file(config_path)

    run = rave.core.search_for_run(str(model_path))
    if run is None:
        raise FileNotFoundError(f"checkpoint not found near {model_path}")

    model = rave.RAVE()
    model = model.load_from_checkpoint(run)
    model.eval()

    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    return model


def load_audio(
    path: str | Path,
    model: rave.RAVE,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load and preprocess audio for RAVE inference. Returns [C, T] on device."""
    import soundfile as sf
    import torchaudio

    if device is None:
        device = next(model.parameters()).device

    data, sr = sf.read(str(path), always_2d=True)
    x = torch.from_numpy(data.T).float()
    if sr != model.sr:
        x = torchaudio.functional.resample(x, sr, model.sr)
    if model.n_channels < x.shape[0]:
        x = x[: model.n_channels]
    elif model.n_channels > x.shape[0]:
        raise ValueError(
            f"file has {x.shape[0]} channels but model expects {model.n_channels}"
        )
    return x.to(device)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save [C, T] waveform to WAV (avoids torchaudio 2.9+ torchcodec requirement)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy().T, sample_rate)


def compression_ratio(model: rave.RAVE) -> int:
    return rave.core.get_minimum_size(model)


def has_pca(model: rave.RAVE) -> bool:
    """True if checkpoint has a fitted variational latent PCA."""
    if type(model.encoder).__name__ != "VariationalEncoder":
        return False
    return float(model.fidelity[-1]) > 0.01


def to_pca_latent(model: rave.RAVE, z: torch.Tensor) -> torch.Tensor:
    """Rotate [B, L, T] VAE latents into the checkpoint PCA basis."""
    import torch.nn.functional as F

    if not has_pca(model):
        raise RuntimeError("checkpoint has no fitted latent PCA (variational validation pass)")
    z = z - model.latent_mean.unsqueeze(-1)
    return F.conv1d(z, model.latent_pca.unsqueeze(-1))


def from_pca_latent(model: rave.RAVE, z_pca: torch.Tensor) -> torch.Tensor:
    """Inverse PCA: [B, L, T] PCA coefficients → VAE latents for decode."""
    import torch.nn.functional as F

    if not has_pca(model):
        raise RuntimeError("checkpoint has no fitted latent PCA (variational validation pass)")
    z = F.conv1d(z_pca, model.latent_pca.T.unsqueeze(-1))
    return z + model.latent_mean.unsqueeze(-1)


def apply_latent_mask(
    model: rave.RAVE,
    z: torch.Tensor,
    mask: torch.Tensor,
    *,
    mask_space: str = "vae",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply mask and return VAE latents for decode plus optional masked PCA view.

    - vae: mask in encoder latent space (default)
    - pca: rotate → mask → inverse rotate (``latent`` style = PCA components)
    """
    if mask_space == "vae":
        z_out = z * mask
        z_pca_masked = to_pca_latent(model, z_out) if has_pca(model) else None
        return z_out, z_pca_masked

    if mask_space == "pca":
        if not has_pca(model):
            raise RuntimeError("--mask-space pca requires a fitted PCA in the checkpoint")
        z_pca = to_pca_latent(model, z)
        z_pca_masked = z_pca * mask
        z_out = from_pca_latent(model, z_pca_masked)
        return z_out, z_pca_masked

    raise ValueError(f"unknown mask_space {mask_space!r}; choose 'vae' or 'pca'")


def pca_display_dims(model: rave.RAVE, fidelity: float = 0.95) -> int:
    """
    Number of leading PCA components that explain at least ``fidelity`` variance.

    Uses the cumulative explained-variance buffer RAVE stores at validation time
    (same criterion as ``scripts/export.py --fidelity``).
    """
    if not has_pca(model):
        raise RuntimeError("checkpoint has no fitted latent PCA")
    var = model.fidelity.detach().cpu().numpy()
    n = int(np.argmax(var > fidelity)) + 1
    return max(n, 1)
