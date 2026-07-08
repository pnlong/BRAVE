"""Shared RAVE / FaderRAVE checkpoint loader for latent exploration scripts.

Handles:
  - Gin config parse + checkpoint restore
  - FaderRAVE auto-detection from config.gin
  - attribute_stats.yaml loading (required for Fader)
  - Helper APIs: extract_normalized_attributes, build_constant_attr
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cached_conv as cc
import gin
import torch

import numpy as np

from paths import rave_root

_RAVE = rave_root()
if _RAVE.is_dir() and str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

import rave  # noqa: E402


@dataclass
class FaderBundle:
    """Loaded FaderRAVE plus optional canonicalizer warps and domain metadata."""

    model: Any
    domain_profile: Optional[Any] = None
    waveform_canonicalizer: Optional[torch.nn.Module] = None
    latent_canonicalizer: Optional[torch.nn.Module] = None


def gin_operative_config_str_from_file(config_path: Path) -> str:
    """Parse gin file and return operative config string."""
    gin.clear_config()
    gin.parse_config_file(str(config_path))
    return gin.operative_config_str()


def is_fader_checkpoint(config_path: str | Path) -> bool:
    """True if gin config configures FaderRAVE."""
    config_path = Path(config_path)
    text = config_path.read_text()
    if "rave.fader.model.FaderRAVE" in text:
        return True
    return "FaderRAVE" in gin_operative_config_str_from_file(config_path)


def find_stats_path(
    model_path: str | Path,
    stats_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Search attribute_stats.yaml near explicit path, checkpoint, or db."""
    from rave.fader.attributes import resolve_stats_path

    if stats_path:
        p = Path(stats_path)
        if p.is_file():
            return p
    model_path = Path(model_path)
    search_dirs = []
    if model_path.is_file():
        search_dirs.append(model_path.parent)
    else:
        search_dirs.append(model_path)
    for d in search_dirs:
        candidate = d / "attribute_stats.yaml"
        if candidate.is_file():
            return candidate
    if db_path:
        resolved = resolve_stats_path(str(db_path))
        if resolved:
            return resolved
    return None


def load_model(
    model_path: str | Path,
    use_gpu: bool = False,
    stats_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
) -> Union[rave.RAVE, Any]:
    """Load RAVE or FaderRAVE from a run directory or .ckpt file."""
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

    use_fader = "rave.fader.model.FaderRAVE" in gin.operative_config_str()
    # --- Instantiate correct model class before load_from_checkpoint ---
    if use_fader:
        from rave.fader.model import FaderRAVE
        model = FaderRAVE()
    else:
        model = rave.RAVE()
    model = model.load_from_checkpoint(run)

    # --- Load attribute stats for FaderRAVE ---
    if use_fader:
        sp = find_stats_path(model_path, stats_path, db_path)
        if sp is None:
            raise FileNotFoundError(
                "FaderRAVE requires attribute_stats.yaml; pass --stats-path or --db-path"
            )
        model.load_attribute_stats_from_file(sp)

    model.eval()

    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    return model


def _attach_canonicalizer_warp(
    model,
    state_dict: dict,
    canonicalizer_type: str,
) -> None:
    from rave.canonicalizer.config import attach_canonicalizer_to_model

    attach_canonicalizer_to_model(model, state_dict, canonicalizer_type)


def load_fader_with_canonicalizer(
    model_path: str | Path,
    *,
    config_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
    stats_path: Optional[str | Path] = None,
    waveform_canonicalizer_ckpt: Optional[str | Path] = None,
    latent_canonicalizer_ckpt: Optional[str | Path] = None,
    use_gpu: bool = False,
    validate_manifest: bool = True,
) -> FaderBundle:
    """
    Load FaderRAVE and optionally attach waveform or latent canonicalizer weights.
    """
    if waveform_canonicalizer_ckpt and latent_canonicalizer_ckpt:
        raise ValueError("Pass at most one of waveform_canonicalizer_ckpt, latent_canonicalizer_ckpt")

    from rave.canonicalizer.config import (
        build_domain_profile,
        load_canonicalizer_checkpoint,
        validate_manifest as _validate_manifest,
    )

    model_path = Path(model_path)
    if config_path is None:
        config_path = rave.core.search_for_config(str(model_path))
    if config_path is None:
        raise FileNotFoundError(f"config not found near {model_path}")

    profile = None
    if db_path is not None:
        profile = build_domain_profile(config_path, db_path, stats_path=stats_path)

    model = load_model(
        model_path,
        use_gpu=use_gpu,
        stats_path=stats_path,
        db_path=db_path,
    )
    if not is_fader_model(model):
        raise TypeError("load_fader_with_canonicalizer requires FaderRAVE checkpoint")

    wf_ckpt = Path(waveform_canonicalizer_ckpt) if waveform_canonicalizer_ckpt else None
    lat_ckpt = Path(latent_canonicalizer_ckpt) if latent_canonicalizer_ckpt else None
    ckpt_path = wf_ckpt or lat_ckpt

    wf_mod = None
    lat_mod = None
    if ckpt_path is not None:
        state, manifest = load_canonicalizer_checkpoint(ckpt_path)
        if validate_manifest and profile is not None:
            _validate_manifest(
                manifest,
                config_path=config_path,
                ckpt_path=model_path,
                db_path=db_path,
                strict=False,
            )
        _attach_canonicalizer_warp(model, state, manifest.canonicalizer_type)
        if manifest.canonicalizer_type == "waveform":
            wf_mod = model.waveform_canonicalizer
        else:
            lat_mod = model.latent_canonicalizer

    return FaderBundle(
        model=model,
        domain_profile=profile,
        waveform_canonicalizer=wf_mod,
        latent_canonicalizer=lat_mod,
    )


def encode_to_latent_with_warp(
    model,
    waveform: torch.Tensor,
    *,
    use_mean: bool = True,
) -> torch.Tensor:
    """Encode [C,T] or [1,C,T] with optional canonicalizers attached."""
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if is_fader_model(model) and (
        model.waveform_canonicalizer is not None or model.latent_canonicalizer is not None
    ):
        z, _ = model.encode_with_warp(waveform)
        return z
    return model.encode_to_latent(waveform, use_mean=use_mean)


def load_rave(model_path: str | Path, use_gpu: bool = False) -> rave.RAVE:
    """Load a RAVE/BRAVE checkpoint (backward compat; use load_model for Fader)."""
    return load_model(model_path, use_gpu=use_gpu)


def is_fader_model(model) -> bool:
    """True if loaded module is FaderRAVE."""
    return type(model).__name__ == "FaderRAVE"


def extract_normalized_attributes(
    model,
    waveform: torch.Tensor,
    *,
    latent_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract normalized attribute trajectories for Fader decode.

    Args:
        model: FaderRAVE with stats loaded
        waveform: [C, T] audio on model device

    Returns:
        attr_norm: [1, D, T_lat]
    """
    if not is_fader_model(model):
        raise TypeError("extract_normalized_attributes requires FaderRAVE")

    from rave.fader.providers import AudioDescriptorProvider
    from rave.fader.attributes import latent_length_from_config

    device = waveform.device
    x = waveform.detach().cpu().numpy()
    # --- Mono for descriptor extractors ---
    if x.ndim == 2:
        mono = x.mean(axis=0)
    else:
        mono = x.reshape(-1)

    if latent_length is None:
        t_lat = max(1, mono.shape[-1] // rave.core.get_minimum_size(model))
    else:
        t_lat = latent_length

    provider = AudioDescriptorProvider(
        continuous_attributes=model.continuous_attributes,
        sampling_rate=model.sr,
    )
    raw_cont = provider.load(0, mono, model.sr, t_lat)
    parts = [raw_cont]

    # --- Discrete rows default to zero unless sidecar present at train time ---
    if model.discrete_attributes:
        disc = np.zeros((len(model.discrete_attributes), t_lat), dtype=np.float32)
        parts.append(disc)

    raw = np.concatenate(parts, axis=0)
    raw_t = torch.from_numpy(raw).float().to(device).unsqueeze(0)
    # --- Same path as training: raw → attr_norm via _prepare_attributes ---
    attr_norm, _ = model._prepare_attributes(raw_t)
    return attr_norm


def build_constant_attr(
    model,
    attribute_values: Dict[str, float],
    *,
    time_frames: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build constant attribute tensor from name -> raw value or discrete class index.

    Returns attr_norm [1, D, T_lat] ready for decode().
    """
    if not is_fader_model(model):
        raise TypeError("build_constant_attr requires FaderRAVE")
    if device is None:
        device = next(model.parameters()).device

    rows = []
    for name in model.attribute_names:
        val = attribute_values.get(name, 0.0)
        row = np.full(time_frames, float(val), dtype=np.float32)
        rows.append(row)
    raw = np.stack(rows, axis=0)
    raw_t = torch.from_numpy(raw).float().to(device).unsqueeze(0)
    attr_norm, _ = model._prepare_attributes(raw_t)
    return attr_norm


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
