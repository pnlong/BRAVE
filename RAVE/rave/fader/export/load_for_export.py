"""Shared FaderRAVE loading for TorchScript / nn~ export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gin
import torch

import rave
from rave.fader.attributes import resolve_stats_path
from rave.fader.model import FaderRAVE

CANON_WAVEFORM_NAME = "waveform_canonicalizer.ckpt"
CANON_LATENT_NAME = "latent_canonicalizer.ckpt"


def is_fader_config(config_path: str) -> bool:
    """True if ``config.gin`` configures FaderRAVE (handles gin import aliases)."""
    text = Path(config_path).read_text()
    if "FaderRAVE" in text:
        return True
    gin.parse_config_file(config_path)
    try:
        gin.query_parameter("FaderRAVE.latent_size")
        return True
    except ValueError:
        return False


def is_fader_model(model_path: str) -> bool:
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        return False
    return is_fader_config(config_path)


def resolve_canonicalizer_ckpt(
    model_path: str,
    *,
    mode: str = "auto",
    waveform_canonicalizer: Optional[str] = None,
    latent_canonicalizer: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve canonicalizer checkpoint path.

    ``mode`` is ``auto`` (search run dir), ``none``, or an explicit ckpt path.
    """
    if waveform_canonicalizer and latent_canonicalizer:
        raise ValueError(
            "Specify only one of waveform_canonicalizer or latent_canonicalizer"
        )
    if waveform_canonicalizer:
        return waveform_canonicalizer
    if latent_canonicalizer:
        return latent_canonicalizer
    if mode == "none":
        return None
    if mode not in ("auto", "none"):
        return mode

    run = rave.core.search_for_run(model_path)
    if run is None:
        return None
    run_dir = Path(rave.core.run_dir_from_checkpoint(run))
    wf = run_dir / CANON_WAVEFORM_NAME
    lt = run_dir / CANON_LATENT_NAME
    if wf.is_file():
        return str(wf)
    if lt.is_file():
        return str(lt)
    return None


def load_fader_for_export(
    model_path: str,
    *,
    db_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    canonicalizer_ckpt: Optional[str] = None,
) -> Tuple[FaderRAVE, str, str]:
    """Load FaderRAVE, resolve stats, optionally attach canonicalizer."""
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        raise FileNotFoundError(f"config not found for {model_path}")
    gin.parse_config_file(config_path)
    run = rave.core.search_for_run(model_path)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found for {model_path}")

    model = FaderRAVE()
    model = model.load_from_checkpoint(run)
    model.eval()

    if canonicalizer_ckpt:
        from rave.fader.canonicalizer_config import load_canonicalizer_onto_model

        load_canonicalizer_onto_model(model, canonicalizer_ckpt)

    stats = resolve_stats_path(db_path, stats_path)
    if stats is None:
        raise FileNotFoundError("attribute_stats.yaml not found")

    return model, run, stats


def strip_weight_norm(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
