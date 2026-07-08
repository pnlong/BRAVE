"""FaderRAVE loading for TorchScript / nn~ export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gin
import torch

import rave
from rave.fader.attributes import resolve_stats_path
from rave.fader.model import FaderRAVE


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
        from rave.canonicalizer.export import attach_canonicalizer_for_export

        attach_canonicalizer_for_export(model, canonicalizer_ckpt)

    stats = resolve_stats_path(db_path, stats_path)
    if stats is None:
        raise FileNotFoundError("attribute_stats.yaml not found")

    return model, run, stats


def strip_weight_norm(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
