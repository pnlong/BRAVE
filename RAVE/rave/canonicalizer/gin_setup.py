"""Gin parse helpers for canonicalizer training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import gin

from .in_domain_discriminator import build_in_domain_discriminator

# Bindings that exist only in configs/brave_canonicalizer.gin (not plain brave.gin).
_CANONICALIZER_MARKERS: tuple[str, ...] = (
    "CanonicalizerTrainer.lambda_gan",
    "CanonicalizerTrainer.lambda_rec",
    "InDomainAudioDiscriminator.discriminator",
)

# Heuristic markers when a backbone/fader gin was parsed as --config by mistake.
_FADER_LEAK_MARKERS: tuple[str, ...] = (
    "NUM_TEXTURE_CLASSES",
    "CONTINUOUS_ATTRIBUTES",
    "rave.fader.latent_discriminator",
)


def parse_gin_file(
    cfg_path: str | Path,
    *,
    overrides: Iterable[str] | None = None,
) -> Path:
    """Parse a gin file from its parent directory (so ``include`` resolves)."""
    cfg_path = Path(cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Gin config not found: {cfg_path}")
    prev_cwd = os.getcwd()
    os.chdir(cfg_path.parent)
    try:
        gin.parse_config_file(cfg_path.name)
        for override in overrides or []:
            gin.parse_config(override)
    finally:
        os.chdir(prev_cwd)
    return cfg_path


def validate_canonicalizer_gin(*, canon_cfg: Path) -> None:
    """Fail fast if canonicalizer gin did not parse."""
    cfg = gin.config_str()
    missing = [m for m in _CANONICALIZER_MARKERS if m not in cfg]
    if not missing:
        return

    leak = [m for m in _FADER_LEAK_MARKERS if m in cfg]
    hint = ""
    if leak:
        hint = (
            " Parsed config looks like a backbone/fader gin file. "
            "Use CANONICALIZER_CONFIG=configs/brave_canonicalizer.gin "
            "(and unset CONFIG before sbatch)."
        )
    raise RuntimeError(
        f"Canonicalizer gin {canon_cfg.name!r} is missing bindings {missing}.{hint} "
        f"config_str length={len(cfg)}."
    )


def configure_backbone_gin(
    backbone_cfg: str | Path,
    n_channels: int,
) -> None:
    """Parse backbone gin only (frozen RAVE / FaderRAVE architecture)."""
    gin.clear_config()
    parse_gin_file(backbone_cfg)
    gin.bind_parameter("RAVE.n_channels", n_channels)


def configure_canonicalizer_gin(
    canon_cfg: str | Path,
    n_channels: int,
    *,
    overrides: Iterable[str] | None = None,
) -> Path:
    """Parse canonicalizer gin (includes brave.gin via include)."""
    gin.clear_config()
    canon_cfg = parse_gin_file(canon_cfg, overrides=overrides)
    gin.bind_parameter("RAVE.n_channels", n_channels)
    validate_canonicalizer_gin(canon_cfg=canon_cfg)
    return canon_cfg


__all__ = [
    "build_in_domain_discriminator",
    "configure_backbone_gin",
    "configure_canonicalizer_gin",
    "parse_gin_file",
    "validate_canonicalizer_gin",
]
