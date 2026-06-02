"""Paths for latent exploration artifacts."""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = MODULE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
RECONSTRUCTIONS_DIR = ARTIFACTS_DIR / "reconstructions"

_RAVE_ROOT = MODULE_DIR.parent / "RAVE"


def rave_root() -> Path:
    return _RAVE_ROOT
