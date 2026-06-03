"""
Tabla ISMIR 2021 (4-way stroke) layout under Hai lab storage + local exploration artifacts.

Official release: https://zenodo.org/records/7110248
"""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_DIR = Path(os.environ.get("BRAVE_STORAGE", "/deepfreeze/pnlong/hai_lab/BRAVE"))

DATA_ROOT = STORAGE_DIR / "tabla_ismir21"
RAW_DIR = DATA_ROOT / "raw"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
LISTEN_SAMPLES_DIR = ARTIFACTS_DIR / "listen_samples"

# Top folder name inside the Zenodo zip after unpack (adjust if your tree differs).
DEFAULT_UNPACKED_NAME = "4way-tabla-ismir21-dataset"

SPLIT_CHOICES: tuple[str, ...] = ("train", "test")


def unpacked_root(explicit: Path | None = None) -> Path:
    """Root of extracted Zenodo archive (contains ``train/`` and ``test/``)."""
    if explicit is not None:
        return Path(explicit)
    candidate = DATA_ROOT / DEFAULT_UNPACKED_NAME
    if candidate.is_dir():
        return candidate
    # Fallback: single child under DATA_ROOT from a flat unzip into raw/
    raw_parent = RAW_DIR.parent
    for child in sorted(raw_parent.iterdir()) if raw_parent.is_dir() else []:
        if child.is_dir() and (child / "train").is_dir():
            return child
    return candidate


def split_audio_dir(split: str, *, unpacked: Path | None = None) -> Path:
    """``train`` or ``test`` tree (stroke-class subfolders with ``.wav`` + ``.onsets``)."""
    if split not in SPLIT_CHOICES:
        raise ValueError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")
    return unpacked_root(unpacked) / split


def default_preprocess_input(split: str = "train") -> Path:
    """RAVE ``preprocess.py --input_path`` for BRAVE training (train split only)."""
    return split_audio_dir(split)
