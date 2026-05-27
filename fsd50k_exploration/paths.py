"""
FSD50K layout under the Hai lab storage tree + writable preprocessing dirs.

Official release layout (see ``FSD50K.doc``): WAV under ``FSD50K.dev_audio`` /
``FSD50K.eval_audio`` and clip-level labels in ``FSD50K.ground_truth/*.csv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Writable project data — override machine root via ``BRAVE_STORAGE``.
STORAGE_DIR = Path(os.environ.get("BRAVE_STORAGE", "/deepfreeze/pnlong/hai_lab/BRAVE"))

# Official dataset root containing ``FSD50K.dev_audio``, ``FSD50K.ground_truth``, etc.
FSD50K_ROOT = STORAGE_DIR / "FSD50K"

DATA_ROOT = STORAGE_DIR / "fsd50k_brave"
AUDIO_SUBSET_DIR = DATA_ROOT / "audio_subset"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"

CANON_PARTITION_KEYS: tuple[str, ...] = ("dev_train", "dev_val", "eval")

# Optional ``--partition`` synonyms (CSV ``split`` uses train/val; eval is evaluation).
PARTITION_ALIASES: dict[str, str] = {
    "train": "dev_train",
    "valid": "dev_val",
    "test": "eval",
}

PARTITION_CHOICES: tuple[str, ...] = tuple(
    sorted(set(CANON_PARTITION_KEYS) | set(PARTITION_ALIASES))
)


def canonical_partition(name: str) -> str:
    return PARTITION_ALIASES.get(name, name)


def fsd50k_dataset_root(dataset_root: Path | None = None) -> Path:
    return FSD50K_ROOT if dataset_root is None else Path(dataset_root)


@dataclass(frozen=True)
class FsdPartition:
    """Logical split: WAV under ``audio_dir``, rows filtered from ``csv_path``."""

    name: str
    audio_dir: Path
    csv_path: Path
    csv_split: str | None


def partitions_for(fsd_root: Path | None = None) -> dict[str, FsdPartition]:
    root = Path(fsd_root if fsd_root is not None else FSD50K_ROOT)
    gt = root / "FSD50K.ground_truth"
    dev_audio = root / "FSD50K.dev_audio"
    eval_audio = root / "FSD50K.eval_audio"
    return {
        "dev_train": FsdPartition(
            "dev_train", dev_audio, gt / "dev.csv", csv_split="train"
        ),
        "dev_val": FsdPartition("dev_val", dev_audio, gt / "dev.csv", csv_split="val"),
        "eval": FsdPartition("eval", eval_audio, gt / "eval.csv", csv_split=None),
    }


def default_subset_audio_dir() -> Path:
    """Default **`build_subset --output-dir`**: flat staged WAV folder for RAVE preprocess."""
    return AUDIO_SUBSET_DIR
