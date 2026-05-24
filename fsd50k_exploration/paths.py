"""
FSD50K graft paths (read-only) and writable dataset root under Arrakis storage.
"""

from __future__ import annotations

from pathlib import Path

# Graft partitions (read-only mirrors of FSD50K audio + metadata)
FSD50K_TRAIN = Path(
    "/graft1/datasets/kechen/fsd50k/fsd50k/train/mnt/audio_clip/processed_datasets/FSD50K/train"
)
FSD50K_VALID = Path(
    "/graft1/datasets/kechen/fsd50k/fsd50k/valid/mnt/audio_clip/processed_datasets/FSD50K/valid"
)
FSD50K_TEST = Path(
    "/graft1/datasets/kechen/fsd50k/fsd50k/test/mnt/audio_clip/processed_datasets/FSD50K/test"
)

FSD50K_PARTITIONS: dict[str, Path] = {
    "train": FSD50K_TRAIN,
    "valid": FSD50K_VALID,
    "test": FSD50K_TEST,
}

# Writable staging + preprocessed datasets (symlink farms, rave preprocess output, etc.)
DATA_ROOT = Path("/mnt/arrakis_data/pnlong/fsd50k_brave")
TRAIN_AUDIO_SYMLINKS = DATA_ROOT / "train_audio_symlinks"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"


def default_symlink_pool() -> Path:
    return TRAIN_AUDIO_SYMLINKS
