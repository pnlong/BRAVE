"""
JSUT (Japanese speech corpus) layout under Hai lab storage.

Official release: https://sites.google.com/site/shinnosuketakamichi/publication/jsut
Shared unpack: ``$JSUT_ROOT`` (default ``/data/hai-res/shared/p1long/JSUT/jsut_ver1.1``).
Writable BRAVE pool: ``$BRAVE_STORAGE/jsut/``.
"""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_DIR = Path(
    os.environ.get("BRAVE_STORAGE", f"/data/hai-res/{os.environ.get('USER', 'p1long')}/BRAVE-data")
)

JSUT_ROOT = Path(
    os.environ.get("JSUT_ROOT", "/data/hai-res/shared/p1long/JSUT/jsut_ver1.1")
)

DATA_ROOT = STORAGE_DIR / "jsut"
AUDIO_SUBSET_DIR = DATA_ROOT / "audio_subset"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
RUNS_DIR = DATA_ROOT / "runs"

# Official subset folder names under jsut_ver1.1/
ALL_SUBSETS: tuple[str, ...] = (
    "basic5000",
    "countersuffix26",
    "loanword128",
    "onomatopee300",
    "precedent130",
    "repeat500",
    "travel1000",
    "utparaphrase512",
    "voiceactress100",
)
