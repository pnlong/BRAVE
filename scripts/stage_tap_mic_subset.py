#!/usr/bin/env python3
"""Classify tap WAVs by mic and optionally symlink a training subset.

Writes ``mic_type.yaml`` (path → shotgun | contact | sm57) beside the audio
root. ``--stage-mic shotgun`` builds a directory of symlinks for preprocess.

    python scripts/stage_tap_mic_subset.py \\
      --audio-dir $BRAVE_STORAGE/tap_samples/audio_subset \\
      --stage-mic shotgun
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import yaml

_BRAVE = Path(__file__).resolve().parents[1]
_TAP = _BRAVE / "dataset_exploration" / "tap_samples"
if str(_TAP) not in sys.path:
    sys.path.insert(0, str(_TAP))

from mic_type import (  # noqa: E402
    MIC_TYPES,
    classify_dir,
    counts,
    files_for_mic,
    rename_tap_files,
)

BRAVE_STORAGE = Path(
    os.environ.get("BRAVE_STORAGE", f"/data/hai-res/{os.environ.get('USER', 'p1long')}/BRAVE-data")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audio-dir",
        type=Path,
        default=BRAVE_STORAGE / "tap_samples" / "audio_subset",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="YAML output (default: <audio-dir>/../mic_type.yaml)",
    )
    p.add_argument(
        "--stage-mic",
        choices=MIC_TYPES,
        default=None,
        help="Symlink this mic class into --stage-dir",
    )
    p.add_argument(
        "--rename",
        action="store_true",
        help="Rewrite names to Shotgun- / Contact Mic L|R- / sm57- prefixes",
    )
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Default: sibling audio_subset_<mic>",
    )
    return p.parse_args()


def stage_symlinks(files: Iterable[Path], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in files:
        link = dest / src.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())
        n += 1
    return n


def main() -> None:
    args = parse_args()
    audio_dir = args.audio_dir.resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir missing: {audio_dir}")
    if args.rename:
        pairs = rename_tap_files(audio_dir)
        print(f"renamed {len(pairs)} files in {audio_dir}")
        for src, dest in pairs:
            print(f"  {src.name} -> {dest.name}")
    mapping = classify_dir(audio_dir)
    c = counts(mapping)
    manifest = args.manifest or (audio_dir.parent / "mic_type.yaml")
    payload = {
        "audio_dir": str(audio_dir),
        "rules": {
            "shotgun": "filename starts with 'Shotgun-'",
            "contact": "filename starts with 'Contact Mic L-' or 'Contact Mic R-'",
            "sm57": "filename starts with 'sm57-'",
        },
        "counts": c,
        "files": mapping,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    print(f"wrote {manifest}")
    print("counts:", c)

    if args.stage_mic:
        dest = args.stage_dir or (audio_dir.parent / f"audio_subset_{args.stage_mic}")
        files = files_for_mic(audio_dir, args.stage_mic)  # type: ignore[arg-type]
        n = stage_symlinks(files, dest)
        print(f"staged {n} {args.stage_mic} files → {dest}")


if __name__ == "__main__":
    main()
