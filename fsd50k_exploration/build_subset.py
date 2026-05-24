#!/usr/bin/env python3
"""
Build a flat folder of symlinked WAV clips whose ground-truth labels match a whitelist.

Whitelist: UTF-8 text, one stripped-lowercase ontology class token per non-empty line.

Labels come from official ``*.csv`` ``labels`` cells (comma-separated). A clip is kept
when any normalized token intersects the whitelist (exact match).

Default partition is ``dev_train`` (training split inside the official development set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from fsd50k_manifest import count_manifest_clips, iter_manifest_clips
from paths import PARTITION_CHOICES, canonical_partition, default_symlink_pool, partitions_for
from tag_utils import normalize_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--whitelist",
        required=True,
        type=Path,
        help="Plain text file with one strip+lower class token per non-empty line.",
    )
    parser.add_argument(
        "--partition",
        choices=list(PARTITION_CHOICES),
        default="dev_train",
        help="Split manifest (dev_train/dev_val/eval or synonyms train/valid/test). Default: dev_train.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Directory containing official FSD50K.* folders (default: $BRAVE_STORAGE/FSD50K).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Symlink staging directory (default: {default_symlink_pool()!s}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing symlinks in the output folder when names collide.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (stderr); only prints final counters.",
    )
    return parser.parse_args()


def load_whitelist(path: Path) -> set[str]:
    out: set[str] = set()
    data = path.read_text(encoding="utf-8").splitlines()
    for line in data:
        t = normalize_tag(line)
        if t:
            out.add(t)
    if not out:
        raise ValueError(f"Whitelist appears empty after parsing: {path}")
    return out


def main() -> None:
    args = parse_args()
    roots = partitions_for(args.dataset_root)
    name = canonical_partition(args.partition)
    part = roots[name]

    out_dir = args.output_dir if args.output_dir is not None else default_symlink_pool()
    audio_dir = part.audio_dir

    if not audio_dir.is_dir():
        print(f"error: audio partition directory missing: {audio_dir}", file=sys.stderr)
        sys.exit(1)
    if not part.csv_path.is_file():
        print(f"error: CSV manifest missing: {part.csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    whitelist = load_whitelist(args.whitelist)

    selected = 0
    skipped_no_wav = 0
    skipped_no_match = 0

    clip_iter = iter_manifest_clips(part)

    if args.no_progress:
        clip_iter_wrapped = clip_iter
    else:
        n_clips = count_manifest_clips(part)
        clip_iter_wrapped = tqdm(
            clip_iter,
            total=n_clips,
            desc=f"Subset {name}",
            unit="clip",
            smoothing=0.05,
            file=sys.stderr,
        )

    for cid, labels in clip_iter_wrapped:
        if not set(labels) & whitelist:
            skipped_no_match += 1
            continue
        wav_path = audio_dir / f"{cid}.wav"
        if not wav_path.is_file():
            skipped_no_wav += 1
            continue

        link_path = out_dir / f"{cid}.wav"
        abs_target = wav_path.resolve()

        if link_path.exists() or link_path.is_symlink():
            if args.overwrite:
                link_path.unlink()
            else:
                print(
                    f"# skip existing ({cid}.wav); use --overwrite to replace",
                    file=sys.stderr,
                )
                continue

        link_path.symlink_to(abs_target)
        selected += 1

    print(
        f"# partition={name} linked={selected} no_match={skipped_no_match} no_wav={skipped_no_wav}",
        file=sys.stderr,
    )
    print(f"# output dir: {out_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
