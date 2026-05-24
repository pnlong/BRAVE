#!/usr/bin/env python3
"""
Build a flat folder of symlinked FLAC clips whose JSON matches a tag whitelist.

Whitelist: UTF-8 text, one stripped-lowercase tags-of-interest per non-empty line.
A clip is included if any normalized tag from the clip JSON intersects the whitelist (exact match).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paths import FSD50K_PARTITIONS, default_symlink_pool
from tag_utils import iter_clip_tags_raw, normalize_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--whitelist",
        required=True,
        type=Path,
        help="Plain text file with one strip+lower tag token per non-empty line.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Partition root containing paired id.json / id.flac (default: FSD train).",
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
        "--quiet-skip-errors",
        action="store_true",
        help="Warn on malformed JSON rather than exiting.",
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


def clip_normalized_tags(metadata: dict) -> set[str]:
    tags: set[str] = set()
    for raw in iter_clip_tags_raw(metadata):
        nt = normalize_tag(raw)
        if nt:
            tags.add(nt)
    return tags


def main() -> None:
    args = parse_args()
    src = args.source if args.source is not None else FSD50K_PARTITIONS["train"]
    out_dir = args.output_dir if args.output_dir is not None else default_symlink_pool()

    if not src.is_dir():
        print(f"error: source is not a directory: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    whitelist = load_whitelist(args.whitelist)

    selected = 0
    skipped_no_flac = 0
    skipped_no_match = 0
    bad_json = 0

    json_paths = sorted(src.glob("*.json"))
    for jp in json_paths:
        try:
            with open(jp, encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                bad_json += 1
                continue
            if not clip_normalized_tags(meta) & whitelist:
                skipped_no_match += 1
                continue
            flac = jp.with_suffix(".flac")
            if not flac.is_file():
                skipped_no_flac += 1
                continue

            stem = jp.stem
            link_path = out_dir / f"{stem}.flac"
            abs_target = flac.resolve()

            if link_path.exists() or link_path.is_symlink():
                if args.overwrite:
                    link_path.unlink()
                else:
                    print(
                        f"# skip existing ({stem}.flac); use --overwrite to replace",
                        file=sys.stderr,
                    )
                    continue

            link_path.symlink_to(abs_target)
            selected += 1
        except (OSError, json.JSONDecodeError) as e:
            bad_json += 1
            if not args.quiet_skip_errors:
                print(f"# bad json {jp}: {e}", file=sys.stderr)

    print(
        f"# linked={selected} no_match={skipped_no_match} no_flac={skipped_no_flac} bad_meta={bad_json}",
        file=sys.stderr,
    )
    print(f"# output dir: {out_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
