#!/usr/bin/env python3
"""
Enumerate normalized FSD50K tags across selected partitions (train/valid/test) and print
a descending frequency table (tab-separated: tag,count).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from paths import FSD50K_PARTITIONS
from tag_utils import iter_clip_tags_raw, normalize_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--partition",
        action="append",
        choices=list(FSD50K_PARTITIONS),
        metavar="PART",
        dest="partitions",
        help="Train/valid/test; repeat flag to pick several. Default: all three.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max JSON files scanned per partition (for quick tests).",
    )
    parser.add_argument(
        "--quiet-skip-errors",
        action="store_true",
        help="On JSON parse/read errors print to stderr once per file category and continue.",
    )
    return parser.parse_args()


def iter_json_paths(partition_dir: Path, limit: int | None):
    paths = sorted(partition_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    yield from paths


def main() -> None:
    args = parse_args()
    partitions = args.partitions or list(FSD50K_PARTITIONS)

    counter: Counter[str] = Counter()
    scanned = 0
    unreadable = 0

    for name in partitions:
        root = FSD50K_PARTITIONS[name]
        if not root.is_dir():
            print(f"# warning: partition directory missing: {root}", file=sys.stderr)
            continue
        for jp in iter_json_paths(root, args.limit):
            scanned += 1
            try:
                with open(jp, encoding="utf-8") as f:
                    meta = json.load(f)
                if not isinstance(meta, dict):
                    unreadable += 1
                    continue
                for raw in iter_clip_tags_raw(meta):
                    nt = normalize_tag(raw)
                    if nt:
                        counter[nt] += 1
            except (OSError, json.JSONDecodeError) as e:
                unreadable += 1
                if not args.quiet_skip_errors:
                    print(f"# skip {jp}: {e}", file=sys.stderr)

    print(f"# partitions={partitions}", file=sys.stderr)
    print(f"# json_seen={scanned} json_skipped_bad={unreadable}", file=sys.stderr)

    # Descending frequency, then alphabetical for ties
    for tag, ct in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{tag}\t{ct}")


if __name__ == "__main__":
    main()
