#!/usr/bin/env python3
"""
Enumerate normalized FSD50K class labels across selected splits and print a
descending frequency table (tab-separated: ``tag,count``).

Reads official ``dev.csv`` / ``eval.csv`` under the dataset root — see ``paths.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from fsd50k_manifest import count_manifest_clips, iter_manifest_clips
from paths import PARTITION_CHOICES, canonical_partition, partitions_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--partition",
        action="append",
        choices=list(PARTITION_CHOICES),
        metavar="PART",
        dest="partitions",
        help="Canonical dev_train/dev_val/eval or synonyms train/valid/test. Repeat flag to combine. Default: all three.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Directory containing official FSD50K.* release folders "
            "(default: $BRAVE_STORAGE/FSD50K)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max clips scanned per canonical partition after split filtering (for quick tests).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar on stderr.",
    )
    return parser.parse_args()


def ordered_canonical_partitions(
    partitions_arg: list[str] | None, roots: dict[str, object]
) -> list[str]:
    order = tuple(roots.keys())
    if partitions_arg:
        keys: list[str] = []
        seen: set[str] = set()
        for raw in partitions_arg:
            c = canonical_partition(raw)
            if c in seen:
                continue
            if c not in roots:
                continue
            seen.add(c)
            keys.append(c)
        out = sorted(keys, key=lambda k: order.index(k))
        return out
    return list(order)


def main() -> None:
    args = parse_args()
    roots = partitions_for(args.dataset_root)
    use_parts = ordered_canonical_partitions(args.partitions, roots)

    counter: Counter[str] = Counter()
    clips_seen = 0

    for name in use_parts:
        part = roots[name]
        if not part.csv_path.is_file():
            print(f"# warning: missing manifest CSV: {part.csv_path}", file=sys.stderr)
            continue
        iterator = iter_manifest_clips(part, limit=args.limit)
        if args.no_progress:
            iterator_wrapped = iterator
        else:
            n_targets = count_manifest_clips(part, limit=args.limit)
            iterator_wrapped = tqdm(
                iterator,
                total=n_targets,
                desc=name,
                unit="clip",
                smoothing=0.05,
                file=sys.stderr,
            )
        for _cid, labels in iterator_wrapped:
            clips_seen += 1
            for nt in labels:
                if nt:
                    counter[nt] += 1

    print(f"# partitions={use_parts}", file=sys.stderr)
    print(f"# clips_seen={clips_seen}", file=sys.stderr)

    for tag, ct in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{tag}\t{ct}")


if __name__ == "__main__":
    main()
