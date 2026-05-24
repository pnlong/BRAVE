"""
Iterate clips from official FSD50K ``*.csv`` ground-truth manifests.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from pathlib import Path

from paths import FsdPartition
from tag_utils import normalize_tag


def clip_id_from_fname_cell(cell: str) -> str:
    """Stem of clip id — official CSV rows typically use bare ids without extension."""
    return Path(cell.strip()).stem


def csv_labels_normalized(labels_cell: str) -> list[str]:
    """Normalize comma-separated class names from the official ``labels`` CSV column."""
    return [normalize_tag(part) for part in labels_cell.split(",") if part.strip()]


def iter_manifest_rows(
    partition: FsdPartition,
    *,
    limit: int | None = None,
) -> Iterator[tuple[str, str]]:
    """
    Yield ``(clip_id, labels_cell)`` rows after manifest ``split`` filtering (same as clip iteration).
    ``labels_cell`` is the raw ``labels`` column string.
    """
    if not partition.csv_path.is_file():
        return

    with partition.csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        n = 0
        if reader.fieldnames is None:
            print(f"# empty csv manifest: {partition.csv_path}", file=sys.stderr)
            return
        for row in reader:
            split_key = partition.csv_split
            if split_key is not None and row.get("split") != split_key:
                continue
            cell = row.get("fname") or ""
            cid = clip_id_from_fname_cell(cell)
            if not cid:
                continue
            labs_cell = row.get("labels") or ""
            yield cid, labs_cell
            n += 1
            if limit is not None and n >= limit:
                break


def count_manifest_clips(partition: FsdPartition, *, limit: int | None = None) -> int:
    """Matching clip count after ``split`` filtering (respects optional ``limit``)."""
    return sum(1 for _ in iter_manifest_rows(partition, limit=limit))


def iter_manifest_clips(
    partition: FsdPartition,
    *,
    limit: int | None = None,
) -> Iterator[tuple[str, list[str]]]:
    """
    Yield ``(clip_id, normalized_label_tokens)`` for each clip in ``partition``.
    """
    for cid, lab_cell in iter_manifest_rows(partition, limit=limit):
        yield cid, csv_labels_normalized(lab_cell)
