"""
Build attribute_sidecar.yaml from lmdb_index_manifest + FSD50K CSV labels.

Usage (BRAVE root):
  python RAVE/scripts/build_attribute_sidecar.py \\
    --db_path /path/to/lmdb \\
    --scheme water_scene \\
    --partition dev_train
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)
_FSD50K = _BRAVE_ROOT / "dataset_exploration" / "fsd50k"
if str(_FSD50K) not in sys.path:
    sys.path.insert(0, str(_FSD50K))

import yaml
from absl import app, flags

from fsd50k_manifest import clip_id_from_fname_cell, csv_labels_normalized
from paths import canonical_partition, fsd50k_dataset_root, partitions_for
from tag_utils import normalize_tag

FLAGS = flags.FLAGS

flags.DEFINE_string("db_path", None, "LMDB with lmdb_index_manifest.yaml", required=True)
flags.DEFINE_string("scheme", "water_scene", "Sidecar scheme: water_scene")
flags.DEFINE_string("partition", "dev_train", "FSD50K partition for CSV labels")
flags.DEFINE_string(
    "dataset_root",
    None,
    "FSD50K release root (default: $BRAVE_STORAGE/FSD50K)",
)
flags.DEFINE_string(
    "tags_config",
    None,
    "YAML with class_1_storm / class_2_coastal lists",
)
flags.DEFINE_string(
    "priority",
    "storm_first",
    "Multi-label: storm_first (1>2) or coastal_first (2>1)",
)
flags.DEFINE_string(
    "manifest_path",
    None,
    "Override manifest yaml path",
)


def _default_tags_config() -> Path:
    return _BRAVE_ROOT / "configs" / "fader_water_scene_tags.yaml"


def load_tags_config(path: Path) -> tuple[Set[str], Set[str]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    c1 = {normalize_tag(t) for t in data.get("class_1_storm", [])}
    c2 = {normalize_tag(t) for t in data.get("class_2_coastal", [])}
    return c1, c2


def classify_water_scene(
    tags: Set[str],
    class_1: Set[str],
    class_2: Set[str],
    priority: str,
) -> int:
    has_1 = bool(tags & class_1)
    has_2 = bool(tags & class_2)
    if priority == "coastal_first":
        if has_2:
            return 2
        if has_1:
            return 1
        return 0
    if has_1:
        return 1
    if has_2:
        return 2
    return 0


def load_clip_tags(partition_key: str, dataset_root: Path) -> Dict[str, List[str]]:
    part = partitions_for(dataset_root)[canonical_partition(partition_key)]
    out: Dict[str, List[str]] = {}
    if not part.csv_path.is_file():
        return out
    with part.csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if part.csv_split is not None and row.get("split") != part.csv_split:
                continue
            cid = clip_id_from_fname_cell(row.get("fname") or "")
            if not cid:
                continue
            out[cid] = csv_labels_normalized(row.get("labels") or "")
    return out


def tags_for_entry(entry: dict, clip_tags: Dict[str, List[str]]) -> Set[str]:
    tags: Set[str] = set()
    for cid in entry.get("clip_ids") or [entry.get("clip_id", "")]:
        if not cid:
            continue
        tags.update(clip_tags.get(cid, []))
    return tags


def main(argv):
    del argv
    db_path = FLAGS.db_path
    manifest_path = FLAGS.manifest_path or os.path.join(
        db_path, "lmdb_index_manifest.yaml")
    if not os.path.isfile(manifest_path):
        print(f"Missing manifest: {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f) or {}
    entries = manifest.get("entries", [])

    tags_path = Path(FLAGS.tags_config) if FLAGS.tags_config else _default_tags_config()
    class_1, class_2 = load_tags_config(tags_path)
    root = fsd50k_dataset_root(
        Path(FLAGS.dataset_root) if FLAGS.dataset_root else None)
    clip_tags = load_clip_tags(FLAGS.partition, root)

    values: Dict[str, int] = {}
    counts = {0: 0, 1: 0, 2: 0}
    missing_csv = 0

    for entry in entries:
        idx = int(entry["lmdb_index"])
        tag_set = tags_for_entry(entry, clip_tags)
        if not tag_set:
            missing_csv += 1
        label = classify_water_scene(
            tag_set, class_1, class_2, FLAGS.priority)
        key = f"{idx:08d}"
        values[key] = label
        counts[label] = counts.get(label, 0) + 1

    sidecar = {
        "index_key": "lmdb_index",
        "attributes": {
            "water_scene": {
                "values": values,
            },
        },
    }
    out_yaml = os.path.join(db_path, "attribute_sidecar.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)
    print(f"Wrote {out_yaml} ({len(values)} indices)")

    total = max(len(values), 1)
    summary = {
        "scheme": FLAGS.scheme,
        "partition": FLAGS.partition,
        "priority": FLAGS.priority,
        "counts": counts,
        "fractions": {str(k): v / total for k, v in counts.items()},
        "missing_csv_labels": missing_csv,
        "tags_config": str(tags_path),
    }
    counts_path = os.path.join(db_path, "water_scene_class_counts.json")
    with open(counts_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {counts_path}: {summary}")


if __name__ == "__main__":
    app.run(main)
