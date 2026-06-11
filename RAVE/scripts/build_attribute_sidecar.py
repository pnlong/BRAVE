"""
Build attribute_sidecar.yaml from lmdb_index_manifest + FSD50K CSV labels.

Usage (BRAVE root):
  python RAVE/scripts/build_attribute_sidecar.py \\
    --db_path /path/to/lmdb \\
    --scheme water_scene \\
    --partition dev_train

  python RAVE/scripts/build_attribute_sidecar.py \\
    --db_path /path/to/lmdb \\
    --scheme texture_class \\
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
flags.DEFINE_string("scheme", "water_scene", "Sidecar scheme: water_scene | texture_class")
flags.DEFINE_multi_string(
    "partition",
    ["dev_train"],
    "FSD50K partition(s) for CSV labels (repeat for dev_train+dev_val+eval)",
)
flags.DEFINE_string(
    "dataset_root",
    None,
    "FSD50K release root (default: $BRAVE_STORAGE/FSD50K)",
)
flags.DEFINE_string(
    "tags_config",
    None,
    "YAML tag sets (default: scheme-specific under dataset_exploration/fsd50k/configs/)",
)
flags.DEFINE_bool(
    "texture_only",
    True,
    "texture_class: omit sidecar rows for classes 10–11 (human/vocal, music)",
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


TEXTURE_CLASS_COUNT = 10


def _default_tags_config(scheme: str) -> Path:
    if scheme == "texture_class":
        return _FSD50K / "configs" / "fader_texture_class_tags.yaml"
    return _FSD50K / "configs" / "fader_water_scene_tags.yaml"


def load_texture_tags_config(path: Path) -> tuple[List[str], Dict[str, Set[str]], Dict[str, int]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    priority = list(data.get("priority_order", []))
    id_by_key = {v: int(k) for k, v in (data.get("class_names") or {}).items()}
    tag_sets: Dict[str, Set[str]] = {}
    for key in priority:
        tag_sets[key] = {normalize_tag(t) for t in data.get(key, [])}
    return priority, tag_sets, id_by_key


def classify_texture_class(
    tags: Set[str],
    priority: List[str],
    tag_sets: Dict[str, Set[str]],
    id_by_key: Dict[str, int],
) -> int:
    for key in priority:
        if tags & tag_sets.get(key, set()):
            return id_by_key[key]
    return 0


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


def load_clip_tags_merged(
    partition_keys: List[str],
    dataset_root: Path,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in partition_keys:
        out.update(load_clip_tags(key, dataset_root))
    return out


def tags_for_entry(entry: dict, clip_tags: Dict[str, List[str]]) -> Set[str]:
    tags: Set[str] = set()
    for cid in entry.get("clip_ids") or [entry.get("clip_id", "")]:
        if not cid:
            continue
        tags.update(clip_tags.get(cid, []))
    return tags


def _build_water_scene(
    entries: List[dict],
    clip_tags: Dict[str, List[str]],
    tags_path: Path,
) -> tuple[Dict[str, int], Dict[int, int], int, int]:
    class_1, class_2 = load_tags_config(tags_path)
    values: Dict[str, int] = {}
    counts = {0: 0, 1: 0, 2: 0}
    missing_csv = 0
    skipped = 0

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

    return values, counts, missing_csv, skipped


def _build_texture_class(
    entries: List[dict],
    clip_tags: Dict[str, List[str]],
    tags_path: Path,
) -> tuple[Dict[str, int], Dict[int, int], int, int]:
    priority, tag_sets, id_by_key = load_texture_tags_config(tags_path)
    values: Dict[str, int] = {}
    counts: Dict[int, int] = {}
    missing_csv = 0
    skipped = 0

    for entry in entries:
        idx = int(entry["lmdb_index"])
        tag_set = tags_for_entry(entry, clip_tags)
        if not tag_set:
            missing_csv += 1
        label = classify_texture_class(tag_set, priority, tag_sets, id_by_key)
        counts[label] = counts.get(label, 0) + 1
        if FLAGS.texture_only and label >= TEXTURE_CLASS_COUNT:
            skipped += 1
            continue
        values[f"{idx:08d}"] = label

    return values, counts, missing_csv, skipped


def main(argv):
    del argv
    scheme = FLAGS.scheme
    if scheme not in ("water_scene", "texture_class"):
        print(f"Unknown scheme: {scheme}")
        return

    db_path = FLAGS.db_path
    manifest_path = FLAGS.manifest_path or os.path.join(
        db_path, "lmdb_index_manifest.yaml")
    if not os.path.isfile(manifest_path):
        print(f"Missing manifest: {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f) or {}
    entries = manifest.get("entries", [])

    tags_path = (
        Path(FLAGS.tags_config)
        if FLAGS.tags_config
        else _default_tags_config(scheme)
    )
    root = fsd50k_dataset_root(
        Path(FLAGS.dataset_root) if FLAGS.dataset_root else None)
    clip_tags = load_clip_tags_merged(list(FLAGS.partition), root)

    if scheme == "water_scene":
        values, counts, missing_csv, skipped = _build_water_scene(
            entries, clip_tags, tags_path)
        attr_name = "water_scene"
    else:
        values, counts, missing_csv, skipped = _build_texture_class(
            entries, clip_tags, tags_path)
        attr_name = "texture_class"

    sidecar = {
        "index_key": "lmdb_index",
        "attributes": {
            attr_name: {
                "values": values,
            },
        },
    }
    out_yaml = os.path.join(db_path, "attribute_sidecar.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(sidecar, f, sort_keys=False)
    print(f"Wrote {out_yaml} ({len(values)} indices)")

    total = max(len(entries), 1)
    summary = {
        "scheme": scheme,
        "partitions": list(FLAGS.partition),
        "counts": counts,
        "fractions": {str(k): v / total for k, v in counts.items()},
        "missing_csv_labels": missing_csv,
        "skipped_non_texture": skipped,
        "texture_only": FLAGS.texture_only if scheme == "texture_class" else None,
        "tags_config": str(tags_path),
    }
    if scheme == "water_scene":
        summary["priority"] = FLAGS.priority
    counts_path = os.path.join(db_path, f"{scheme}_class_counts.json")
    with open(counts_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {counts_path}: {summary}")


if __name__ == "__main__":
    app.run(main)
