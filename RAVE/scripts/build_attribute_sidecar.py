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
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = Path(_RAVE_ROOT).parent
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)
_FSD50K = _BRAVE_ROOT / "dataset_exploration" / "fsd50k"
if str(_FSD50K) not in sys.path:
    sys.path.insert(0, str(_FSD50K))

import yaml
from absl import app, flags
from tqdm import tqdm

from fsd50k_manifest import clip_id_from_fname_cell, csv_labels_normalized
from paths import canonical_partition, fsd50k_dataset_root, partitions_for
from tag_utils import normalize_tag
from rave.fader.discrete_class_labels import load_class_labels_from_tags_yaml

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
flags.DEFINE_integer(
    "workers",
    0,
    "Classification worker processes (0=all logical CPU cores; 1=serial)",
)
flags.DEFINE_bool("no_progress", False, "Disable progress bars")

TEXTURE_CLASS_COUNT = 10
RowResult = Tuple[int, int, bool, bool]  # idx, label, missing_csv, skipped
_WORKER_STATE: Dict = {}


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


def load_clip_tags(
    partition_key: str,
    dataset_root: Path,
    *,
    show_progress: bool = False,
) -> Dict[str, List[str]]:
    part = partitions_for(dataset_root)[canonical_partition(partition_key)]
    out: Dict[str, List[str]] = {}
    if not part.csv_path.is_file():
        return out
    with part.csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        row_iter = reader
        if show_progress:
            row_iter = tqdm(
                reader,
                desc=f"csv:{partition_key}",
                unit="row",
            )
        for row in row_iter:
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
    *,
    show_progress: bool = False,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in partition_keys:
        out.update(load_clip_tags(key, dataset_root, show_progress=show_progress))
    return out


def tags_for_entry(entry: dict, clip_tags: Dict[str, List[str]]) -> Set[str]:
    tags: Set[str] = set()
    for cid in entry.get("clip_ids") or [entry.get("clip_id", "")]:
        if not cid:
            continue
        tags.update(clip_tags.get(cid, []))
    return tags


def _init_sidecar_worker(state: Dict) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _classify_entry_worker(entry: dict) -> RowResult:
    state = _WORKER_STATE
    clip_tags = state["clip_tags"]
    tag_set = tags_for_entry(entry, clip_tags)
    missing = not bool(tag_set)
    idx = int(entry["lmdb_index"])
    if state["scheme"] == "water_scene":
        label = classify_water_scene(
            tag_set,
            state["class_1"],
            state["class_2"],
            state["priority_str"],
        )
        skipped = False
    else:
        label = classify_texture_class(
            tag_set,
            state["priority"],
            state["tag_sets"],
            state["id_by_key"],
        )
        skipped = state["texture_only"] and label >= TEXTURE_CLASS_COUNT
    return idx, label, missing, skipped


def _worker_count(workers: int) -> int:
    if workers == 1:
        return 1
    if workers > 0:
        return workers
    return max(1, os.cpu_count() or 1)


def _build_worker_state(
    scheme: str,
    tags_path: Path,
    clip_tags: Dict[str, List[str]],
) -> Dict:
    state: Dict = {
        "scheme": scheme,
        "clip_tags": clip_tags,
        "priority_str": FLAGS.priority,
        "texture_only": FLAGS.texture_only,
    }
    if scheme == "water_scene":
        class_1, class_2 = load_tags_config(tags_path)
        state["class_1"] = frozenset(class_1)
        state["class_2"] = frozenset(class_2)
    else:
        priority, tag_sets, id_by_key = load_texture_tags_config(tags_path)
        state["priority"] = priority
        state["tag_sets"] = {k: frozenset(v) for k, v in tag_sets.items()}
        state["id_by_key"] = id_by_key
    return state


def _classify_entries(
    entries: List[dict],
    scheme: str,
    tags_path: Path,
    clip_tags: Dict[str, List[str]],
    *,
    show_progress: bool = True,
) -> tuple[Dict[str, int], Dict[int, int], int, int]:
    worker_state = _build_worker_state(scheme, tags_path, clip_tags)
    n_workers = _worker_count(FLAGS.workers)
    values: Dict[str, int] = {}
    counts: Dict[int, int] = {}
    missing_csv = 0
    skipped = 0

    if n_workers == 1:
        _init_sidecar_worker(worker_state)
        row_iter = entries
        if show_progress:
            row_iter = tqdm(entries, desc="classify", unit="row")
        results = (_classify_entry_worker(entry) for entry in row_iter)
    else:
        chunksize = max(1, len(entries) // (n_workers * 8))
        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_sidecar_worker,
            initargs=(worker_state,),
        ) as pool:
            row_iter = pool.imap(_classify_entry_worker, entries, chunksize=chunksize)
            if show_progress:
                row_iter = tqdm(
                    row_iter,
                    total=len(entries),
                    desc="classify",
                    unit="row",
                )
            results = list(row_iter)

    for idx, label, missing, row_skipped in results:
        if missing:
            missing_csv += 1
        counts[label] = counts.get(label, 0) + 1
        if row_skipped:
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

    show_progress = not FLAGS.no_progress
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
    clip_tags = load_clip_tags_merged(
        list(FLAGS.partition), root, show_progress=show_progress)

    values, counts, missing_csv, skipped = _classify_entries(
        entries,
        scheme,
        tags_path,
        clip_tags,
        show_progress=show_progress,
    )
    attr_name = "water_scene" if scheme == "water_scene" else "texture_class"
    if scheme == "texture_class":
        n_cls = TEXTURE_CLASS_COUNT if FLAGS.texture_only else max(
            (int(k) for k in counts), default=-1) + 1
        if n_cls <= 0:
            n_cls = TEXTURE_CLASS_COUNT
    else:
        n_cls = 3
    class_labels = load_class_labels_from_tags_yaml(
        tags_path, n_cls, attr_name=attr_name)
    counts_labeled = {
        class_labels[k]: counts[k]
        for k in counts
        if isinstance(k, int) and k < len(class_labels)
    }

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
        "class_labels": class_labels,
        "counts_labeled": counts_labeled,
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
