"""
Resolve human-readable labels for discrete Fader attributes.

Used by precompute_descriptors.py (bake into attribute_stats.yaml),
build_attribute_sidecar.py (class_counts JSON), and export (Max menus).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import yaml

_BUILTIN_LABELS: Dict[str, List[str]] = {
    "water_scene": ["other", "storm", "coastal"],
}

_FSD50K_ATTR_TO_SCHEME = {
    "texture_class": "texture_class",
    "water_scene": "water_scene",
}


def _brave_root() -> Path:
    return Path(__file__).resolve().parents[3]


def fsd50k_default_tags_yaml(attr_name: str) -> Optional[Path]:
    """Default FSD50K tags YAML for a discrete attribute name."""
    scheme = _FSD50K_ATTR_TO_SCHEME.get(attr_name)
    if scheme is None:
        return None
    root = _brave_root() / "dataset_exploration" / "fsd50k" / "configs"
    if scheme == "texture_class":
        path = root / "fader_texture_class_tags.yaml"
    else:
        path = root / "fader_water_scene_tags.yaml"
    return path if path.is_file() else None


def load_class_labels_from_tags_yaml(
    yaml_path: Union[str, Path],
    num_classes: int,
    *,
    attr_name: Optional[str] = None,
) -> List[str]:
    """Load ``num_classes`` labels (indices 0..K-1) from a tags config YAML."""
    labels = [str(i) for i in range(num_classes)]
    path = Path(yaml_path)
    if path.is_file():
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("class_names") or {}
        for key, name in raw.items():
            idx = int(key)
            if 0 <= idx < num_classes:
                labels[idx] = str(name)
    if attr_name and attr_name in _BUILTIN_LABELS:
        builtin = _BUILTIN_LABELS[attr_name]
        for i, name in enumerate(builtin):
            if i < num_classes:
                labels[i] = name
    return labels


def discover_tags_config(db_path: Union[str, Path], attr_name: str) -> Optional[Path]:
    """Find tags YAML via ``{attr}_class_counts.json`` or FSD50K defaults."""
    db_path = Path(db_path)
    counts_file = db_path / f"{attr_name}_class_counts.json"
    if counts_file.is_file():
        data = json.loads(counts_file.read_text())
        labels = data.get("class_labels")
        if isinstance(labels, list) and labels:
            return None  # caller can use labels from JSON directly
        tags = data.get("tags_config")
        if tags:
            path = Path(tags)
            if path.is_file():
                return path
    return fsd50k_default_tags_yaml(attr_name)


def class_labels_from_class_counts(
    db_path: Union[str, Path],
    attr_name: str,
    num_classes: int,
) -> Optional[List[str]]:
    """Return ``class_labels`` from sidecar counts JSON when already present."""
    counts_file = Path(db_path) / f"{attr_name}_class_counts.json"
    if not counts_file.is_file():
        return None
    data = json.loads(counts_file.read_text())
    labels = data.get("class_labels")
    if not isinstance(labels, list) or len(labels) < num_classes:
        return None
    return [str(x) for x in labels[:num_classes]]


def resolve_discrete_class_labels(
    db_path: Union[str, Path],
    discrete_attributes: Sequence[str],
    discrete_num_classes: Dict[str, int],
    *,
    stats: Optional[dict] = None,
    tags_yaml_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Resolve menu labels for each discrete attribute.

    Priority: baked stats → class_counts JSON → tags YAML → built-in names → indices.
    """
    db_path = Path(db_path)
    out: Dict[str, List[str]] = {}
    baked = (stats or {}).get("discrete_class_labels") or {}
    overrides = tags_yaml_overrides or {}

    for name in discrete_attributes:
        if name in baked and isinstance(baked[name], list):
            k = int(discrete_num_classes.get(name, len(baked[name])))
            out[name] = [str(x) for x in baked[name][:k]]
            continue

        k = int(discrete_num_classes.get(name, 2))
        from_counts = class_labels_from_class_counts(db_path, name, k)
        if from_counts is not None:
            out[name] = from_counts
            continue

        tags_path = overrides.get(name)
        if tags_path:
            out[name] = load_class_labels_from_tags_yaml(tags_path, k, attr_name=name)
            continue

        discovered = discover_tags_config(db_path, name)
        if discovered is not None:
            out[name] = load_class_labels_from_tags_yaml(discovered, k, attr_name=name)
            continue

        if name in _BUILTIN_LABELS:
            base = _BUILTIN_LABELS[name]
            out[name] = [base[i] if i < len(base) else str(i) for i in range(k)]
        else:
            out[name] = [f"{name} {i}" for i in range(k)]

    return out
