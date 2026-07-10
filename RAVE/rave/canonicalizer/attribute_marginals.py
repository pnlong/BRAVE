"""Discrete class marginals and in-domain class pools for conditional training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import yaml


def _sidecar_path(db_path: Union[str, Path]) -> Path:
    return Path(db_path) / "attribute_sidecar.yaml"


def _class_counts_path(db_path: Union[str, Path], attr_name: str) -> Path:
    return Path(db_path) / f"{attr_name}_class_counts.json"


def _lookup_sidecar_value(values: dict, index: int):
    key = f"{index:08d}"
    if key in values:
        return values[key]
    if str(index) in values:
        return values[str(index)]
    return 0


def read_discrete_class_per_index(
    db_path: Union[str, Path],
    attr_name: str,
    num_indices: int,
) -> np.ndarray:
    """Per-LMDB-index discrete class id (scalar broadcast in sidecar)."""
    path = _sidecar_path(db_path)
    if not path.is_file():
        return np.zeros(num_indices, dtype=np.int64)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    attr_def = (data.get("attributes") or {}).get(attr_name, {})
    values = attr_def.get("values") or {}

    out = np.zeros(num_indices, dtype=np.int64)
    for idx in range(num_indices):
        val = _lookup_sidecar_value(values, idx)
        if isinstance(val, list):
            val = val[0] if val else 0
        out[idx] = int(val)
    return out


def marginal_probs_from_counts(
    counts: Dict[int, int],
    n_classes: int,
) -> np.ndarray:
    """Normalize count dict to probability vector of length ``n_classes``."""
    probs = np.zeros(n_classes, dtype=np.float64)
    for key, count in counts.items():
        idx = int(key)
        if 0 <= idx < n_classes:
            probs[idx] += float(count)
    total = probs.sum()
    if total <= 0:
        return np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    return probs / total


def load_discrete_marginal_probs(
    db_path: Union[str, Path],
    attr_name: str,
    n_classes: int,
    *,
    num_train_indices: Optional[int] = None,
) -> np.ndarray:
    """
    Y marginal for a discrete attribute.

    Prefers per-index sidecar histogram on the train split; falls back to
    ``{attr}_class_counts.json`` when the sidecar is missing.
    """
    db_path = Path(db_path)
    if num_train_indices is not None and num_train_indices > 0:
        per_index = read_discrete_class_per_index(
            db_path, attr_name, num_train_indices)
        counts: Dict[int, int] = defaultdict(int)
        for cls in per_index.tolist():
            if 0 <= int(cls) < n_classes:
                counts[int(cls)] += 1
        if counts:
            return marginal_probs_from_counts(counts, n_classes)

    counts_path = _class_counts_path(db_path, attr_name)
    if counts_path.is_file():
        with open(counts_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        raw = summary.get("counts") or {}
        counts = {int(k): int(v) for k, v in raw.items()}
        if counts:
            return marginal_probs_from_counts(counts, n_classes)

    return np.full(n_classes, 1.0 / n_classes, dtype=np.float64)


def build_in_domain_class_pools(
    db_path: Union[str, Path],
    attr_name: str,
    num_indices: int,
    n_classes: int,
) -> Dict[int, List[int]]:
    """Map discrete class id -> in-domain LMDB indices (train split range)."""
    per_index = read_discrete_class_per_index(db_path, attr_name, num_indices)
    pools: Dict[int, List[int]] = defaultdict(list)
    for idx, cls in enumerate(per_index.tolist()):
        c = int(cls)
        if c < 0 or c >= n_classes:
            c = 0
        pools[c].append(idx)
    if not pools:
        pools[0] = list(range(num_indices))
    return dict(pools)


def build_ood_discrete_marginals(
    attribute_names: Sequence[str],
    attribute_kinds: Dict[str, str],
    discrete_num_classes: Dict[str, int],
    db_path: Union[str, Path],
    *,
    num_train_indices: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Per discrete attribute name, normalized sampling probabilities."""
    out: Dict[str, np.ndarray] = {}
    for name in attribute_names:
        if attribute_kinds.get(name) != "discrete":
            continue
        n_cls = int(discrete_num_classes.get(name, 0))
        if n_cls <= 1:
            continue
        out[name] = load_discrete_marginal_probs(
            db_path,
            name,
            n_cls,
            num_train_indices=num_train_indices,
        )
    return out
