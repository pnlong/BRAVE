"""YAML sidecar for per-LMDB-index attribute values."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Union

import gin
import numpy as np
import yaml


@gin.configurable
class SidecarAttributeProvider:
    """
    Loads per-sample attribute values from attribute_sidecar.yaml.

    Supports continuous scalars/trajectories and discrete class indices.
    """

    def __init__(
        self,
        sidecar_path: str,
        attribute_names: Sequence[str],
        attribute_kinds: Dict[str, str],
        index_key: str = "lmdb_index",
    ) -> None:
        self.attribute_names = list(attribute_names)
        self.attribute_kinds = dict(attribute_kinds)
        self.index_key = index_key
        path = Path(sidecar_path)
        if not path.is_file():
            self._schema: Dict = {}
            self._values: Dict = {}
            return
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        self._schema = data.get("attributes", {})
        self._values = data
        self.index_key = data.get("index_key", index_key)

    def _lookup_key(self, index: int) -> str:
        return f"{index:08d}"

    def _get_value(self, name: str, index: int) -> Union[float, int, list]:
        attr_def = self._schema.get(name, {})
        values = attr_def.get("values", {})
        key = self._lookup_key(index)
        if key in values:
            return values[key]
        if str(index) in values:
            return values[str(index)]
        return 0

    def load_continuous(
        self,
        names: Sequence[str],
        index: int,
        latent_length: int,
    ) -> np.ndarray:
        rows = []
        for name in names:
            val = self._get_value(name, index)
            if isinstance(val, list):
                row = np.asarray(val, dtype=np.float32)
                if len(row) != latent_length:
                    if len(row) >= latent_length:
                        row = row[:latent_length]
                    else:
                        row = np.pad(row, (0, latent_length - len(row)))
            else:
                row = np.full(latent_length, float(val), dtype=np.float32)
            rows.append(row)
        if not rows:
            return np.zeros((0, latent_length), dtype=np.float32)
        return np.stack(rows, axis=0)

    def load_discrete(
        self,
        names: Sequence[str],
        index: int,
        latent_length: int,
    ) -> np.ndarray:
        rows = []
        for name in names:
            val = self._get_value(name, index)
            if isinstance(val, list):
                row = np.asarray(val, dtype=np.float32)
                if len(row) != latent_length:
                    if len(row) >= latent_length:
                        row = row[:latent_length]
                    else:
                        row = np.pad(row, (0, latent_length - len(row)))
            else:
                row = np.full(latent_length, float(int(val)), dtype=np.float32)
            rows.append(row)
        if not rows:
            return np.zeros((0, latent_length), dtype=np.float32)
        return np.stack(rows, axis=0)
