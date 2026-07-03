"""Write host-facing control metadata beside exported Fader models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from ..attributes import load_attribute_stats
from ..discrete_class_labels import resolve_discrete_class_labels
from .trace_model import FaderTraceModel


def _discrete_labels_for_host(stats_data: dict, stats_path: Union[str, Path]) -> dict:
    labels = stats_data.get("discrete_class_labels") or {}
    if labels:
        return labels
    discrete = stats_data.get("discrete_attributes") or []
    if not discrete:
        return {}
    db_path = Path(stats_path).parent
    return resolve_discrete_class_labels(
        db_path,
        discrete,
        stats_data.get("discrete_num_classes") or {},
        stats=stats_data,
    )


def write_host_controls_json(
    output_ts_path: Union[str, Path],
    stats_path: Union[str, Path],
    trace: FaderTraceModel,
) -> Path:
    """Write ``{stem}_host_controls.json`` next to a ``.ts`` export."""
    stats_data = load_attribute_stats(stats_path)
    discrete_class_labels = _discrete_labels_for_host(stats_data, stats_path)
    host = {
        "attribute_names": stats_data.get("attribute_names", []),
        "attribute_kinds": stats_data.get("attribute_kinds", {}),
        "continuous_attributes": stats_data.get("continuous_attributes", []),
        "discrete_attributes": stats_data.get("discrete_attributes", []),
        "discrete_num_classes": stats_data.get("discrete_num_classes", {}),
        "discrete_class_labels": discrete_class_labels,
        "min_max_features": {
            k: list(v)
            for k, v in stats_data.get("min_max_features", {}).items()
        },
        "latent_length": stats_data.get("latent_length"),
        "sr": stats_data.get("sr"),
        "content_latent_size": int(trace.content_latent_size.item()),
        "num_attributes": int(trace.num_attributes.item()),
        "decoder_latent_size": int(
            trace.content_latent_size.item() + trace.num_attributes.item()
        ),
        "nn_attributes": _nn_attribute_schema(stats_data),
    }
    host_out = Path(output_ts_path).with_suffix("") 
    host_out = Path(str(host_out) + "_host_controls.json")
    with open(host_out, "w") as f:
        json.dump(host, f, indent=2)
    return host_out


def _nn_attribute_schema(stats_data: dict) -> list:
    """Document nn~ attribute names exported by ScriptedFaderRAVE."""
    schema = [{"name": "attr_mode", "kind": "int", "default": 2,
               "doc": "0=extract+scale/override, 1=manual-only, 2=extract-only"}]
    names = stats_data.get("attribute_names", [])
    kinds = stats_data.get("attribute_kinds", {})
    min_max = stats_data.get("min_max_features", {})
    for name in names:
        kind = kinds.get(name, "continuous")
        if kind == "continuous":
            lo, hi = min_max.get(name, (0.0, 1.0))
            default = (lo + hi) * 0.5
        else:
            default = 0
        schema.append({"name": name, "kind": kind, "default": default})
        schema.append({"name": f"{name}_scale", "kind": "float", "default": 1.0})
        schema.append({"name": f"{name}_override", "kind": "bool", "default": False})
    return schema
