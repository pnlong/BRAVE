#!/usr/bin/env python3
"""Merge attribute_stats.yaml from two LMDBs for joint Fader training.

Example:
  python RAVE/scripts/merge_attribute_stats.py \\
    --db_path_x /path/tap_lmdb --db_path_y /path/water_lmdb \\
    --output /path/tap_lmdb/attribute_stats_merged.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rave.fader.attributes import (
    load_attribute_stats,
    merge_attribute_stats,
    resolve_stats_path,
    save_attribute_stats,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db_path_x", required=True)
    p.add_argument("--db_path_y", required=True)
    p.add_argument(
        "--output",
        default="",
        help="Output yaml (default: <db_path_x>/attribute_stats_merged.yaml)",
    )
    args = p.parse_args()

    path_x = resolve_stats_path(args.db_path_x)
    path_y = resolve_stats_path(args.db_path_y)
    if path_x is None:
        raise SystemExit(f"No attribute_stats.yaml under {args.db_path_x}")
    if path_y is None:
        raise SystemExit(f"No attribute_stats.yaml under {args.db_path_y}")

    st = merge_attribute_stats(
        load_attribute_stats(path_x),
        load_attribute_stats(path_y),
    )
    out = Path(args.output) if args.output else (
        Path(args.db_path_x) / "attribute_stats_merged.yaml"
    )
    save_attribute_stats(
        out,
        attribute_names=st["attribute_names"],
        min_max_features=st["min_max_features"],
        bin_values=st["bin_values"],
        nb_bins=st["nb_bins"],
        latent_length=st["latent_length"],
        sr=st["sr"],
        continuous_attributes=st["continuous_attributes"],
        discrete_attributes=st["discrete_attributes"],
        attribute_kinds=st["attribute_kinds"],
        discrete_num_classes=st.get("discrete_num_classes"),
        discrete_class_labels=st.get("discrete_class_labels"),
        version=st.get("version", 1),
        split=st.get("split"),
    )
    print(f"Wrote {out}")
    print(f"  attributes={st['attribute_names']}")


if __name__ == "__main__":
    main()
