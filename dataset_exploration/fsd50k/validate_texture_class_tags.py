#!/usr/bin/env python3
"""Verify fader_texture_class_tags.yaml covers all FSD50K ontology tags exactly once."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_FSD50K = Path(__file__).resolve().parent
CONFIG = _FSD50K / "configs" / "fader_texture_class_tags.yaml"
FREQ = _FSD50K / "artifacts" / "tag_frequencies.tsv"

CLASS_KEYS = [
    "music_percussion",
    "human_vocal_body",
    "weather_storm",
    "water_liquid",
    "air_wind",
    "fire_explosion",
    "nature_living",
    "domestic_indoor",
    "mechanical_machine",
    "transport_vehicle",
    "materials_impact",
    "tonal_alert",
]


def main() -> int:
    with CONFIG.open() as f:
        data = yaml.safe_load(f)

    ontology = [line.split("\t")[0] for line in FREQ.read_text().splitlines() if line.strip()]
    assigned: dict[str, str] = {}
    for key in CLASS_KEYS:
        for tag in data.get(key, []):
            if tag in assigned:
                print(f"DUPLICATE: {tag} in {assigned[tag]} and {key}")
                return 1
            assigned[tag] = key

    missing = [t for t in ontology if t not in assigned]
    extra = [t for t in assigned if t not in ontology]
    if missing:
        print(f"MISSING ({len(missing)}): {missing}")
        return 1
    if extra:
        print(f"EXTRA ({len(extra)}): {extra}")
        return 1

    print(f"OK: {len(ontology)} tags mapped to {len(CLASS_KEYS)} classes")
    freq = {line.split("\t")[0]: int(line.split("\t")[1]) for line in FREQ.read_text().splitlines() if line.strip()}
    names = data["class_names"]
    key_to_id = {k: i for i, k in enumerate(CLASS_KEYS)}
    totals: dict[int, int] = {i: 0 for i in range(len(CLASS_KEYS))}
    for tag, key in assigned.items():
        totals[key_to_id[key]] += freq[tag]
    print("\nTag-frequency mass by class (multi-label overcounts):")
    for key in CLASS_KEYS:
        cid = key_to_id[key]
        texture = "texture" if cid < 10 else "EXCLUDE"
        print(f"  {cid:2d} {names[cid]:22s} {totals[cid]:8d}  [{texture}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
