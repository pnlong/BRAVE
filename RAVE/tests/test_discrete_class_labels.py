"""Tests for discrete class label resolution."""

import json
from pathlib import Path

import yaml

from rave.fader.discrete_class_labels import (
    load_class_labels_from_tags_yaml,
    resolve_discrete_class_labels,
)


def test_load_texture_class_labels_from_repo_yaml():
    root = Path(__file__).resolve().parents[3]
    yaml_path = (
        root
        / "dataset_exploration"
        / "fsd50k"
        / "configs"
        / "fader_texture_class_tags.yaml"
    )
    if not yaml_path.is_file():
        return
    labels = load_class_labels_from_tags_yaml(yaml_path, 10, attr_name="texture_class")
    assert labels[0] == "water_liquid"
    assert labels[9] == "tonal_alert"
    assert len(labels) == 10


def test_resolve_from_class_counts_json(tmp_path):
    tags = tmp_path / "tags.yaml"
    tags.write_text(yaml.safe_dump({"class_names": {0: "a", 1: "b", 2: "c"}}))
    counts = {
        "scheme": "texture_class",
        "tags_config": str(tags),
        "counts": {0: 5, 1: 3},
        "class_labels": ["a", "b", "c"],
    }
    (tmp_path / "texture_class_class_counts.json").write_text(json.dumps(counts))
    out = resolve_discrete_class_labels(
        tmp_path,
        ["texture_class"],
        {"texture_class": 3},
    )
    assert out["texture_class"] == ["a", "b", "c"]


def test_resolve_water_scene_builtin(tmp_path):
    out = resolve_discrete_class_labels(
        tmp_path,
        ["water_scene"],
        {"water_scene": 3},
    )
    assert out["water_scene"] == ["other", "storm", "coastal"]


def test_baked_stats_take_priority(tmp_path):
    stats = {
        "discrete_class_labels": {
            "texture_class": ["custom_a", "custom_b"],
        }
    }
    out = resolve_discrete_class_labels(
        tmp_path,
        ["texture_class"],
        {"texture_class": 2},
        stats=stats,
    )
    assert out["texture_class"] == ["custom_a", "custom_b"]
