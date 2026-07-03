"""Tests for auto-generated Max play patches."""

import json
import importlib.util
from pathlib import Path

import pytest


def _load_max_patch_module():
    """Import max_patch without pulling in rave (avoids heavy deps in CI)."""
    path = Path(__file__).resolve().parents[1] / "rave" / "fader" / "export" / "max_patch.py"
    spec = importlib.util.spec_from_file_location("max_patch", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_vanilla_play_patch_json(tmp_path):
    max_patch = _load_max_patch_module()
    out = max_patch.write_vanilla_play_patch(tmp_path / "play.maxpat", ts_name="model.ts")
    data = json.loads(out.read_text())
    assert "patcher" in data
    boxes = data["patcher"]["boxes"]
    texts = [b["box"].get("text", "") for b in boxes if b["box"].get("maxclass") == "newobj"]
    assert any("nn~ model.ts forward 512" in t for t in texts)
    assert any(t == "selector~ 2" for t in texts)
    assert any(t == "opendialog" for t in texts)
    assert sum(1 for b in boxes if b["box"].get("maxclass") == "meter~") >= 3
    classes = {b["box"]["maxclass"] for b in boxes}
    assert "textbutton" in classes
    assert any(t == "adc~ 1" for t in texts)
    assert not any("prepend set attr_mode" in t for t in texts)


def test_fader_play_patch_json(tmp_path):
    max_patch = _load_max_patch_module()
    host = {
        "attribute_names": ["rms", "texture_class"],
        "attribute_kinds": {"rms": "continuous", "texture_class": "discrete"},
        "min_max_features": {"rms": [0.0, 0.1]},
        "discrete_num_classes": {"texture_class": 4},
        "discrete_class_labels": {
            "texture_class": [
                "water_liquid",
                "air_wind",
                "weather_storm",
                "fire_explosion",
            ],
        },
    }
    host_path = tmp_path / "model_host_controls.json"
    host_path.write_text(json.dumps(host))
    out = max_patch.write_fader_play_patch(host_path, tmp_path / "play.maxpat")
    data = json.loads(out.read_text())
    lines = data["patcher"]["lines"]
    assert len(lines) >= 3
    boxes = data["patcher"]["boxes"]
    texts = [b["box"].get("text", "") for b in boxes if b["box"].get("maxclass") == "newobj"]
    assert any("prepend set attr_mode" in t for t in texts)
    assert not any("prepend set rms" in t for t in texts)
    classes = {b["box"]["maxclass"] for b in boxes}
    assert "live.slider" in classes
    assert "live.menu" in classes  # source menu only
    assert "dropfile" in classes
    nn_boxes = [
        b["box"] for b in boxes
        if b["box"].get("maxclass") == "newobj"
        and b["box"].get("text", "").startswith("nn~")
    ]
    assert nn_boxes
    assert nn_boxes[0].get("numinlets") == 1
    nn_id = nn_boxes[0]["id"]
    nn_dest = [
        ln for ln in lines
        if ln["patchline"]["destination"][0] == nn_id
    ]
    assert all(ln["patchline"]["destination"][1] == 0 for ln in nn_dest)
    assert len(nn_dest) == 2  # audio in + hidden attr_mode loadbang
