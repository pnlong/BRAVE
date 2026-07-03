"""Generate pre-wired Max 9 .maxpat files for nn~ Fader / RAVE bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

_NN_INLET = 0  # forward: audio + attribute messages share one inlet

# Single-column layout — signal flows top → bottom, controls to the right.
_COL_X = 40.0
_CTRL_X = 300.0
_PATCH_WIDTH = 680.0


def _box(
    obj_id: str,
    maxclass: str,
    rect: List[float],
    **extra: Any,
) -> Dict[str, Any]:
    b: Dict[str, Any] = {
        "id": obj_id,
        "maxclass": maxclass,
        "patching_rect": rect,
    }
    b.update(extra)
    return {"box": b}


def _line(src_id: str, src_out: int, dst_id: str, dst_in: int) -> Dict[str, Any]:
    return {
        "patchline": {
            "source": [src_id, src_out],
            "destination": [dst_id, dst_in],
        }
    }


def _patcher_root(boxes: List[Dict], lines: List[Dict], height: float = 600) -> Dict:
    return {
        "patcher": {
            "fileversion": 1,
            "appversion": {"major": 9, "minor": 0, "revision": 0},
            "classnamespace": "box",
            "rect": [50.0, 50.0, _PATCH_WIDTH, height],
            "gridsize": [15.0, 15.0],
            "boxes": boxes,
            "lines": lines,
            "dependency_cache": [],
            "autosave": 0,
        }
    }


def _section(boxes: List[Dict], next_id: Callable[[], str], y: float, title: str) -> None:
    boxes.append(
        _box(next_id(), "comment", [_COL_X, y, 520.0, 22.0], fontsize=12.0, fontface=1, text=title)
    )


def _hint(boxes: List[Dict], next_id: Callable[[], str], y: float, text: str) -> None:
    boxes.append(_box(next_id(), "comment", [_COL_X, y, 520.0, 28.0], text=text))


def _live_slider(
    obj_id: str,
    rect: List[float],
    *,
    longname: str,
    shortname: str,
    lo: float,
    hi: float,
    default: float,
) -> Dict[str, Any]:
    return _box(
        obj_id,
        "live.slider",
        rect,
        numinlets=1,
        numoutlets=2,
        outlettype=["", "float"],
        parameter_enable=1,
        saved_attribute_attributes={
            "valueof": {
                "parameter_longname": longname,
                "parameter_shortname": shortname,
                "parameter_type": 0,
                "parameter_mmin": float(lo),
                "parameter_mmax": float(hi),
                "parameter_initial": [float(default)],
            }
        },
    )


def _live_menu(
    obj_id: str,
    rect: List[float],
    *,
    longname: str,
    shortname: str,
    items: List[str],
    default_index: int = 0,
) -> Dict[str, Any]:
    return _box(
        obj_id,
        "live.menu",
        rect,
        numinlets=1,
        numoutlets=3,
        outlettype=["int", "", "float"],
        parameter_enable=1,
        saved_attribute_attributes={
            "valueof": {
                "parameter_longname": longname,
                "parameter_shortname": shortname,
                "parameter_type": 2,
                "parameter_enum": items,
                "parameter_initial": [int(default_index)],
            }
        },
    )


def _live_toggle(
    obj_id: str,
    rect: List[float],
    *,
    longname: str,
    shortname: str,
    labels: tuple[str, str] = ("off", "on"),
    default: int = 0,
) -> Dict[str, Any]:
    return _box(
        obj_id,
        "live.toggle",
        rect,
        numinlets=1,
        numoutlets=1,
        outlettype=[""],
        parameter_enable=1,
        saved_attribute_attributes={
            "valueof": {
                "parameter_longname": longname,
                "parameter_shortname": shortname,
                "parameter_type": 2,
                "parameter_enum": list(labels),
                "parameter_initial": [int(default)],
            }
        },
    )


def _nn_tilde_box(obj_id: str, rect: List[float], ts_name: str) -> Dict[str, Any]:
    return _box(
        obj_id,
        "newobj",
        rect,
        text=f"nn~ {ts_name} forward 512",
        numinlets=1,
        numoutlets=1,
        outlettype=["signal"],
    )


def _wire_extract_only(
    boxes: List[Dict],
    lines: List[Dict],
    next_id: Callable[[], str],
    nn_id: str,
    *,
    y: float,
) -> None:
    """Hidden loadbang: force attr_mode=2 (extract only). No UI menu."""
    lb = next_id()
    prep = next_id()
    val = next_id()
    boxes.extend([
        _box(lb, "newobj", [_CTRL_X + 180.0, y, 60.0, 22.0], text="loadbang"),
        _box(prep, "newobj", [_CTRL_X + 250.0, y, 120.0, 22.0], text="prepend set attr_mode"),
        _box(val, "message", [_CTRL_X + 180.0, y + 28.0, 30.0, 22.0], text="2"),
    ])
    lines.extend([
        _line(lb, 0, val, 0),
        _line(val, 0, prep, 0),
        _line(prep, 0, nn_id, _NN_INLET),
    ])


def _build_play_patch(
    boxes: List[Dict],
    lines: List[Dict],
    next_id: Callable[[], str],
    *,
    title: str,
    ts_name: str,
    extract_only: bool = False,
) -> None:
    """Single-column patch: source → level → nn~ → model gain → output."""

    y = 20.0
    boxes.append(
        _box(next_id(), "comment", [_COL_X, y, 520.0, 22.0], fontsize=12.0, text=title)
    )
    y += 35.0
    hint = (
        "44100 Hz · extract-only · ~3 s warmup · timbral attrs use neutral defaults · click ezdac~"
        if extract_only
        else "44100 Hz · click ezdac~ for audio on"
    )
    _hint(boxes, next_id, y, hint)
    y += 40.0

    # --- 1 · source ---
    _section(boxes, next_id, y, "1 · INPUT")
    y += 30.0
    src_menu = next_id()
    adc_id = next_id()
    in_meter_id = next_id()
    sfplay_id = next_id()
    open_btn = next_id()
    open_dlg = next_id()
    open_trig = next_id()
    open_path = next_id()
    open_play = next_id()
    drop_id = next_id()
    drop_trig = next_id()
    play_toggle = next_id()
    boxes.extend([
        _live_menu(
            src_menu,
            [_COL_X, y, 140.0, 22.0],
            longname="audio_source",
            shortname="source",
            items=["live in", "file"],
            default_index=0,
        ),
        _box(adc_id, "newobj", [_COL_X, y + 35.0, 57.0, 22.0], text="adc~ 1"),
        _box(in_meter_id, "meter~", [_CTRL_X, y + 35.0, 80.0, 22.0]),
        _box(
            open_btn,
            "textbutton",
            [_COL_X, y + 70.0, 120.0, 22.0],
            text="Choose file…",
            fontsize=11.0,
        ),
        _box(open_dlg, "newobj", [_COL_X + 140.0, y + 70.0, 80.0, 22.0], text="opendialog"),
        _box(open_trig, "newobj", [_COL_X + 240.0, y + 70.0, 40.0, 22.0], text="t b l"),
        _box(open_path, "message", [_COL_X + 300.0, y + 70.0, 55.0, 22.0], text="open $1"),
        _box(open_play, "message", [_COL_X + 300.0, y + 95.0, 30.0, 22.0], text="1"),
        _box(drop_id, "dropfile", [_COL_X, y + 100.0, 120.0, 22.0]),
        _box(drop_trig, "newobj", [_COL_X + 140.0, y + 100.0, 40.0, 22.0], text="t b l"),
        _live_toggle(
            play_toggle,
            [_COL_X + 240.0, y + 70.0, 50.0, 22.0],
            longname="file_play",
            shortname="play",
            labels=("off", "on"),
            default=1,
        ),
        _box(sfplay_id, "newobj", [_COL_X + 210.0, y + 100.0, 95.0, 22.0], text="sfplay~ 1 @loop 1"),
    ])
    y += 140.0

    # --- 2 · level ---
    _section(boxes, next_id, y, "2 · LEVEL")
    y += 30.0
    sel_id = next_id()
    pre_meter_id = next_id()
    in_gain_slider = next_id()
    in_gain_id = next_id()
    src_idx = next_id()
    boxes.extend([
        _box(sel_id, "newobj", [_COL_X, y, 65.0, 22.0], text="selector~ 2"),
        _box(pre_meter_id, "meter~", [_CTRL_X, y, 80.0, 22.0]),
        _live_slider(
            in_gain_slider,
            [_CTRL_X + 100.0, y - 2.0, 120.0, 22.0],
            longname="input_gain",
            shortname="in gain",
            lo=0.0,
            hi=2.0,
            default=1.0,
        ),
        _box(in_gain_id, "newobj", [_COL_X, y + 40.0, 45.0, 22.0], text="*~ 1"),
        _box(src_idx, "newobj", [_COL_X + 80.0, y + 40.0, 30.0, 22.0], text="+ 1"),
    ])
    y += 80.0

    # --- 3 · model ---
    _section(boxes, next_id, y, "3 · MODEL (nn~)")
    y += 30.0
    nn_id = next_id()
    model_meter_id = next_id()
    model_gain_slider = next_id()
    model_gain_id = next_id()
    boxes.extend([
        _nn_tilde_box(nn_id, [_COL_X, y, 200.0, 22.0], ts_name),
        _box(model_meter_id, "meter~", [_CTRL_X, y, 80.0, 22.0]),
        _live_slider(
            model_gain_slider,
            [_CTRL_X + 100.0, y - 2.0, 120.0, 22.0],
            longname="model_gain",
            shortname="model gain",
            lo=0.0,
            hi=8.0,
            default=3.0,
        ),
        _box(model_gain_id, "newobj", [_COL_X, y + 40.0, 45.0, 22.0], text="*~ 1"),
    ])
    if extract_only:
        _wire_extract_only(boxes, lines, next_id, nn_id, y=y + 40.0)
    y += 90.0

    # --- 4 · output ---
    _section(boxes, next_id, y, "4 · OUTPUT (click ezdac~)")
    y += 30.0
    path_toggle = next_id()
    path_idx = next_id()
    path_sel = next_id()
    dac_id = next_id()
    out_meter_id = next_id()
    boxes.extend([
        _live_toggle(
            path_toggle,
            [_COL_X, y, 160.0, 22.0],
            longname="output_path",
            shortname="path",
            labels=("direct in", "nn~ model"),
            default=1,
        ),
        _box(path_idx, "newobj", [_COL_X + 180.0, y, 30.0, 22.0], text="+ 1"),
        _box(path_sel, "newobj", [_COL_X, y + 40.0, 65.0, 22.0], text="selector~ 2"),
        _box(dac_id, "ezdac~", [_COL_X, y + 90.0, 45.0, 45.0]),
        _box(out_meter_id, "meter~", [_CTRL_X, y + 98.0, 80.0, 22.0]),
    ])
    y += 150.0

    # --- startup (one loadbang) ---
    lb = next_id()
    t = next_id()
    m_src = next_id()
    m_in_gain = next_id()
    m_model_gain = next_id()
    m_path = next_id()
    m_file_sel = next_id()
    boxes.extend([
        _box(lb, "newobj", [_CTRL_X + 180.0, 20.0, 60.0, 22.0], text="loadbang"),
        _box(t, "newobj", [_CTRL_X + 250.0, 20.0, 30.0, 22.0], text="t b b b b"),
        _box(m_src, "message", [_CTRL_X + 300.0, 20.0, 30.0, 22.0], text="1"),
        _box(m_in_gain, "message", [_CTRL_X + 300.0, 45.0, 30.0, 22.0], text="1."),
        _box(m_model_gain, "message", [_CTRL_X + 300.0, 70.0, 30.0, 22.0], text="3."),
        _box(m_path, "message", [_CTRL_X + 300.0, 95.0, 30.0, 22.0], text="2"),
        _box(m_file_sel, "message", [_CTRL_X + 340.0, 70.0, 30.0, 22.0], text="2"),
    ])

    # --- signal routing (vertical, minimal crossings) ---
    lines.extend([
        _line(adc_id, 0, sel_id, 1),
        _line(sfplay_id, 0, sel_id, 2),
        _line(src_menu, 0, src_idx, 0),
        _line(src_idx, 0, sel_id, 0),
        _line(sel_id, 0, pre_meter_id, 0),
        _line(sel_id, 0, in_gain_id, 0),
        _line(in_gain_slider, 1, in_gain_id, 1),
        _line(in_gain_id, 0, nn_id, _NN_INLET),
        _line(nn_id, 0, model_gain_id, 0),
        _line(model_gain_slider, 1, model_gain_id, 1),
        _line(model_gain_id, 0, model_meter_id, 0),
        _line(in_gain_id, 0, path_sel, 1),
        _line(model_gain_id, 0, path_sel, 2),
        _line(path_toggle, 0, path_idx, 0),
        _line(path_idx, 0, path_sel, 0),
        _line(path_sel, 0, dac_id, 0),
        _line(path_sel, 0, dac_id, 1),
        _line(path_sel, 0, out_meter_id, 0),
        # file
        _line(open_btn, 0, open_dlg, 0),
        _line(open_dlg, 0, open_trig, 0),
        _line(open_trig, 1, open_path, 0),
        _line(open_path, 0, sfplay_id, 0),
        _line(open_trig, 0, open_play, 0),
        _line(open_play, 0, sfplay_id, 0),
        _line(open_trig, 0, m_file_sel, 0),
        _line(drop_id, 0, drop_trig, 0),
        _line(drop_trig, 1, open_path, 0),
        _line(drop_trig, 0, open_play, 0),
        _line(drop_trig, 0, m_file_sel, 0),
        _line(play_toggle, 0, sfplay_id, 0),
        # startup
        _line(lb, 0, t, 0),
        _line(t, 0, m_src, 0),
        _line(m_src, 0, sel_id, 0),
        _line(t, 0, m_in_gain, 0),
        _line(m_in_gain, 0, in_gain_id, 1),
        _line(t, 0, m_model_gain, 0),
        _line(m_model_gain, 0, model_gain_id, 1),
        _line(t, 0, m_path, 0),
        _line(m_path, 0, path_sel, 0),
        _line(m_file_sel, 0, sel_id, 0),
    ])


def write_vanilla_play_patch(output_path: Union[str, Path], ts_name: str = "model.ts") -> Path:
    output_path = Path(output_path)
    boxes: List[Dict] = []
    lines: List[Dict] = []
    obj_idx = 1

    def next_id() -> str:
        nonlocal obj_idx
        oid = f"obj-{obj_idx}"
        obj_idx += 1
        return oid

    _build_play_patch(
        boxes,
        lines,
        next_id,
        title="BRAVE nn~ — set Max Audio to 44100 Hz",
        ts_name=ts_name,
        extract_only=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_patcher_root(boxes, lines, height=620), f, indent=2)
    return output_path


def write_fader_play_patch(
    host_controls_path: Union[str, Path],
    output_path: Union[str, Path],
    ts_name: str = "model.ts",
) -> Path:
    """Extract-only Fader patch — no attribute sliders or mode menu."""
    del host_controls_path  # sidecar kept for docs; patch is minimal
    output_path = Path(output_path)
    boxes: List[Dict] = []
    lines: List[Dict] = []
    obj_idx = 1

    def next_id() -> str:
        nonlocal obj_idx
        oid = f"obj-{obj_idx}"
        obj_idx += 1
        return oid

    _build_play_patch(
        boxes,
        lines,
        next_id,
        title="BRAVE Fader — attributes extracted from your audio (extract only)",
        ts_name=ts_name,
        extract_only=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_patcher_root(boxes, lines, height=620), f, indent=2)
    return output_path
