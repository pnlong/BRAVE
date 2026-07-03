"""Write self-contained nn~ export bundles for Max."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

from .host_controls import write_host_controls_json
from .max_patch import write_fader_play_patch, write_vanilla_play_patch


def bundle_ts_stem(ts_path: Union[str, Path]) -> Path:
    """Sidecar stem: ``model.ts`` -> ``model``."""
    return Path(ts_path).with_suffix("")


def finalize_nn_bundle(
    ts_path: Union[str, Path],
    stats_src: Optional[Union[str, Path]] = None,
    *,
    is_fader: bool,
    trace=None,
    play_patch_name: str = "play.maxpat",
) -> Path:
    """
    Copy stats, write host controls (Fader), and generate ``play.maxpat``.

    Returns path to ``play.maxpat``.
    """
    ts_path = Path(ts_path)
    out_dir = ts_path.parent
    stem = bundle_ts_stem(ts_path)

    if stats_src is not None and Path(stats_src).is_file():
        stats_out = Path(str(stem) + "_attribute_stats.yaml")
        shutil.copy2(stats_src, stats_out)

    host_path: Optional[Path] = None
    if is_fader and trace is not None and stats_src is not None:
        host_path = write_host_controls_json(ts_path, stats_src, trace)
    elif is_fader:
        candidate = Path(str(stem) + "_host_controls.json")
        if candidate.is_file():
            host_path = candidate

    play_path = out_dir / play_patch_name
    if is_fader and host_path is not None:
        write_fader_play_patch(host_path, play_path, ts_name=ts_path.name)
    else:
        write_vanilla_play_patch(play_path, ts_name=ts_path.name)
    return play_path


def print_max_copy_instructions(bundle_dir: Union[str, Path]) -> None:
    bundle_dir = Path(bundle_dir).resolve()
    mac_dest = "~/Documents/Max 9/Packages/nn_tilde/models/"
    print("\n" + "=" * 60)
    print(f"Bundle ready: {bundle_dir}/")
    print(f"  scp -r {bundle_dir} YOU@YOUR-MAC:{mac_dest}")
    print("  Then open play.maxpat in Max 9 (audio I/O at 44100 Hz).")
    print("=" * 60 + "\n")
