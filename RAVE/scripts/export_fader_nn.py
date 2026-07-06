"""
Export FaderRAVE to nn~-compatible TorchScript with attribute knobs.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/export_fader_nn.py \\
    --model runs/brave_fader_texture \\
    --db_path /path/to/lmdb \\
    --output_path exports/fader_texture/model.ts
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import torch
from absl import app, flags, logging

import cached_conv as cc
from rave.fader.attributes import load_attribute_stats
from rave.fader.export.host_controls import write_host_controls_json
from rave.fader.export.load_for_export import (
    load_fader_for_export,
    resolve_canonicalizer_ckpt,
    strip_weight_norm,
)
from rave.fader.export.max_patch import write_fader_play_patch
from rave.fader.export.nn_module import create_scripted_fader_rave
from rave.fader.export.trace_model import build_trace_model


def _define_flags():
    flags.DEFINE_string("model", None, "FaderRAVE run dir or ckpt", required=True)
    flags.DEFINE_string("output_path", None, "Output .ts path", required=True)
    flags.DEFINE_string("stats_path", default=None, help="attribute_stats.yaml")
    flags.DEFINE_string("db_path", default=None, help="LMDB to find stats")
    flags.DEFINE_bool(
        "streaming",
        default=True,
        help="Cached conv streaming for nn~ (default on; --nostreaming for offline-only)",
    )
    flags.DEFINE_integer("channels", default=None, help="Output channels for export")
    flags.DEFINE_string(
        "waveform_canonicalizer",
        default=None,
        help="waveform_canonicalizer.ckpt to embed in export",
    )
    flags.DEFINE_string(
        "latent_canonicalizer",
        default=None,
        help="latent_canonicalizer.ckpt to embed in export",
    )
    flags.DEFINE_string(
        "canonicalizer",
        default="none",
        help="auto | none | path to canonicalizer ckpt",
    )
    flags.DEFINE_bool(
        "write_play_patch",
        default=False,
        help="Write play.maxpat beside the .ts (bundle mode)",
    )


@torch.no_grad()
def export_fader_nn(
    model_path: str,
    output_path: str,
    *,
    db_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    streaming: bool = True,
    channels: Optional[int] = None,
    canonicalizer: str = "none",
    waveform_canonicalizer: Optional[str] = None,
    latent_canonicalizer: Optional[str] = None,
    write_play_patch: bool = False,
) -> Path:
    cc.use_cached_conv(streaming)
    if streaming:
        logging.info(
            "streaming=True (default): Fader nn~ may go silent after ~1s at "
            "forward 512; use forward 8192 in play.maxpat if that happens",
        )
    else:
        logging.warning(
            "streaming=False: nn~ realtime may click; omit --nostreaming for Max use",
        )
    torch.set_default_dtype(torch.float32)

    canon_ckpt = resolve_canonicalizer_ckpt(
        model_path,
        mode=canonicalizer,
        waveform_canonicalizer=waveform_canonicalizer,
        latent_canonicalizer=latent_canonicalizer,
    )
    model, _run, stats = load_fader_for_export(
        model_path,
        db_path=db_path,
        stats_path=stats_path,
        canonicalizer_ckpt=canon_ckpt,
    )
    strip_weight_norm(model)

    trace = build_trace_model(model, stats, deterministic=True)
    trace.eval()

    stats_data = load_attribute_stats(stats)
    scripted = create_scripted_fader_rave(
        core=trace,
        min_max_features=stats_data["min_max_features"],
        continuous_attributes=stats_data.get("continuous_attributes", []),
        n_channels=model.n_channels,
        target_channels=channels or model.n_channels,
        generated_module_path=Path(output_path).parent / "_scripted_fader_rave.py",
    )
    scripted.eval()

    x = torch.zeros(1, model.n_channels, 2**14)
    scripted(x)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.export_to_ts(str(output_path))
    logging.info("saved nn~ TorchScript to %s", output_path)

    stats_out = Path(str(output_path.with_suffix("")) + "_attribute_stats.yaml")
    shutil.copy2(stats, stats_out)
    logging.info("copied stats to %s", stats_out)

    host_out = write_host_controls_json(output_path, stats, trace)
    logging.info("wrote host controls to %s", host_out)

    if write_play_patch:
        play_path = output_path.parent / "play.maxpat"
        write_fader_play_patch(host_out, play_path, ts_name=output_path.name)
        logging.info("wrote %s", play_path)

    return output_path


@torch.no_grad()
def main(argv):
    del argv
    export_fader_nn(
        flags.FLAGS.model,
        flags.FLAGS.output_path,
        db_path=flags.FLAGS.db_path,
        stats_path=flags.FLAGS.stats_path,
        streaming=flags.FLAGS.streaming,
        channels=flags.FLAGS.channels,
        canonicalizer=flags.FLAGS.canonicalizer,
        waveform_canonicalizer=flags.FLAGS.waveform_canonicalizer,
        latent_canonicalizer=flags.FLAGS.latent_canonicalizer,
        write_play_patch=flags.FLAGS.write_play_patch,
    )


if __name__ == "__main__":
    _define_flags()
    app.run(main)
