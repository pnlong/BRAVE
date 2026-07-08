"""
Export FaderRAVE to TorchScript with attribute concat API.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/export_fader_ts.py \\
    --model runs/brave_fader_run \\
    --stats_path /path/to/lmdb/attribute_stats.yaml \\
    --output_path exports/fader.ts
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
from rave.fader.export.host_controls import write_host_controls_json
from rave.fader.export.load_for_export import (
    load_fader_for_export,
    strip_weight_norm,
)
from rave.canonicalizer.export import resolve_canonicalizer_ckpt
from rave.fader.export.trace_model import build_trace_model


def _define_flags():
    flags.DEFINE_string("model", None, "FaderRAVE run dir or ckpt", required=True)
    flags.DEFINE_string("output_path", None, "Output .ts path", required=True)
    flags.DEFINE_string("stats_path", default=None, help="attribute_stats.yaml")
    flags.DEFINE_string("db_path", default=None, help="LMDB to find stats")
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
    flags.DEFINE_string(
        "fader_config",
        default=None,
        help="Fader gin config (required with canonicalizer ckpt if not in run config)",
    )


@torch.no_grad()
def export_fader_ts(
    model_path: str,
    output_path: str,
    *,
    db_path: Optional[str] = None,
    stats_path: Optional[str] = None,
    canonicalizer: str = "none",
    waveform_canonicalizer: Optional[str] = None,
    latent_canonicalizer: Optional[str] = None,
) -> Path:
    cc.use_cached_conv(False)
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
    scripted = torch.jit.script(trace)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    logging.info("saved TorchScript to %s", output_path)

    stats_out = Path(str(output_path.with_suffix("")) + "_attribute_stats.yaml")
    shutil.copy2(stats, stats_out)
    logging.info("copied stats to %s", stats_out)

    write_host_controls_json(output_path, stats, trace)
    return output_path


@torch.no_grad()
def main(argv):
    del argv
    export_fader_ts(
        flags.FLAGS.model,
        flags.FLAGS.output_path,
        db_path=flags.FLAGS.db_path,
        stats_path=flags.FLAGS.stats_path,
        canonicalizer=flags.FLAGS.canonicalizer,
        waveform_canonicalizer=flags.FLAGS.waveform_canonicalizer,
        latent_canonicalizer=flags.FLAGS.latent_canonicalizer,
    )


if __name__ == "__main__":
    _define_flags()
    app.run(main)
