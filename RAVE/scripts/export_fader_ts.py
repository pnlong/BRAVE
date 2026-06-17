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

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import gin
import torch
from absl import app, flags, logging

import rave
from rave.fader.attributes import resolve_stats_path
from rave.fader.export.host_controls import write_host_controls_json
from rave.fader.export.trace_model import build_trace_model
from rave.fader.model import FaderRAVE
import cached_conv as cc

FLAGS = flags.FLAGS

flags.DEFINE_string("model", required=True, help="FaderRAVE run dir or ckpt")
flags.DEFINE_string("output_path", required=True, help="Output .ts path")
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
    "fader_config",
    default=None,
    help="Fader gin config (required with canonicalizer ckpt if not in run config)",
)


@torch.no_grad()
def main(argv):
    del argv
    cc.use_cached_conv(False)
    torch.set_default_dtype(torch.float32)

    model_path = FLAGS.model
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        logging.error("config not found for %s", model_path)
        return
    gin.parse_config_file(config_path)
    run = rave.core.search_for_run(model_path)
    if run is None:
        logging.error("checkpoint not found for %s", model_path)
        return

    model = FaderRAVE()
    model = model.load_from_checkpoint(run)
    model.eval()

    if FLAGS.waveform_canonicalizer or FLAGS.latent_canonicalizer:
        from rave.fader.canonicalizer_config import load_canonicalizer_onto_model

        ckpt = FLAGS.waveform_canonicalizer or FLAGS.latent_canonicalizer
        load_canonicalizer_onto_model(model, ckpt)

    stats = resolve_stats_path(FLAGS.db_path, FLAGS.stats_path)
    if stats is None:
        logging.error("attribute_stats.yaml not found")
        return

    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

    trace = build_trace_model(model, stats, deterministic=True)
    trace.eval()
    scripted = torch.jit.script(trace)
    os.makedirs(os.path.dirname(FLAGS.output_path) or ".", exist_ok=True)
    scripted.save(FLAGS.output_path)
    logging.info("saved TorchScript to %s", FLAGS.output_path)

    # --- Ship stats yaml beside .ts for runtime unnormalize ---
    stats_out = os.path.splitext(FLAGS.output_path)[0] + "_attribute_stats.yaml"
    shutil.copy2(stats, stats_out)
    logging.info("copied stats to %s", stats_out)

    stats_data = load_attribute_stats(stats)
    host_out = write_host_controls_json(FLAGS.output_path, stats, trace)
    logging.info("wrote host controls to %s", host_out)


if __name__ == "__main__":
    app.run(main)
