"""
Export FaderRAVE to nn~-compatible TorchScript with attribute knobs.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/export_fader_nn.py \\
    --model runs/brave_fader_texture \\
    --db_path /path/to/lmdb \\
    --output_path exports/fader_texture.ts \\
    --streaming
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

import cached_conv as cc
import rave
from rave.fader.attributes import load_attribute_stats, resolve_stats_path
from rave.fader.export.host_controls import write_host_controls_json
from rave.fader.export.nn_module import ScriptedFaderRAVE
from rave.fader.export.trace_model import build_trace_model
from rave.fader.model import FaderRAVE

FLAGS = flags.FLAGS

flags.DEFINE_string("model", required=True, help="FaderRAVE run dir or ckpt")
flags.DEFINE_string("output_path", required=True, help="Output .ts path")
flags.DEFINE_string("stats_path", default=None, help="attribute_stats.yaml")
flags.DEFINE_string("db_path", default=None, help="LMDB to find stats")
flags.DEFINE_bool("streaming", default=False, help="Enable cached conv streaming")
flags.DEFINE_integer("channels", default=None, help="Output channels for export")


@torch.no_grad()
def main(argv):
    del argv
    cc.use_cached_conv(FLAGS.streaming)
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

    stats = resolve_stats_path(FLAGS.db_path, FLAGS.stats_path)
    if stats is None:
        logging.error("attribute_stats.yaml not found")
        return

    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)

    trace = build_trace_model(model, stats, deterministic=True)
    trace.eval()

    stats_data = load_attribute_stats(stats)
    scripted = ScriptedFaderRAVE(
        core=trace,
        min_max_features=stats_data["min_max_features"],
        continuous_attributes=stats_data.get("continuous_attributes", []),
        n_channels=model.n_channels,
        target_channels=FLAGS.channels or model.n_channels,
    )
    scripted.eval()

    x = torch.zeros(1, model.n_channels, 2**14)
    scripted(x)

    os.makedirs(os.path.dirname(FLAGS.output_path) or ".", exist_ok=True)
    scripted.export_to_ts(FLAGS.output_path)
    logging.info("saved nn~ TorchScript to %s", FLAGS.output_path)

    stats_out = os.path.splitext(FLAGS.output_path)[0] + "_attribute_stats.yaml"
    shutil.copy2(stats, stats_out)
    logging.info("copied stats to %s", stats_out)

    host_out = write_host_controls_json(FLAGS.output_path, stats, trace)
    logging.info("wrote host controls to %s", host_out)


if __name__ == "__main__":
    app.run(main)
