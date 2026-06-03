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

import json
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
from rave.fader.attributes import load_attribute_stats, resolve_stats_path
from rave.fader.export.trace_model import build_trace_model
from rave.fader.model import FaderRAVE
import cached_conv as cc

FLAGS = flags.FLAGS

flags.DEFINE_string("model", required=True, help="FaderRAVE run dir or ckpt")
flags.DEFINE_string("output_path", required=True, help="Output .ts path")
flags.DEFINE_string("stats_path", default=None, help="attribute_stats.yaml")
flags.DEFINE_string("db_path", default=None, help="LMDB to find stats")


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
    host = {
        "attribute_names": stats_data.get("attribute_names", []),
        "attribute_kinds": stats_data.get("attribute_kinds", {}),
        "continuous_attributes": stats_data.get("continuous_attributes", []),
        "discrete_attributes": stats_data.get("discrete_attributes", []),
        "discrete_num_classes": stats_data.get("discrete_num_classes", {}),
        "min_max_features": {
            k: list(v) for k, v in stats_data.get("min_max_features", {}).items()
        },
        "latent_length": stats_data.get("latent_length"),
        "sr": stats_data.get("sr"),
        "content_latent_size": trace.content_latent_size.item(),
        "num_attributes": trace.num_attributes.item(),
        "decoder_latent_size": int(trace.content_latent_size.item() + trace.num_attributes.item()),
    }
    host_out = os.path.splitext(FLAGS.output_path)[0] + "_host_controls.json"
    with open(host_out, "w") as f:
        json.dump(host, f, indent=2)
    logging.info("wrote host controls to %s", host_out)


if __name__ == "__main__":
    app.run(main)
