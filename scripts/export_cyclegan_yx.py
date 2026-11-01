#!/usr/bin/env python3
"""Export the CycleGAN return direction: Enc_Y → W_yx → Dec_X.

``scripts/export_model.py`` still writes Enc_X → W_xy → Dec_Y. This script
writes the other warp from the same ``cyclegan_latent.ckpt`` into its own
nn~ bundle (``model.ts`` + ``play.maxpat``).

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python scripts/export_cyclegan_yx.py \\
    --model /path/to/run/cyclegan_latent.ckpt \\
    --output_dir exports/sax_to_drums \\
    --title "CycleGAN sax → drums — set Max Audio to 44100 Hz"
"""

from __future__ import annotations

import sys
from pathlib import Path

_BRAVE_ROOT = Path(__file__).resolve().parents[1]
_RAVE_ROOT = _BRAVE_ROOT / "RAVE"
if str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

from absl import app, flags, logging
from rave.canonicalizer.export.cyclegan_nn import export_cyclegan_yx_nn
from rave.canonicalizer.export.resolve import resolve_cyclegan_ckpt
from rave.fader.export.bundle import print_max_copy_instructions
from rave.fader.export.max_patch import write_vanilla_play_patch

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "CycleGAN run dir or cyclegan_latent.ckpt", required=True)
flags.DEFINE_string(
    "output_dir",
    default=None,
    help="Bundle directory (default: exports/<run_name>_yx)",
)
flags.DEFINE_string(
    "title",
    default="CycleGAN Y→X — set Max Audio to 44100 Hz",
    help="Comment written into play.maxpat",
)
flags.DEFINE_bool(
    "streaming",
    default=True,
    help="Cached conv streaming for nn~ (default on)",
)
flags.DEFINE_bool(
    "max_patch",
    default=True,
    help="Write play.maxpat next to model.ts",
)


def main(argv):
    del argv
    model_path = FLAGS.model
    cyclegan_ckpt = resolve_cyclegan_ckpt(model_path)
    if cyclegan_ckpt is None:
        logging.error("No cyclegan_latent.ckpt found for %s", model_path)
        sys.exit(1)

    if FLAGS.output_dir:
        output_dir = Path(FLAGS.output_dir)
    else:
        output_dir = _BRAVE_ROOT / "exports" / f"{Path(cyclegan_ckpt).parent.name}_yx"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("CycleGAN Y→X export from %s", cyclegan_ckpt)
    ts_path = output_dir / "model.ts"
    export_cyclegan_yx_nn(cyclegan_ckpt, ts_path, streaming=FLAGS.streaming)
    if FLAGS.max_patch:
        write_vanilla_play_patch(
            output_dir / "play.maxpat",
            ts_name="model.ts",
            title=FLAGS.title,
            default_source_file=True,
            sidechain_toggle=True,
        )
        print_max_copy_instructions(output_dir)
    logging.info("done: %s", output_dir)


if __name__ == "__main__":
    app.run(main)
