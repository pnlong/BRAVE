#!/usr/bin/env python3
"""
Unified export router for BRAVE / FaderRAVE / canonicalizer models.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python scripts/export_model.py \\
    --model runs/my_run \\
    --db_path /path/to/lmdb \\
    --output_dir exports/my_run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BRAVE_ROOT = Path(__file__).resolve().parents[1]
_RAVE_ROOT = _BRAVE_ROOT / "RAVE"
if str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

import rave
from absl import app, flags, logging
from rave.fader.export.bundle import print_max_copy_instructions
from rave.fader.export.load_for_export import is_fader_model
from rave.canonicalizer.export import resolve_canonicalizer_ckpt
from rave.fader.export.max_patch import write_vanilla_play_patch

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "Run dir or .ckpt path", required=True)
flags.DEFINE_string(
    "output_dir",
    default=None,
    help="Bundle output directory (default: exports/<run_name>)",
)
flags.DEFINE_string("db_path", default=None, help="LMDB path (Fader stats lookup)")
flags.DEFINE_string("stats_path", default=None, help="attribute_stats.yaml (Fader)")
flags.DEFINE_enum("host", "nn", ["nn", "ts", "h5"], help="Export target")
flags.DEFINE_string(
    "canonicalizer",
    default="auto",
    help="auto | none | path to canonicalizer .ckpt",
)
flags.DEFINE_string("waveform_canonicalizer", default=None, help="Explicit waveform ckpt")
flags.DEFINE_string("latent_canonicalizer", default=None, help="Explicit latent ckpt")
flags.DEFINE_bool(
    "streaming",
    default=True,
    help="Cached conv streaming for nn~ (default on; use --nostreaming for offline-only .ts)",
)
flags.DEFINE_integer("channels", default=None, help="Output channels (Fader nn export)")


def _default_output_dir(model_path: str) -> Path:
    run = rave.core.search_for_run(model_path)
    if run is not None:
        name = Path(rave.core.run_dir_from_checkpoint(run)).name
    else:
        name = Path(model_path).stem
    return _BRAVE_ROOT / "exports" / name


def _export_vanilla_nn(
    model_path: str,
    output_dir: Path,
    streaming: bool,
    canonicalizer_ckpt: str | None = None,
) -> Path:
    ts_path = output_dir / "model.ts"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_RAVE_ROOT / "scripts" / "export.py"),
        f"--run={model_path}",
        f"--output={output_dir}",
        "--name=model",
    ]
    if canonicalizer_ckpt:
        cmd.append(f"--canonicalizer_ckpt={canonicalizer_ckpt}")
    if not streaming:
        cmd.append("--nostreaming")
    logging.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_BRAVE_ROOT))
    streaming_ts = output_dir / "model_streaming.ts"
    if not ts_path.is_file() and streaming_ts.is_file():
        streaming_ts.rename(ts_path)
    if not ts_path.is_file():
        raise FileNotFoundError(f"expected export at {ts_path}")
    return ts_path


def _export_h5(model_path: str, output_dir: Path) -> Path:
    h5_path = output_dir / "model.h5"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_BRAVE_ROOT / "scripts" / "export_brave_plugin.py"),
        f"--model={model_path}",
        f"--output_path={h5_path}",
    ]
    logging.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_BRAVE_ROOT))
    return h5_path


def main(argv):
    del argv
    model_path = FLAGS.model
    if "..." in model_path or not model_path.strip():
        logging.error(
            "Invalid --model path %r. Use the full checkpoint path, not '...' placeholders.",
            model_path,
        )
        sys.exit(1)

    output_dir = Path(FLAGS.output_dir) if FLAGS.output_dir else _default_output_dir(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fader = is_fader_model(model_path)
    logging.info("model type: %s", "FaderRAVE" if fader else "RAVE")

    if FLAGS.db_path and not fader:
        logging.error(
            "Model at %r is not a FaderRAVE run (config not found). "
            "Check --model points to your fader run dir or best.ckpt.",
            model_path,
        )
        sys.exit(1)

    if FLAGS.host == "h5":
        out = _export_h5(model_path, output_dir)
        logging.info("plugin export: %s", out)
        return

    ts_path = output_dir / "model.ts"

    if fader:
        rave_scripts = str(_RAVE_ROOT / "scripts")
        if rave_scripts not in sys.path:
            sys.path.insert(0, rave_scripts)

        if FLAGS.host == "ts":
            from export_fader_ts import export_fader_ts

            export_fader_ts(
                model_path,
                str(ts_path),
                db_path=FLAGS.db_path,
                stats_path=FLAGS.stats_path,
                canonicalizer=FLAGS.canonicalizer,
                waveform_canonicalizer=FLAGS.waveform_canonicalizer,
                latent_canonicalizer=FLAGS.latent_canonicalizer,
            )
            logging.info("TorchScript export: %s", ts_path)
            return

        from export_fader_nn import export_fader_nn

        canon_ckpt = resolve_canonicalizer_ckpt(
            model_path,
            mode=FLAGS.canonicalizer,
            waveform_canonicalizer=FLAGS.waveform_canonicalizer,
            latent_canonicalizer=FLAGS.latent_canonicalizer,
        )
        if canon_ckpt:
            logging.info("embedding canonicalizer: %s", canon_ckpt)

        export_fader_nn(
            model_path,
            str(ts_path),
            db_path=FLAGS.db_path,
            stats_path=FLAGS.stats_path,
            streaming=FLAGS.streaming,
            channels=FLAGS.channels,
            canonicalizer=FLAGS.canonicalizer,
            waveform_canonicalizer=FLAGS.waveform_canonicalizer,
            latent_canonicalizer=FLAGS.latent_canonicalizer,
            write_play_patch=True,
        )
    else:
        if FLAGS.host == "ts":
            logging.error("plain TorchScript (--host ts) is only for FaderRAVE models")
            sys.exit(1)
        canon_ckpt = resolve_canonicalizer_ckpt(
            model_path,
            mode=FLAGS.canonicalizer,
            waveform_canonicalizer=FLAGS.waveform_canonicalizer,
            latent_canonicalizer=FLAGS.latent_canonicalizer,
        )
        if canon_ckpt:
            logging.info("embedding canonicalizer: %s", canon_ckpt)
        ts_path = _export_vanilla_nn(
            model_path, output_dir, FLAGS.streaming, canonicalizer_ckpt=canon_ckpt)
        write_vanilla_play_patch(output_dir / "play.maxpat", ts_name="model.ts")

    print_max_copy_instructions(output_dir)
    logging.info("done: %s", output_dir)


if __name__ == "__main__":
    app.run(main)
