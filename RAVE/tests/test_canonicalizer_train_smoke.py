"""
Optional integration training smoke test.

Set RUN_CANONICALIZER_TRAIN=1 and provide:
  FADER_CKPT, DB_PATH, FADER_CONFIG, OOD_PATH

Example:
  RUN_CANONICALIZER_TRAIN=1 \\
  FADER_CKPT=runs/birdsong.ckpt \\
  DB_PATH=/data/birdsong_lmdb \\
  FADER_CONFIG=configs/brave_fader_pitched.gin \\
  OOD_PATH=ood_samples \\
  pytest RAVE/tests/test_canonicalizer_train_smoke.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BRAVE = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.environ.get("RUN_CANONICALIZER_TRAIN") != "1",
    reason="Set RUN_CANONICALIZER_TRAIN=1 with FADER_CKPT, DB_PATH, FADER_CONFIG, OOD_PATH",
)
def test_train_canonicalizer_smoke():
    env = os.environ.copy()
    required = ["FADER_CKPT", "DB_PATH", "FADER_CONFIG", "OOD_PATH"]
    for k in required:
        assert k in env, f"missing env {k}"

    cmd = [
        sys.executable,
        str(_BRAVE / "RAVE" / "scripts" / "train_canonicalizer.py"),
        "--config",
        "configs/brave_canonicalizer.gin",
        "--fader_config",
        env["FADER_CONFIG"],
        "--ckpt",
        env["FADER_CKPT"],
        "--db_path",
        env["DB_PATH"],
        "--ood_path",
        env["OOD_PATH"],
        "--canonicalizer_type",
        "waveform",
        "--name",
        "pytest_canon_smoke",
        "--smoke_test",
    ]
    subprocess.run(cmd, cwd=str(_BRAVE), check=True, env=env)
