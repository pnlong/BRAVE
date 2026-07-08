"""
Optional integration training smoke test.

Set RUN_CANONICALIZER_TRAIN=1 and provide:
  BACKBONE_CKPT, DB_PATH, OOD_DB_PATH, BACKBONE_CONFIG

Example:
  RUN_CANONICALIZER_TRAIN=1 \\
  BACKBONE_CKPT=runs/birdsong.ckpt \\
  DB_PATH=/data/birdsong_lmdb \\
  OOD_DB_PATH=/data/tap_preprocessed \\
  BACKBONE_CONFIG=configs/brave_fader_birdsong.gin \\
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
    reason="Set RUN_CANONICALIZER_TRAIN=1 with BACKBONE_CKPT, DB_PATH, OOD_DB_PATH, BACKBONE_CONFIG",
)
def test_train_canonicalizer_smoke():
    env = os.environ.copy()
    required = ["BACKBONE_CKPT", "DB_PATH", "OOD_DB_PATH", "BACKBONE_CONFIG"]
    for k in required:
        assert k in env, f"missing env {k}"

    cmd = [
        sys.executable,
        str(_BRAVE / "RAVE" / "scripts" / "train_canonicalizer.py"),
        "--config",
        "configs/brave_canonicalizer.gin",
        "--backbone_config",
        env["BACKBONE_CONFIG"],
        "--ckpt",
        env["BACKBONE_CKPT"],
        "--db_path",
        env["DB_PATH"],
        "--ood_db_path",
        env["OOD_DB_PATH"],
        "--canonicalizer_type",
        "waveform",
        "--name",
        "pytest_canon_smoke",
        "--smoke_test",
    ]
    subprocess.run(cmd, cwd=str(_BRAVE), check=True, env=env)
