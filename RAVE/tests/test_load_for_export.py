"""Tests for export load helpers."""

from pathlib import Path

import pytest

from rave.fader.export.load_for_export import (
    is_fader_config,
    is_fader_model,
)


def test_is_fader_config_detects_aliased_gin(tmp_path):
    cfg = tmp_path / "config.gin"
    cfg.write_text(
        "import rave.fader.model as rave5\n"
        "rave5.FaderRAVE.latent_size = 128\n"
    )
    assert is_fader_config(str(cfg))


def test_is_fader_config_rejects_vanilla_rave(tmp_path):
    cfg = tmp_path / "config.gin"
    cfg.write_text("rave.model.RAVE.latent_size = 128\n")
    assert not is_fader_config(str(cfg))


def test_is_fader_model_from_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.gin").write_text("rave5.FaderRAVE.latent_size = 128\n")
    ckpt = run_dir / "best.ckpt"
    ckpt.write_bytes(b"")
    assert is_fader_model(str(ckpt))

