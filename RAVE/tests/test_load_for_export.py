"""Tests for export load helpers."""

from pathlib import Path

import pytest

from rave.fader.export.load_for_export import (
    is_fader_config,
    is_fader_model,
    resolve_canonicalizer_ckpt,
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


def test_resolve_canonicalizer_explicit_path(tmp_path):
    ckpt = tmp_path / "waveform_canonicalizer.ckpt"
    ckpt.write_bytes(b"x")
    resolved = resolve_canonicalizer_ckpt(
        str(tmp_path),
        mode="auto",
        waveform_canonicalizer=str(ckpt),
    )
    assert resolved == str(ckpt)


def test_resolve_canonicalizer_auto_in_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = run_dir / "latent_canonicalizer.ckpt"
    ckpt.write_bytes(b"x")
    fake_run = run_dir / "epoch_1.ckpt"
    fake_run.write_bytes(b"")
    resolved = resolve_canonicalizer_ckpt(str(fake_run), mode="auto")
    assert resolved == str(ckpt)


def test_resolve_canonicalizer_none():
    assert resolve_canonicalizer_ckpt("/nonexistent", mode="none") is None
