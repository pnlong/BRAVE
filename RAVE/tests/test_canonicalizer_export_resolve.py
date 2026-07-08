"""Tests for canonicalizer export resolve helpers."""

from rave.canonicalizer.export import resolve_canonicalizer_ckpt


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
