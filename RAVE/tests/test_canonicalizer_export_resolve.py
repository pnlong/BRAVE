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


def test_resolve_cyclegan_ckpt_prefers_latent_not_last(tmp_path):
    from rave.canonicalizer.export import resolve_cyclegan_ckpt

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.ckpt").write_bytes(b"lightning")
    latent = run_dir / "cyclegan_latent.ckpt"
    latent.write_bytes(b"warps")
    assert resolve_cyclegan_ckpt(str(run_dir)) == str(latent.resolve())
    assert resolve_cyclegan_ckpt(str(run_dir / "last.ckpt")) == str(latent.resolve())
    assert resolve_cyclegan_ckpt(str(latent)) == str(latent.resolve())
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "last.ckpt").write_bytes(b"x")
    assert resolve_cyclegan_ckpt(str(empty)) is None
