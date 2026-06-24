import numpy as np

from rave.preprocess_normalize import NormalizeConfig, normalize_pcm


def test_disabled_is_noop():
    pcm = np.array([[0.01, 0.02, -0.01]], dtype=np.float32)
    out = normalize_pcm(pcm, NormalizeConfig.disabled())
    np.testing.assert_array_equal(out, pcm)


def test_boosts_quiet_chunk_toward_unit_peak():
    pcm = np.array([[0.1, -0.05, 0.08]], dtype=np.float32)
    out = normalize_pcm(pcm, NormalizeConfig(enabled=True, max_gain_db=30.0))
    assert abs(float(np.max(np.abs(out))) - 1.0) < 1e-5


def test_caps_gain_at_max_gain_db():
    pcm = np.array([[1e-4, -2e-4, 1e-4]], dtype=np.float32)
    out = normalize_pcm(pcm, NormalizeConfig(enabled=True, max_gain_db=20.0))
    assert float(np.max(np.abs(out))) < 1.0
