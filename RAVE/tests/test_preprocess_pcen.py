import importlib.util
import unittest
from pathlib import Path

import numpy as np

_RAVE_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = _RAVE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pcen_mod = _load_module('preprocess_pcen', 'rave/preprocess_pcen.py')
PcenConfig = pcen_mod.PcenConfig
pcen_pcm = pcen_mod.pcen_pcm


class PreprocessPcenTest(unittest.TestCase):

    def test_disabled_is_noop(self):
        sr = 44100
        rng = np.random.default_rng(0)
        pcm = rng.normal(0, 0.1, (1, sr)).astype(np.float32)
        out = pcen_pcm(pcm, sr, PcenConfig.disabled())
        np.testing.assert_array_equal(out, pcm)

    def test_boosts_transient_over_steady_noise(self):
        sr = 44100
        n = sr * 3
        t = np.arange(n, dtype=np.float64) / sr
        rng = np.random.default_rng(2)
        steady = 0.08 * rng.normal(0, 1, n)
        chirp = np.zeros(n, dtype=np.float64)
        center = n // 2
        width = sr // 20
        chirp[center - width:center + width] = 0.35 * np.sin(
            2 * np.pi * 4000 * t[center - width:center + width])
        pcm = (steady + chirp)[None, :].astype(np.float32)

        def transient_snr(x: np.ndarray) -> float:
            seg = x[0, center - width:center + width]
            rest = np.concatenate([
                x[0, :center - width],
                x[0, center + width:],
            ])
            return float(np.sqrt(np.mean(seg ** 2)) / (np.sqrt(np.mean(rest ** 2)) + 1e-8))

        raw_snr = transient_snr(pcm)
        out = pcen_pcm(
            pcm,
            sr,
            PcenConfig(enabled=True, n_fft=2048, n_mels=128, max_gain=10.0),
        )
        self.assertGreater(transient_snr(out), raw_snr * 1.1)


if __name__ == '__main__':
    unittest.main()
