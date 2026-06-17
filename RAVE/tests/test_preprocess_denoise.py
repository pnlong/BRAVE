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


denoise_mod = _load_module('preprocess_denoise', 'rave/preprocess_denoise.py')
DenoiseConfig = denoise_mod.DenoiseConfig
denoise_pcm = denoise_mod.denoise_pcm


class PreprocessDenoiseTest(unittest.TestCase):

    def test_disabled_is_noop(self):
      sr = 44100
      rng = np.random.default_rng(0)
      pcm = rng.normal(0, 0.1, (1, sr)).astype(np.float32)
      out = denoise_pcm(pcm, sr, DenoiseConfig.disabled())
      np.testing.assert_array_equal(out, pcm)

    def test_reduces_broadband_hiss(self):
      sr = 44100
      n = sr * 2
      rng = np.random.default_rng(1)
      noise = rng.normal(0, 0.05, (1, n)).astype(np.float32)
      tone = 0.2 * np.sin(
          2 * np.pi * 440 * np.arange(n) / sr, dtype=np.float32)
      pcm = (tone + noise)[None, :]
      out = denoise_pcm(
          pcm,
          sr,
          DenoiseConfig(enabled=True, strength=0.9, noise_sec=0.0),
      )
      self.assertLess(
          float(np.std(out - tone)),
          float(np.std(pcm - tone)),
      )


if __name__ == '__main__':
    unittest.main()
