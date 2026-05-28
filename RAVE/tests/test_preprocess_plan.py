import importlib.util
import unittest
from pathlib import Path

_RAVE_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = _RAVE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load_module('preprocess_plan', 'rave/preprocess_plan.py')
AudioProbe = pp.AudioProbe
build_preprocess_plan = pp.build_preprocess_plan
expected_preprocess_chunks = pp.expected_preprocess_chunks
min_samples_for_mode = pp.min_samples_for_mode
resolve_packed_position = pp.resolve_packed_position


class PreprocessPlanTest(unittest.TestCase):

    def test_min_samples_non_lazy(self):
        self.assertEqual(min_samples_for_mode(100, False), 200)
        self.assertEqual(min_samples_for_mode(100, True), 100)

    def test_expected_chunks(self):
        sr = 44100
        num_signal = 131072
        # 6 seconds -> floor(6*44100) = 264600 samples -> 264600 // 262144 = 1 chunk
        self.assertEqual(
            expected_preprocess_chunks(6.0, sr, num_signal, lazy=False), 1)
        self.assertEqual(
            expected_preprocess_chunks(2.0, sr, num_signal, lazy=False), 0)

    def test_build_packs_greedy(self):
        sr = 44100
        num_signal = 1000
        probes = [
            AudioProbe('/a.wav', 0.005, 1),
            AudioProbe('/b.wav', 0.005, 1),
            AudioProbe('/c.wav', 10.0, 1),
        ]
        plan = build_preprocess_plan(
            probes,
            sr,
            num_signal,
            lazy=False,
            concat_short=True,
            concat_seed=0,
            pad_short_remainder=False,
        )
        self.assertEqual(len(plan.long_files), 1)
        self.assertEqual(plan.long_files[0].path, '/c.wav')
        self.assertEqual(plan.short_file_count, 2)
        self.assertGreaterEqual(len(plan.packs), 1)
        self.assertGreaterEqual(
            plan.packs[0].total_samples(sr),
            min_samples_for_mode(num_signal, False),
        )

    def test_pad_remainder(self):
        sr = 44100
        num_signal = 10000
        probes = [AudioProbe('/only.wav', 0.01, 1)]
        plan = build_preprocess_plan(
            probes,
            sr,
            num_signal,
            lazy=False,
            concat_short=True,
            concat_seed=1,
            pad_short_remainder=True,
        )
        self.assertEqual(len(plan.packs), 1)
        self.assertGreater(plan.packs[0].pad_samples, 0)
        self.assertEqual(plan.remainder_discarded_samples, 0)

    def test_resolve_packed_position(self):
        lengths = [1.0, 2.0]
        sr = 100
        file_idx, offset = resolve_packed_position(lengths, sr, 150)
        self.assertEqual(file_idx, 1)
        self.assertEqual(offset, 50)


class RmsDbfsTest(unittest.TestCase):

    def test_rms_dbfs(self):
        import math
        import numpy as np

        def rms_dbfs(x):
            rms = math.sqrt(float(np.mean(x.astype(np.float64)**2)))
            if rms < 1e-12:
                return float('-inf')
            return float(20.0 * math.log10(rms))

        silence = np.zeros((1, 1024), dtype=np.float32)
        self.assertEqual(rms_dbfs(silence), float('-inf'))

        tone = np.full((1, 1024), 0.1, dtype=np.float32)
        self.assertGreater(rms_dbfs(tone), -50.0)


if __name__ == '__main__':
    unittest.main()
