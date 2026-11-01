import importlib.util
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

import numpy as np

_RAVE_ROOT = Path(__file__).resolve().parents[1]


def _load_preprocess_plan():
    pkg = types.ModuleType('rave')
    pkg.__path__ = [str(_RAVE_ROOT / 'rave')]
    pkg.__package__ = 'rave'
    sys.modules.setdefault('rave', pkg)
    spec = importlib.util.spec_from_file_location(
        'rave.preprocess_plan',
        _RAVE_ROOT / 'rave' / 'preprocess_plan.py',
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['rave.preprocess_plan'] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load_preprocess_plan()


def _write_stereo(path: str, left: int, right: int, n: int, sr: int) -> None:
    interleaved = np.empty(n * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(path, 'w') as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(interleaved.tobytes())


class PreprocessDownmixTest(unittest.TestCase):

    def test_mono_filter_averages_every_input_channel(self):
        self.assertEqual(
            pp._output_channel_filters(2, 1),
            ['pan=mono|c0<c0+c1'],
        )
        self.assertEqual(
            pp._output_channel_filters(3, 1),
            ['pan=mono|c0<c0+c1+c2'],
        )

    def test_channel_select_unchanged_when_not_collapsing_to_mono(self):
        self.assertEqual(
            pp._output_channel_filters(2, 2),
            ['channelmap=0-0', 'channelmap=1-0'],
        )
        self.assertEqual(
            pp._output_channel_filters(1, 1),
            ['channelmap=0-0'],
        )

    def test_decode_and_stream_average_stereo(self):
        sr = 8000
        n_signal = 1000
        n = 2 * n_signal
        left, right = 16384, -8192
        expected = (left + right) // 2
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'stereo.wav')
            _write_stereo(path, left, right, n, sr)
            pcm = pp.decode_path_pcm(path, sr, channels=1, input_channels=2)
            self.assertEqual(pcm.shape[0], 1)
            self.assertGreater(pcm.shape[1], 0)
            np.testing.assert_allclose(
                pcm[0],
                expected / (2**15 - 1),
                atol=1.0 / (2**15 - 1),
            )
            chunks = list(pp.iter_long_file_chunks(
                path, n_signal, sr, channels=1, input_channels=2))
        self.assertEqual(len(chunks), 1)
        stored = np.frombuffer(chunks[0], dtype=np.int16)
        self.assertTrue(np.all(stored == expected))


if __name__ == '__main__':
    unittest.main()
