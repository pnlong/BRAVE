"""Room impulse-response augmentation for canonicalizer OOD training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import gin
import numpy as np
import soundfile as sf
import torchaudio
from scipy import signal


def _normalize_ir(ir: np.ndarray, peak: float = 1.0) -> np.ndarray:
    ir = ir.astype(np.float32)
    p = float(np.max(np.abs(ir)))
    if p < 1e-12:
        return ir
    return ir * (peak / p)


def synthetic_room_ir(
    sampling_rate: int,
    decay_sec: float = 0.35,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Short exponential-decay noise burst when no IR library is available."""
    rng = rng or np.random.default_rng()
    n = max(64, int(sampling_rate * decay_sec))
    noise = rng.standard_normal(n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sampling_rate
    env = np.exp(-4.0 * t / max(decay_sec, 1e-3))
    return _normalize_ir(noise * env)


@gin.configurable
class ImpulseResponseAug:
    """
    Randomly convolve mono/stereo audio with a room IR (wet/dry mix).

    IRs are loaded from ``ir_path`` (*.wav). When the directory is empty or
    missing, falls back to short synthetic room IRs.
    """

    def __init__(
        self,
        ir_path: Union[str, Path, None] = None,
        sampling_rate: int = 44100,
        prob: float = 0.5,
        wet_min: float = 0.15,
        wet_max: float = 0.55,
        max_ir_sec: float = 2.0,
        use_synthetic_fallback: bool = True,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if wet_min > wet_max:
            raise ValueError("wet_min must be <= wet_max")
        self.sampling_rate = sampling_rate
        self.prob = prob
        self.wet_min = wet_min
        self.wet_max = wet_max
        self.max_ir_samples = max(64, int(max_ir_sec * sampling_rate))
        self.use_synthetic_fallback = use_synthetic_fallback
        self._ir_files = self._discover_ir_files(ir_path)
        self._ir_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _discover_ir_files(ir_path: Union[str, Path, None]) -> list[Path]:
        if ir_path is None or str(ir_path).strip() == "":
            return []
        root = Path(ir_path)
        if not root.is_dir():
            return []
        files = sorted(root.rglob("*.wav"))
        return files

    @property
    def enabled(self) -> bool:
        return self.prob > 0.0 and (
            bool(self._ir_files) or self.use_synthetic_fallback)

    def _load_ir(self, path: Optional[Path] = None) -> np.ndarray:
        if path is None:
            return synthetic_room_ir(self.sampling_rate)
        key = str(path.resolve())
        if key in self._ir_cache:
            return self._ir_cache[key]
        audio, sr = sf.read(str(path), always_2d=False)
        ir = np.asarray(audio, dtype=np.float32)
        if ir.ndim > 1:
            ir = ir.mean(axis=1)
        if sr != self.sampling_rate:
            ir_t = torchaudio.functional.resample(
                torch.from_numpy(ir).unsqueeze(0),
                sr,
                self.sampling_rate,
            ).squeeze(0).numpy()
            ir = ir_t
        if len(ir) > self.max_ir_samples:
            ir = ir[: self.max_ir_samples]
        ir = _normalize_ir(ir)
        self._ir_cache[key] = ir
        return ir

    def _pick_ir(self) -> np.ndarray:
        if self._ir_files:
            return self._load_ir(random.choice(self._ir_files))
        if self.use_synthetic_fallback:
            return self._load_ir(None)
        raise RuntimeError("No impulse responses available")

    @staticmethod
    def convolve(
        audio: np.ndarray,
        ir: np.ndarray,
        wet: float,
    ) -> np.ndarray:
        """
        Args:
            audio: (C, T) float32
            ir: (T_ir,) mono IR
            wet: wet mix in [0, 1]
        """
        wet = float(np.clip(wet, 0.0, 1.0))
        dry = 1.0 - wet
        out = []
        for ch in audio:
            wet_ch = signal.fftconvolve(ch, ir, mode="same").astype(np.float32)
            mixed = dry * ch + wet * wet_ch
            peak = float(np.max(np.abs(mixed)))
            if peak > 1.0:
                mixed = mixed / peak
            out.append(mixed)
        return np.stack(out, axis=0).astype(np.float32)

    def maybe_apply(self, audio: np.ndarray) -> np.ndarray:
        if not self.enabled or random.random() >= self.prob:
            return audio
        ir = self._pick_ir()
        wet = random.uniform(self.wet_min, self.wet_max)
        return self.convolve(audio, ir, wet)
