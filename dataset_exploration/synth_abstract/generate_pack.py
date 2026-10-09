#!/usr/bin/env python3
"""
Generate a mono 44.1 kHz abstract / sine / noise pack for foundation-Y BRAVE.

Writes float32 WAVs (scipy) under --output-dir. Deterministic given --seed.

Categories (each clip ~4–12 s unless overridden):
  sine          pure / beating sine tones
  fm            simple FM
  additive      stacked partials with slow vibrato
  noise         white / pink / band-limited noise
  pulse         square / pulse with PWM
  morph         filtered-noise + tone crossfades

Example:
  python3 generate_pack.py --output-dir /path/to/synth_abstract/generated \\
      --hours 4 --seed 42
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile


SR = 44100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--hours",
        type=float,
        default=4.0,
        help="Target total duration of generated audio (default 4 h).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-sec", type=float, default=4.0, help="Min clip length (seconds)."
    )
    p.add_argument(
        "--max-sec", type=float, default=12.0, help="Max clip length (seconds)."
    )
    p.add_argument(
        "--peak",
        type=float,
        default=0.35,
        help="Peak amplitude before soft clip (default 0.35).",
    )
    return p.parse_args()


def _fade(x: np.ndarray, fade_ms: float = 30.0) -> np.ndarray:
    n = max(1, int(SR * fade_ms / 1000.0))
    if x.size < 2 * n:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = x.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def _normalize(x: np.ndarray, peak: float) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-12
    return (x / m) * peak


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Voss-McCartney-ish pink via stacking octaves of white."""
    white = rng.standard_normal(n)
    out = np.zeros(n, dtype=np.float64)
    for octv in range(1, 7):
        step = 2**octv
        chunk = rng.standard_normal(n // step + 2)
        out += np.repeat(chunk, step)[:n]
    out += white * 0.1
    return out


def gen_sine(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    f0 = float(rng.uniform(55.0, 2200.0))
    if rng.random() < 0.4:
        f1 = f0 * float(rng.uniform(1.001, 1.02))
        x = np.sin(2 * math.pi * f0 * t) + 0.7 * np.sin(2 * math.pi * f1 * t)
    else:
        x = np.sin(2 * math.pi * f0 * t)
    if rng.random() < 0.5:
        lfo = 1.0 + 0.15 * np.sin(2 * math.pi * float(rng.uniform(0.1, 4.0)) * t)
        x *= lfo
    return x


def gen_fm(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    fc = float(rng.uniform(80.0, 1200.0))
    fm = float(rng.uniform(0.5, 40.0))
    idx = float(rng.uniform(0.5, 8.0))
    mod = np.sin(2 * math.pi * fm * t)
    return np.sin(2 * math.pi * fc * t + idx * mod)


def gen_additive(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    f0 = float(rng.uniform(40.0, 400.0))
    partials = int(rng.integers(3, 10))
    x = np.zeros(n, dtype=np.float64)
    for k in range(1, partials + 1):
        amp = 1.0 / k
        vib = 1.0 + 0.003 * np.sin(2 * math.pi * float(rng.uniform(2.0, 6.0)) * t)
        x += amp * np.sin(2 * math.pi * f0 * k * vib * t)
    return x


def gen_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(["white", "pink", "band"])
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _pink_noise(n, rng)
    else:
        x = rng.standard_normal(n)
        # crude one-pole lowpass then highpass-ish by differencing
        a = float(rng.uniform(0.01, 0.2))
        y = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            y[i] = y[i - 1] + a * (x[i] - y[i - 1])
        x = y
    return x


def gen_pulse(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SR
    f0 = float(rng.uniform(40.0, 600.0))
    duty0 = float(rng.uniform(0.1, 0.5))
    if rng.random() < 0.6:
        duty = duty0 + 0.25 * np.sin(2 * math.pi * float(rng.uniform(0.05, 2.0)) * t)
    else:
        duty = np.full(n, duty0)
    phase = (f0 * t) % 1.0
    return np.where(phase < duty, 1.0, -1.0).astype(np.float64)


def gen_morph(n: int, rng: np.random.Generator) -> np.ndarray:
    a = gen_noise(n, rng)
    b = gen_sine(n, rng)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if rng.random() < 0.5:
        w = 0.5 - 0.5 * np.cos(math.pi * t)  # raised-cosine wipe
    else:
        w = t
    return (1.0 - w) * a + w * b


GENERATORS = {
    "sine": gen_sine,
    "fm": gen_fm,
    "additive": gen_additive,
    "noise": gen_noise,
    "pulse": gen_pulse,
    "morph": gen_morph,
}


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    target_samples = int(args.hours * 3600 * SR)
    written = 0
    idx = 0
    cats = list(GENERATORS.keys())

    while written < target_samples:
        cat = str(rng.choice(cats))
        dur = float(rng.uniform(args.min_sec, args.max_sec))
        n = int(dur * SR)
        x = GENERATORS[cat](n, rng)
        x = _fade(_normalize(x, args.peak))
        # int16 for wide tool compatibility
        pcm = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        path = out / f"gen_{cat}_{idx:05d}.wav"
        wavfile.write(str(path), SR, pcm)
        written += n
        idx += 1

    hours = written / SR / 3600.0
    print(
        f"# wrote {idx} clips, {hours:.3f} h @ {SR} Hz → {out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
