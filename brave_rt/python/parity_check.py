#!/usr/bin/env python3
"""Compare two float waveforms or .npy arrays; print max-abs and SNR."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    num = float(np.sum(ref.astype(np.float64) ** 2)) + 1e-20
    den = float(np.sum((ref.astype(np.float64) - est.astype(np.float64)) ** 2)) + 1e-20
    return 10.0 * math.log10(num / den)


def load(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    import torchaudio

    w, _ = torchaudio.load(str(path))
    return w.numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--est", type=Path, required=True)
    args = p.parse_args()
    a = load(args.ref).reshape(-1)
    b = load(args.est).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    err = np.max(np.abs(a - b))
    print(f"n={n} max_abs={err:.6e} snr_db={snr_db(a, b):.2f}")


if __name__ == "__main__":
    main()
