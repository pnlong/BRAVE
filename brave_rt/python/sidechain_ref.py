#!/usr/bin/env python3
"""Dump CausalLoudnessSidechain reference for C++ sidechain_parity_cli."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from rave.canonicalizer.silence_gate import CausalLoudnessSidechain


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--hop", type=int, default=128)
    p.add_argument("--n_hops", type=int, default=64)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = args.hop * args.n_hops
    x = np.zeros(n, dtype=np.float32)
    x[: args.hop * 2] = 0.5
    y = np.ones(n, dtype=np.float32)
    sc = CausalLoudnessSidechain(hop=args.hop, knee_db=-40.0, smooth_frames=8)
    with torch.inference_mode():
        y_out = sc(
            torch.from_numpy(x).view(1, 1, -1),
            torch.from_numpy(y).view(1, 1, -1),
        ).view(-1).numpy()
    np.save(args.out_dir / "sc_x.npy", x)
    np.save(args.out_dir / "sc_y_in.npy", y)
    np.save(args.out_dir / "sc_y_ref.npy", y_out.astype(np.float32))
    print(f"Wrote sidechain refs to {args.out_dir}")


if __name__ == "__main__":
    main()
