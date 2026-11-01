#!/usr/bin/env python3
"""Dump BraveRT vanilla encoder/decoder/PQMF weights as individual .npy files for C++."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

KEYS = [
    "pqmf.forward_conv.weight",
    "pqmf.inverse_conv.weight",
    "encoder.encoder.net.0.weight",
    "encoder.encoder.net.1.weight",
    "encoder.encoder.net.1.bias",
    "encoder.encoder.net.1.running_mean",
    "encoder.encoder.net.1.running_var",
    "encoder.encoder.net.3.weight",
    "encoder.encoder.net.4.weight",
    "encoder.encoder.net.4.bias",
    "encoder.encoder.net.4.running_mean",
    "encoder.encoder.net.4.running_var",
    "encoder.encoder.net.6.weight",
    "encoder.encoder.net.7.weight",
    "encoder.encoder.net.7.bias",
    "encoder.encoder.net.7.running_mean",
    "encoder.encoder.net.7.running_var",
    "encoder.encoder.net.9.weight",
    "encoder.encoder.net.10.weight",
    "encoder.encoder.net.10.bias",
    "encoder.encoder.net.10.running_mean",
    "encoder.encoder.net.10.running_var",
    "encoder.encoder.net.12.weight",
    "encoder.encoder.net.14.weight",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with h5py.File(args.h5, "r") as f:
        for k in f.keys():
            safe = k.replace(".", "__")
            path = args.out_dir / f"{safe}.npy"
            ds = f[k]
            arr = np.array(ds)
            np.save(path, arr)
            written.append(k)
    (args.out_dir / "keys.json").write_text(json.dumps(written, indent=2) + "\n")
    print(f"Wrote {len(written)} arrays to {args.out_dir}")


if __name__ == "__main__":
    main()
