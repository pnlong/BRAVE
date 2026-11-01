#!/usr/bin/env python3
"""Generate CycleGAN TorchScript streaming references for C++ parity.

Loads a ``.ts`` (ScriptedCycleGANXY), streams mono audio in ``block`` chunks
with sidechain on or off, writes float32 ``.npy`` (and optional wav).

  PYTHONPATH=RAVE:brave_rt/python \\
  python brave_rt/python/cyclegan_ts_ref.py \\
    --ts BRAVE-data/exports/cyclegan_ts_done/babbling_brook_10122026.ts \\
    --input BRAVE-data/presentation_demos/babbling_brook/0/stems/00_tap.wav \\
    --out_dir brave_rt/testdata/goldens/cyclegan_brook \\
    --sidechain 0 --block 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _load_mono(path: Path, sr: int = 44100) -> np.ndarray:
    x, file_sr = sf.read(str(path), always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if int(file_sr) != sr:
        raise SystemExit(f"expected {sr} Hz, got {file_sr} for {path}")
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ts", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--sidechain", type=int, default=0, choices=[0, 1])
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--max_seconds", type=float, default=0.0,
                   help="0 = full file; else truncate")
    args = p.parse_args()
    if args.block < 1:
        raise SystemExit("block must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(args.ts), map_location="cpu")
    model.eval()
    if hasattr(model, "set_sidechain"):
        model.set_sidechain(bool(args.sidechain))
    # Fresh sidechain / cache state
    if hasattr(model, "loudness_sidechain") and hasattr(
        model.loudness_sidechain, "reset"
    ):
        model.loudness_sidechain.reset()

    x = _load_mono(args.input)
    if args.max_seconds > 0:
        x = x[: int(args.max_seconds * 44100)]
    n = (len(x) // args.block) * args.block
    x = x[:n]
    y = np.zeros_like(x)

    with torch.inference_mode():
        for i in range(0, n, args.block):
            xin = torch.from_numpy(x[i : i + args.block]).view(1, 1, -1)
            yout = model(xin)
            y[i : i + args.block] = yout.view(-1).cpu().numpy()

    tag = f"stream{args.block}_sc{args.sidechain}"
    np.save(args.out_dir / f"full_x.npy", x)
    np.save(args.out_dir / f"{tag}_y.npy", y)
    sf.write(str(args.out_dir / f"{tag}_y.wav"), y, 44100)
    meta = {
        "ts": str(args.ts.resolve()),
        "input": str(args.input.resolve()),
        "sidechain": bool(args.sidechain),
        "block": args.block,
        "n_samples": int(n),
    }
    (args.out_dir / f"{tag}_meta.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n"
    )
    print(f"Wrote {args.out_dir / (tag + '_y.npy')} ({n} samples)")


if __name__ == "__main__":
    main()
