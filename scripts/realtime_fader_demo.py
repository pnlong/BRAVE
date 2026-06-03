"""
Demo: process WAV through exported Fader TorchScript with attribute modulation.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python scripts/realtime_fader_demo.py \\
    --ts exports/fader.ts \\
    --input tap_samples/0.wav \\
    --output out.wav \\
    --modulate-attr rms --modulate-scale 0.5 \\
    --attr-scales rms=1.2,centroid=0.9
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_BRAVE_ROOT = Path(__file__).resolve().parents[1]
_RAVE = _BRAVE_ROOT / "RAVE"
if str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

from rave.fader.realtime.stream import AttributeStream  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ts", required=True, help="TorchScript FaderTraceModel .ts")
    p.add_argument("--stats", default=None, help="attribute_stats yaml (default: beside .ts)")
    p.add_argument("--input", required=True, help="input wav")
    p.add_argument("--output", required=True, help="output wav")
    p.add_argument("--modulate-attr", default=None, help="single attr to scale (legacy)")
    p.add_argument("--modulate-scale", type=float, default=1.0, help="scale for --modulate-attr")
    p.add_argument(
        "--attr-scales",
        default=None,
        help="comma-separated name=float scales, e.g. rms=1.2,centroid=0.9",
    )
    p.add_argument("--sr", type=int, default=44100)
    return p.parse_args()


def parse_attr_scales(spec: str) -> dict[str, float]:
    scales = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([a-zA-Z0-9_]+)=([+-]?\d*\.?\d+)$", part)
        if not m:
            raise ValueError(f"bad attr-scales token: {part!r}")
        scales[m.group(1)] = float(m.group(2))
    return scales


@torch.no_grad()
def main():
    args = parse_args()
    stats_path = args.stats or str(Path(args.ts).with_suffix("")) + "_attribute_stats.yaml"
    model = torch.jit.load(args.ts)
    stream = AttributeStream.from_stats_path(stats_path, args.sr)

    audio, sr = sf.read(args.input, always_2d=True)
    if sr != args.sr:
        raise ValueError(f"expected sr={args.sr}, got {sr}")
    mono = audio.mean(axis=1).astype(np.float32)
    x = torch.from_numpy(mono[None, None, :]).float()

    # --- Encode content latent ---
    z = model.encode(x)
    t_lat = z.shape[-1]

    # --- Extract + optionally modulate attributes ---
    attr_raw = stream.push(mono, latent_length=t_lat).unsqueeze(0)
    attr_norm = model.normalize_all(attr_raw)

    scales = {}
    if args.attr_scales:
        scales.update(parse_attr_scales(args.attr_scales))
    if args.modulate_attr:
        scales[args.modulate_attr] = args.modulate_scale
    names = stream.attribute_names
    for name, scale in scales.items():
        if name not in names:
            raise ValueError(f"unknown attr {name}; have {names}")
        idx = names.index(name)
        attr_norm[:, idx, :] *= scale

    # --- Concat decode ---
    z_c = torch.cat([z, attr_norm], dim=1)
    y = model.decode(z_c).squeeze().numpy()
    sf.write(args.output, y, args.sr)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
