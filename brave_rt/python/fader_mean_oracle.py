#!/usr/bin/env python3
"""PyTorch mean-encode oracle for FaderRAVE (attr_mode=1 manual attrs).

Streaming path matches BraveRT C++: PQMF → Enc mean → concat attr_norm → Dec → iPQMF,
128-sample ticks, causal caches zeroed after cached_conv rebuild.

  export PYTHONPATH="${PWD}/RAVE:${PWD}/brave_rt/python:${PYTHONPATH}"
  python brave_rt/python/fader_mean_oracle.py \\
    --model /path/to/fader_run --stats /path/attribute_stats.yaml \\
    --input tap.wav --out oracle.npy --block 128 \\
    [--attr water_scene=0]...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_BRAVE = Path(__file__).resolve().parents[2]
_RAVE = _BRAVE / "RAVE"
if str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))
if str(_BRAVE / "brave_rt" / "python") not in sys.path:
    sys.path.insert(0, str(_BRAVE / "brave_rt" / "python"))

import cached_conv as cc
import numpy as np
import torch
import torchaudio
import yaml

from export_brave_rt import find_attribute_stats, load_brave_or_fader


def build_constant_attr(
    model,
    attribute_values: Dict[str, float],
    *,
    time_frames: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    rows = []
    for name in model.attribute_names:
        val = attribute_values.get(name, 0.0)
        rows.append(np.full(time_frames, float(val), dtype=np.float32))
    raw_t = torch.from_numpy(np.stack(rows, axis=0)).float().to(device).unsqueeze(0)
    attr_norm, _ = model._prepare_attributes(raw_t)
    return attr_norm


def _zero_streaming_caches(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "pad") and torch.is_tensor(getattr(m, "pad", None)):
            m.pad.zero_()
        if hasattr(m, "cache") and torch.is_tensor(getattr(m, "cache", None)):
            m.cache.zero_()


def _load_stats(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _default_raw_values(stats: dict) -> Dict[str, float]:
    names = list(stats.get("attribute_names", []))
    kinds = stats.get("attribute_kinds", {})
    min_max = stats.get("min_max_features", {})
    out: Dict[str, float] = {}
    for name in names:
        if kinds.get(name, "continuous") == "continuous":
            lo, hi = (float(v) for v in min_max.get(name, (0.0, 1.0)))
            out[name] = (lo + hi) / 2.0
        else:
            out[name] = 0.0
    return out


def _parse_attr_overrides(pairs: list[str], stats: dict) -> Dict[str, float]:
    labels = stats.get("discrete_class_labels", {}) or {}
    names = set(stats.get("attribute_names", []))
    raw = _default_raw_values(stats)
    for kv in pairs:
        if "=" not in kv:
            raise SystemExit(f"--attr expects name=value, got {kv!r}")
        name, val = kv.split("=", 1)
        if name not in names:
            raise SystemExit(f"unknown attribute {name!r}; known: {sorted(names)}")
        kinds = stats.get("attribute_kinds", {})
        if kinds.get(name) == "discrete" and val in labels.get(name, []):
            raw[name] = float(labels[name].index(val))
        else:
            raw[name] = float(val)
    return raw


@torch.inference_mode()
def fader_mean_encode_forward(
    model,
    x: torch.Tensor,
    attr_raw: Dict[str, float],
) -> torch.Tensor:
    """x: (1,1,T) → y mono waveform."""
    bands = model.pqmf(x)
    z_raw = model.encoder(bands)
    if isinstance(z_raw, tuple):
        z_raw = z_raw[0]
    z_mean = z_raw.chunk(2, dim=1)[0]
    t_lat = int(z_mean.shape[-1])
    attr_norm = build_constant_attr(model, attr_raw, time_frames=t_lat, device=x.device)
    y = model.decode(z_mean, attr=attr_norm)
    if y.shape[-1] > x.shape[-1]:
        y = y[..., : x.shape[-1]]
    return y


def _build_streaming_model(model_path: Path, run_ckpt: str) -> torch.nn.Module:
    cc.use_cached_conv(True)
    import gin
    from rave.model import RAVE
    from rave.fader.model import FaderRAVE

    gin.clear_config()
    cfg = __import__("rave").core.search_for_config(str(model_path))
    gin.parse_config_file(cfg)
    is_fader = "FaderRAVE" in open(cfg).read() or "fader.model" in open(cfg).read()
    model = FaderRAVE() if is_fader else RAVE()
    sd = torch.load(run_ckpt, map_location="cpu")
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    model.load_state_dict(state, strict=False)
    model.eval()
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    _zero_streaming_caches(model)
    return model


def _apply_latency(y: np.ndarray, latency: int) -> np.ndarray:
    if latency <= 0 or latency >= y.size:
        return y
    out = np.roll(y, -latency)
    out[-latency:] = 0.0
    return out


def _fifo_latency_samples(host_block: int) -> int:
    tick = 128
    if host_block <= 0:
        return tick - 1
    from math import gcd

    return tick - gcd(host_block, tick)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--block", type=int, default=128)
    p.add_argument(
        "--attr",
        action="append",
        default=[],
        help="Raw training-unit override (name=value or discrete label)",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second waveform/npy to print max-abs vs oracle",
    )
    p.add_argument(
        "--apply-latency",
        action="store_true",
        help="Rotate output by FIFO latency (match fader_parity_cli)",
    )
    args = p.parse_args()

    stats_path = args.stats or find_attribute_stats(args.model)
    if stats_path is None:
        raise SystemExit("--stats required (or place attribute_stats.yaml near run/db)")
    stats = _load_stats(stats_path)
    attr_raw = _parse_attr_overrides(args.attr, stats)

    cc.use_cached_conv(False)
    model, is_fader = load_brave_or_fader(args.model)
    if not is_fader:
        raise SystemExit("model is not FaderRAVE")
    model.load_attribute_stats_from_file(stats_path)

    wav, sr = torchaudio.load(str(args.input))
    if sr != args.sr:
        wav = torchaudio.functional.resample(wav, sr, args.sr)
    if wav.shape[0] > 1:
        wav = wav[:1]
    x = wav.unsqueeze(0)

    b = int(args.block)
    if b <= 0 or b % 128 != 0:
        raise SystemExit("--block must be a positive multiple of 128")

    import rave.core as rave_core

    run = rave_core.search_for_run(str(args.model))
    if run is None:
        raise SystemExit("checkpoint not found")
    model_s = _build_streaming_model(args.model, run)
    model_s.load_attribute_stats_from_file(stats_path)

    chunks: list[torch.Tensor] = []
    T = x.shape[-1]
    for start in range(0, T, b):
        chunk = x[..., start : start + b]
        valid = chunk.shape[-1]
        if valid < b:
            chunk = torch.nn.functional.pad(chunk, (0, b - valid))
        y_c = fader_mean_encode_forward(model_s, chunk, attr_raw)[..., :valid]
        chunks.append(y_c)
    y = torch.cat(chunks, dim=-1).squeeze().cpu().numpy().astype(np.float32)

    if args.apply_latency:
        y = _apply_latency(y, _fifo_latency_samples(b))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, y)
    meta = {
        "model": str(args.model.resolve()),
        "stats": str(Path(stats_path).resolve()),
        "input": str(args.input.resolve()),
        "block": b,
        "attr_raw": attr_raw,
        "oracle": "fader_mean_encode_stream_cached",
        "samples": int(y.size),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} ({y.size} samples)")
    print("attr_raw:", json.dumps(attr_raw))

    if args.compare is not None:
        if args.compare.suffix == ".npy":
            est = np.load(args.compare).reshape(-1)
        else:
            w, _ = torchaudio.load(str(args.compare))
            est = w.numpy().reshape(-1)
        n = min(y.size, est.size)
        ref, est = y[:n], est[:n]
        err = float(np.max(np.abs(ref - est)))
        num = float(np.sum(ref.astype(np.float64) ** 2)) + 1e-20
        den = float(np.sum((ref.astype(np.float64) - est.astype(np.float64)) ** 2)) + 1e-20
        snr = 10.0 * np.log10(num / den)
        print(f"compare n={n} max_abs={err:.6e} snr_db={snr:.2f}")
        n128 = (n // 128) * 128
        if n128 > 0 and n128 < n:
            r, e = ref[:n128], est[:n128]
            err128 = float(np.max(np.abs(r - e)))
            snr128 = 10.0 * np.log10(
                float(np.sum(r.astype(np.float64) ** 2))
                / (float(np.sum((r.astype(np.float64) - e.astype(np.float64)) ** 2)) + 1e-20)
            )
            print(
                f"compare aligned n={n128} max_abs={err128:.6e} snr_db={snr128:.2f} "
                "(parity gate; tail may differ on partial final block)"
            )


if __name__ == "__main__":
    main()
