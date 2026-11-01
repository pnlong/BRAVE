#!/usr/bin/env python3
"""Dump mean-encode layer references for BraveRT C++ parity.

Oracle: PQMF → Enc → mean → Dec → iPQMF (no sampling, no PCA).

  cd BRAVE
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python brave_rt/python/dump_layer_refs.py \\
    --model /path/to/run \\
    --input /path/to/tap.wav \\
    --out_dir brave_rt/testdata/goldens/tap_vanilla
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BRAVE = Path(__file__).resolve().parents[2]
_RAVE = _BRAVE / "RAVE"
if str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

import cached_conv as cc
import numpy as np
import torch
import torchaudio

from export_brave_rt import load_rave_from_path


@torch.inference_mode()
def mean_encode_forward(model, x: torch.Tensor):
    """x: (B, 1, T) → intermediates + y."""
    assert x.dim() == 3
    pqmf = model.pqmf
    bands = pqmf(x) if pqmf is not None else x
    z_raw = model.encoder(bands)
    if isinstance(z_raw, tuple):
        # some paths return distribution
        z_raw = z_raw[0] if z_raw[0].dim() == 3 else z_raw
    z_mean = z_raw.chunk(2, dim=1)[0]
    mb = model.decoder(z_mean)
    if pqmf is not None:
        # decoder returns multiband; inverse
        y = pqmf.inverse(mb)
    else:
        y = mb
    if y.shape[-1] > x.shape[-1]:
        y = y[..., : x.shape[-1]]
    return {
        "x": x,
        "pqmf_bands": bands,
        "z_raw": z_raw,
        "z_mean": z_mean,
        "decoder_mb": mb,
        "y": y,
    }


def _save_npy(out: Path, name: str, t: torch.Tensor) -> None:
    np.save(out / f"{name}.npy", t.detach().cpu().float().numpy())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--block", type=int, default=128, help="Also dump first block of this size")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Streaming caches for multi-block goldens later; for full-clip mean path
    # use non-cached for a clean full-context oracle, plus a streaming block dump.
    cc.use_cached_conv(False)
    model = load_rave_from_path(args.model)

    wav, sr = torchaudio.load(str(args.input))
    if sr != args.sr:
        wav = torchaudio.functional.resample(wav, sr, args.sr)
    if wav.shape[0] > 1:
        wav = wav[:1]
    x = wav.unsqueeze(0)  # (1,1,T)

    outs = mean_encode_forward(model, x)
    for k, v in outs.items():
        _save_npy(args.out_dir, f"full_{k}", v)
    torchaudio.save(
        str(args.out_dir / "full_y.wav"),
        outs["y"].squeeze(0).cpu(),
        args.sr,
    )
    torchaudio.save(
        str(args.out_dir / "full_x.wav"),
        outs["x"].squeeze(0).cpu(),
        args.sr,
    )

    # First block only (full-context truncated input of length block — note:
    # not identical to streaming; also write streaming block recon).
    b = int(args.block)
    x_b = x[..., :b]
    if x_b.shape[-1] < b:
        x_b = torch.nn.functional.pad(x_b, (0, b - x_b.shape[-1]))
    outs_b = mean_encode_forward(model, x_b)
    for k, v in outs_b.items():
        _save_npy(args.out_dir, f"block{b}_{k}", v)

    # Streaming: must rebuild with cached_conv ON (load_rave_from_path forces OFF).
    cc.use_cached_conv(True)
    gin_cfg = None
    try:
        import gin
        from rave.model import RAVE

        gin.clear_config()
        cfg = __import__("rave").core.search_for_config(str(args.model))
        gin.parse_config_file(cfg)
        model_s = RAVE()
        run = __import__("rave").core.search_for_run(str(args.model))
        sd = torch.load(run, map_location="cpu")
        state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
        model_s.load_state_dict(state, strict=False)
        model_s.eval()
        for m in model_s.modules():
            if hasattr(m, "weight_g"):
                torch.nn.utils.remove_weight_norm(m)
    except Exception as e:
        raise SystemExit(f"failed to build streaming cached model: {e}") from e
    for m in model_s.modules():
        if hasattr(m, "pad") and torch.is_tensor(getattr(m, "pad", None)):
            m.pad.zero_()
        if hasattr(m, "cache") and torch.is_tensor(getattr(m, "cache", None)):
            m.cache.zero_()
    chunks = []
    T = x.shape[-1]
    for start in range(0, T, b):
        chunk = x[..., start : start + b]
        valid = chunk.shape[-1]
        if valid < b:
            chunk = torch.nn.functional.pad(chunk, (0, b - valid))
        y_c = mean_encode_forward(model_s, chunk)["y"][..., :valid]
        chunks.append(y_c)
    y_stream = torch.cat(chunks, dim=-1)
    _save_npy(args.out_dir, f"stream{b}_y", y_stream)
    torchaudio.save(
        str(args.out_dir / f"stream{b}_y.wav"),
        y_stream.squeeze(0).cpu(),
        args.sr,
    )

    meta = {
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "sr": args.sr,
        "block": b,
        "full_samples": int(T),
        "oracle": "mean_encode_no_pca",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote goldens to {args.out_dir}")


if __name__ == "__main__":
    main()
