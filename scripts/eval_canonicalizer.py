#!/usr/bin/env python3
"""Evaluate input canonicalizer: descriptor/latent/recon metrics + audio dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BRAVE = Path(__file__).resolve().parents[1]
_LATENT = _BRAVE / "latent_exploration"
_RAVE = _BRAVE / "RAVE"
for p in (_LATENT, _RAVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from load_model import (
    encode_to_latent_with_warp,
    extract_normalized_attributes,
    load_audio,
    load_fader_with_canonicalizer,
    save_audio,
)
from rave.fader.canonicalizer_config import build_domain_profile, load_latent_stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Fader gin config")
    p.add_argument("--ckpt", required=True, help="FaderRAVE checkpoint")
    p.add_argument("--db-path", required=True, help="LMDB / stats path")
    p.add_argument("--ood-path", required=True, help="Directory of OOD WAVs")
    p.add_argument("--waveform-canonicalizer", default=None)
    p.add_argument("--latent-canonicalizer", default=None)
    p.add_argument("--output-dir", default="artifacts/canonicalizer_eval")
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def recon_loss(model, x, attr_norm, *, use_warp: bool) -> float:
    if use_warp:
        z = encode_to_latent_with_warp(model, x.unsqueeze(0))
    else:
        z = model.encode_to_latent(x.unsqueeze(0))
    y = model.decode(z, attr=attr_norm)
    t = min(x.shape[-1], y.shape[-1])
    d_fb = model.audio_distance(x[:t], y[0, :, :t])
    return float(sum(d_fb.values()))


def main():
    args = parse_args()

    profile = build_domain_profile(args.config, args.db_path)
    bundle = load_fader_with_canonicalizer(
        args.ckpt,
        config_path=args.config,
        db_path=args.db_path,
        waveform_canonicalizer_ckpt=args.waveform_canonicalizer,
        latent_canonicalizer_ckpt=args.latent_canonicalizer,
        use_gpu=args.gpu,
    )
    model = bundle.model
    device = next(model.parameters()).device

    latent_mean = None
    if profile.latent_stats_path and profile.latent_stats_path.is_file():
        latent_mean = torch.tensor(
            load_latent_stats(profile.latent_stats_path)["latent_mean"],
            device=device,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    (out_dir / "canonicalized").mkdir(exist_ok=True)
    (out_dir / "recon").mkdir(exist_ok=True)

    ood_files = sorted(Path(args.ood_path).glob("*.wav"))
    rows = []

    for wav in ood_files:
        x = load_audio(wav, model, device=device)
        attr = extract_normalized_attributes(model, x)

        z_raw = model.encode_to_latent(x.unsqueeze(0))
        z_canon = encode_to_latent_with_warp(model, x.unsqueeze(0))

        row = {"file": wav.name}
        if latent_mean is not None:
            row["latent_dist_raw"] = float(
                (z_raw.mean(dim=(0, 2)) - latent_mean).pow(2).mean().sqrt())
            row["latent_dist_canon"] = float(
                (z_canon.mean(dim=(0, 2)) - latent_mean).pow(2).mean().sqrt())

        row["recon_loss_raw"] = recon_loss(model, x, attr, use_warp=False)
        row["recon_loss_canon"] = recon_loss(model, x, attr, use_warp=True)

        if model.waveform_canonicalizer is not None:
            x_c = model.waveform_canonicalizer(x.unsqueeze(0))[0]
            save_audio(out_dir / "canonicalized" / wav.name, x_c, model.sr)
            row["waveform_identity_l1"] = float((x_c - x).abs().mean())

        save_audio(out_dir / "raw" / wav.name, x, model.sr)
        z = encode_to_latent_with_warp(model, x.unsqueeze(0))
        y = model.decode(z, attr=attr)[0]
        save_audio(out_dir / "recon" / wav.name, y, model.sr)
        rows.append(row)

    report = out_dir / "report.json"
    report.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {report} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
