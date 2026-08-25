#!/usr/bin/env python3
"""Presentation demos: phase-1/2/3 tap→Y audio + Y-latent PCA plots.

Phase 1: tap → Enc_Y → Dec_Y (zero-shot)
Phase 2: tap → EQ+reverb(preset) → Enc_Y → Dec_Y
Phase 3: tap → Enc_X → W_xy → Dec_Y (CycleGAN)

Demo WAVs interleave fixed-length segments back-to-back (no gap):
  phase1: tap0 || recon0 || tap1 || recon1 || …
  phase2: tap0 || eq0 || recon0 || tap1 || eq1 || recon1 || …
  phase3: tap0 || cyc0 || tap1 || cyc1 || …
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_BRAVE = Path(__file__).resolve().parents[1]
_LATENT = _BRAVE / "latent_exploration"
_RAVE = _BRAVE / "RAVE"
_SCRIPTS = _BRAVE / "scripts"
for p in (_LATENT, _RAVE, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_model import load_audio, load_model, save_audio  # noqa: E402
from rave.canonicalizer.cycle_inference import (  # noqa: E402
    load_cyclegan_warps_from_checkpoint,
    transfer_x_to_y,
)
from rave.canonicalizer.viz import latent_frames_to_points  # noqa: E402
from rave.dsp import BiquadBank, CausalReverb  # noqa: E402

import demo_assets as da  # noqa: E402

BRAVE_STORAGE = Path(
    os.environ.get("BRAVE_STORAGE", f"/data/hai-res/{os.environ.get('USER', 'p1long')}/BRAVE-data")
)
DEFAULT_DOMAINS = (
    "rain_sounds",
    "babbling_brook",
    "waves_crashing",
    "fire_crackling",
    "night_ambience",
    "birds_chirping",
)

# EQ: 6 log-spaced bands (default BiquadBank 80–12kHz).
# Reverb knobs: [wet_logit, comb_fb×4, ap_gain×2] (pre-sigmoid).
PHASE2_PRESETS: Dict[str, Dict[str, object]] = {
    "waves_crashing": {
        "intent": "Long dark wash",
        "eq_gains_db": [4.0, 2.0, 0.0, -1.0, -3.0, -5.0],
        "wet_logit": 1.2,
        "comb_fb_logits": [1.5, 1.5, 1.5, 1.5],
        "ap_logits": [0.0, 0.0],
    },
    "rain_sounds": {
        "intent": "Medium room + HF hiss",
        "eq_gains_db": [-2.0, 0.0, 1.0, 2.0, 4.0, 5.0],
        "wet_logit": 0.0,
        "comb_fb_logits": [0.5, 0.5, 0.5, 0.5],
        "ap_logits": [0.0, 0.0],
    },
    "babbling_brook": {
        "intent": "Soft room, mid presence",
        "eq_gains_db": [0.0, 1.0, 2.0, 3.0, 1.0, -1.0],
        "wet_logit": -0.5,
        "comb_fb_logits": [0.3, 0.3, 0.3, 0.3],
        "ap_logits": [0.0, 0.0],
    },
    "fire_crackling": {
        "intent": "Dry + strong HF crackle",
        "eq_gains_db": [-3.0, -1.0, 0.0, 2.0, 5.0, 7.0],
        "wet_logit": -3.0,
        "comb_fb_logits": [-1.0, -1.0, -1.0, -1.0],
        "ap_logits": [-1.0, -1.0],
    },
    "night_ambience": {
        "intent": "Long dark reverb",
        "eq_gains_db": [5.0, 3.0, 0.0, -2.0, -5.0, -7.0],
        "wet_logit": 1.5,
        "comb_fb_logits": [2.0, 2.0, 2.0, 2.0],
        "ap_logits": [0.5, 0.5],
    },
    "birds_chirping": {
        "intent": "Near-dry, bright presence",
        "eq_gains_db": [-2.0, 0.0, 2.0, 4.0, 6.0, 4.0],
        "wet_logit": -4.0,
        "comb_fb_logits": [-2.0, -2.0, -2.0, -2.0],
        "ap_logits": [-2.0, -2.0],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help="yt_playlist domain names",
    )
    p.add_argument(
        "--playlists-root",
        type=Path,
        default=BRAVE_STORAGE / "yt_playlists",
    )
    p.add_argument(
        "--tap-dir",
        type=Path,
        default=BRAVE_STORAGE / "tap_samples" / "audio_subset",
    )
    p.add_argument(
        "--tap-backbone",
        type=Path,
        default=(
            BRAVE_STORAGE / "tap_samples" / "runs"
            / "tap_uncond_run_8e1e614287" / "epoch_1500000.ckpt"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=BRAVE_STORAGE / "presentation_demos",
    )
    p.add_argument(
        "--cyclegan-hash",
        default="48072c314a",
        help="Suffix of tap_<domain>_wf_cyc_mlp2_<hash> runs",
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--silence-sec", type=float, default=0.0)
    p.add_argument(
        "--segment-sec",
        type=float,
        default=3.0,
        help="Length of each interleaved input/recon chunk",
    )
    p.add_argument(
        "--max-sec",
        type=float,
        default=15.0,
        help="Max source seconds to take (split into segment-sec chunks; 0 = full)",
    )
    p.add_argument("--pca-frames", type=int, default=512)
    p.add_argument("--n-signal", type=int, default=65536)
    p.add_argument(
        "--y-crops",
        type=int,
        default=32,
        help="Number of in-domain LMDB crops to encode for PCA",
    )
    return p.parse_args()


def discover_tap_files(tap_dir: Path) -> List[Path]:
    files: List[Path] = []
    for name in ("0.wav", "1.wav"):
        p = tap_dir / name
        if p.is_file():
            files.append(p)
    for p in sorted(tap_dir.glob("*-Contact Mic L.unnormalized.wav")):
        files.append(p)
    if not files:
        raise FileNotFoundError(f"no tap clips found under {tap_dir}")
    return files


def decode_latent(backbone, z: torch.Tensor, target_len: int) -> torch.Tensor:
    y = backbone.decode(z)
    return y[..., :target_len]


@torch.no_grad()
def apply_eq_reverb(
    x: torch.Tensor,
    *,
    eq: BiquadBank,
    reverb: CausalReverb,
    eq_gains_db: Sequence[float],
    rev_knobs: Sequence[float],
) -> torch.Tensor:
    """x: (C, T) → (C, T)."""
    xb = x.unsqueeze(0)
    gains = torch.tensor([list(eq_gains_db)], device=x.device, dtype=x.dtype)
    knobs = torch.tensor([list(rev_knobs)], device=x.device, dtype=x.dtype)
    y = eq(xb, gains_db=gains)
    y = reverb(y, knobs=knobs)
    return y[0]


def rev_knob_vector(preset: Dict[str, object]) -> List[float]:
    wet = float(preset["wet_logit"])
    comb = [float(v) for v in preset["comb_fb_logits"]]  # type: ignore[arg-type]
    ap = [float(v) for v in preset["ap_logits"]]  # type: ignore[arg-type]
    return [wet, *comb, *ap]


def domain_paths(
    playlists_root: Path,
    domain: str,
    cyclegan_hash: str,
) -> Tuple[Path, Path, Path]:
    y_ckpt = (
        playlists_root / domain / "runs"
        / f"{domain}_run_8e1e614287" / "epoch_1500000.ckpt"
    )
    cycle_ckpt = (
        playlists_root / domain / "runs"
        / f"tap_{domain}_wf_cyc_mlp2_{cyclegan_hash}" / "cyclegan_latent.ckpt"
    )
    db_path = playlists_root / domain / "preprocessed"
    return y_ckpt, cycle_ckpt, db_path


@torch.no_grad()
def process_tap_segments(
    tap_segs: Sequence[torch.Tensor],
    *,
    backbone_x,
    backbone_y,
    warp_xy,
    mode: str,
    eq: BiquadBank,
    reverb: CausalReverb,
    eq_gains: Sequence[float],
    rev_knobs: Sequence[float],
    silence_sec: float,
    sr: int,
):
    """Interleaved demos + concat stems + pooled latents."""
    p1_parts: List[torch.Tensor] = []
    p2_parts: List[torch.Tensor] = []
    p3_parts: List[torch.Tensor] = []
    tap_stem: List[torch.Tensor] = []
    eq_stem: List[torch.Tensor] = []
    y1_stem: List[torch.Tensor] = []
    y2_stem: List[torch.Tensor] = []
    y3_stem: List[torch.Tensor] = []
    z1_list: List[torch.Tensor] = []
    z2_list: List[torch.Tensor] = []
    zxy_list: List[torch.Tensor] = []

    for seg in tap_segs:
        x = da.peak_normalize(seg)

        z1 = da.encode_mean(backbone_y, x)
        y1 = da.peak_normalize(decode_latent(backbone_y, z1, x.shape[-1])[0])

        x_eq = da.peak_normalize(
            apply_eq_reverb(
                x, eq=eq, reverb=reverb, eq_gains_db=eq_gains, rev_knobs=rev_knobs
            )
        )
        z2 = da.encode_mean(backbone_y, x_eq)
        y2 = da.peak_normalize(decode_latent(backbone_y, z2, x_eq.shape[-1])[0])

        y3 = da.peak_normalize(
            transfer_x_to_y(
                x.unsqueeze(0),
                backbone_x,
                backbone_y,
                warp_xy,
                mode=mode,  # type: ignore[arg-type]
            )[0]
        )
        z_x = da.encode_mean(backbone_x, x)
        z_xy = warp_xy(z_x)

        p1_parts.extend([x, y1])
        p2_parts.extend([x, x_eq, y2])
        p3_parts.extend([x, y3])
        tap_stem.append(x)
        eq_stem.append(x_eq)
        y1_stem.append(y1)
        y2_stem.append(y2)
        y3_stem.append(y3)
        z1_list.append(z1)
        z2_list.append(z2)
        zxy_list.append(z_xy)

    return (
        da.concat_with_silence(p1_parts, silence_sec, sr),
        da.concat_with_silence(p2_parts, silence_sec, sr),
        da.concat_with_silence(p3_parts, silence_sec, sr),
        torch.cat(tap_stem, dim=-1),
        torch.cat(eq_stem, dim=-1),
        torch.cat(y1_stem, dim=-1),
        torch.cat(y2_stem, dim=-1),
        torch.cat(y3_stem, dim=-1),
        torch.cat(z1_list, dim=-1),
        torch.cat(z2_list, dim=-1),
        torch.cat(zxy_list, dim=-1),
    )


def process_domain(
    domain: str,
    *,
    args: argparse.Namespace,
    backbone_x,
    tap_files: Sequence[Path],
    device: torch.device,
) -> None:
    if domain not in PHASE2_PRESETS:
        raise KeyError(f"no phase-2 preset for domain {domain!r}")
    preset = PHASE2_PRESETS[domain]
    y_ckpt, cycle_ckpt, db_path = domain_paths(
        args.playlists_root, domain, args.cyclegan_hash
    )
    for path, label in (
        (y_ckpt, "Y backbone"),
        (cycle_ckpt, "CycleGAN"),
        (db_path, "Y LMDB"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} missing for {domain}: {path}")

    print(f"\n===== {domain} =====")
    print(f"  Y: {y_ckpt}")
    print(f"  CycleGAN: {cycle_ckpt}")
    print(f"  segment_sec={args.segment_sec} max_sec={args.max_sec}")

    backbone_y = load_model(y_ckpt, use_gpu=args.gpu)
    warp_xy, _warp_yx, manifest = load_cyclegan_warps_from_checkpoint(
        str(cycle_ckpt),
        backbone_x=backbone_x,
        backbone_y=backbone_y,
    )
    warp_xy.eval()
    mode = manifest.canonicalizer_type

    eq = BiquadBank(sample_rate=float(backbone_y.sr)).to(device)
    reverb = CausalReverb(sample_rate=float(backbone_y.sr)).to(device)
    eq.eval()
    reverb.eval()

    eq_gains = [float(v) for v in preset["eq_gains_db"]]  # type: ignore[arg-type]
    rev_knobs = rev_knob_vector(preset)

    out_domain = args.out_dir / domain
    out_domain.mkdir(parents=True, exist_ok=True)
    (out_domain / "presets.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "intent": preset["intent"],
                "eq_gains_db": eq_gains,
                "reverb_knobs": {
                    "wet_logit": rev_knobs[0],
                    "comb_fb_logits": rev_knobs[1:5],
                    "ap_logits": rev_knobs[5:7],
                    "wet_approx": float(1 / (1 + np.exp(-rev_knobs[0]))),
                },
                "eq_band_hz_approx": [80, 191, 456, 1090, 2605, 6223],
                "segment_sec": args.segment_sec,
                "max_sec": args.max_sec,
                "silence_sec": args.silence_sec,
            },
            indent=2,
        )
        + "\n"
    )

    ood_p1: List[np.ndarray] = []
    ood_p2: List[np.ndarray] = []
    ood_p3: List[np.ndarray] = []
    sr = int(backbone_y.sr)

    for tap_path in tap_files:
        stem = da.clip_stem(tap_path)
        print(f"  clip: {tap_path.name} → {stem}")
        x = load_audio(tap_path, backbone_y, device=device)
        x = da.truncate(x, sr, args.max_sec)
        x = da.peak_normalize(x)
        tap_segs = da.split_segments(x, sr, args.segment_sec)
        print(f"    {len(tap_segs)} × {args.segment_sec:.1f}s segments")

        (
            demo1, demo2, demo3,
            stem_tap, stem_eq, stem_y1, stem_y2, stem_y3,
            z1_all, z2_all, zxy_all,
        ) = process_tap_segments(
            tap_segs,
            backbone_x=backbone_x,
            backbone_y=backbone_y,
            warp_xy=warp_xy,
            mode=mode,
            eq=eq,
            reverb=reverb,
            eq_gains=eq_gains,
            rev_knobs=rev_knobs,
            silence_sec=args.silence_sec,
            sr=sr,
        )

        ood_p1.append(latent_frames_to_points(z1_all, max_points=256))
        ood_p2.append(latent_frames_to_points(z2_all, max_points=256))
        ood_p3.append(latent_frames_to_points(zxy_all, max_points=256))

        clip_dir = out_domain / stem
        stems_dir = clip_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)

        save_audio(stems_dir / "00_tap.wav", stem_tap, sr)
        save_audio(stems_dir / "01_eq_reverb.wav", stem_eq, sr)
        save_audio(stems_dir / "02_phase1_recon.wav", stem_y1, sr)
        save_audio(stems_dir / "03_phase2_recon.wav", stem_y2, sr)
        save_audio(stems_dir / "04_cyclegan.wav", stem_y3, sr)

        save_audio(clip_dir / "phase1_demo.wav", demo1, sr)
        save_audio(clip_dir / "phase2_demo.wav", demo2, sr)
        save_audio(clip_dir / "phase3_demo.wav", demo3, sr)

        dn = da.domain_display_name(domain)
        da.write_phase_mel_triplet(
            clip_dir,
            stem_tap=stem_tap,
            stem_eq=stem_eq,
            stem_y1=stem_y1,
            stem_y2=stem_y2,
            stem_y3=stem_y3,
            title_prefix=dn,
            sr=sr,
            segment_sec=args.segment_sec,
        )

    print(f"  sampling in-domain latents from {db_path}")
    in_pts = da.collect_in_domain_points(
        backbone_y,
        db_path,
        n_signal=args.n_signal,
        n_crops=args.y_crops,
        max_points=args.pca_frames,
        device=device,
    )
    ood1 = np.concatenate(ood_p1, axis=0)
    ood2 = np.concatenate(ood_p2, axis=0)
    ood3 = np.concatenate(ood_p3, axis=0)

    da.write_domain_phase_pca_plots(
        out_domain,
        in_pts=in_pts,
        ood_by_phase=[
            (1, ood1, "Enc_Y(tap)"),
            (2, ood2, "Enc_Y(EQ+reverb(tap))"),
            (3, ood3, "W_xy(Enc_X(tap))"),
        ],
        title_prefix=da.domain_display_name(domain),
    )
    for phase in (1, 2, 3):
        print(f"  wrote pca_phase{phase}.png")

    del backbone_y, warp_xy, _warp_yx, eq, reverb
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tap_files = discover_tap_files(args.tap_dir)
    print("Tap clips:")
    for t in tap_files:
        print(f"  {t}")

    device = torch.device("cuda" if args.gpu else "cpu")
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("--gpu requested but CUDA is not available")

    print(f"Loading tap backbone X: {args.tap_backbone}")
    backbone_x = load_model(args.tap_backbone, use_gpu=args.gpu)

    for domain in args.domains:
        process_domain(
            domain,
            args=args,
            backbone_x=backbone_x,
            tap_files=tap_files,
            device=device,
        )

    print(f"\nDone. Outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
