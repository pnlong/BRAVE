#!/usr/bin/env python3
"""Compare waveform vs latent CycleGAN (both latent warps).

Writes the same clip bundle as presentation phase-3 demos per model:
  ``{label}/phase3_demo.wav``, ``phase3_demo.png``, ``phase3_pca.png``, ``stems/``.

Default: babbling_brook wf_cyc_mlp2_48072 vs brook_buzz_B2 lat AE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_BRAVE = Path(__file__).resolve().parents[1]
for p in (_BRAVE / "latent_exploration", _BRAVE / "RAVE", _BRAVE / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_model import load_audio, load_model  # noqa: E402
from rave.canonicalizer.cycle_inference import (  # noqa: E402
    load_cyclegan_warps_from_checkpoint,
    transfer_x_to_y,
)

import demo_assets as da  # noqa: E402

BRAVE_STORAGE = Path(
    os.environ.get("BRAVE_STORAGE", f"/data/hai-res/{os.environ.get('USER', 'p1long')}/BRAVE-data")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--waveform-ckpt",
        type=Path,
        default=(
            BRAVE_STORAGE / "yt_playlists/babbling_brook/runs"
            / "tap_babbling_brook_wf_cyc_mlp2_48072c314a/cyclegan_latent.ckpt"
        ),
    )
    p.add_argument(
        "--latent-ckpt",
        type=Path,
        default=(
            BRAVE_STORAGE / "yt_playlists/babbling_brook/runs"
            / "tap_brook_buzz_B2_lat_ae_gan2_8f92317bd8/cyclegan_latent.ckpt"
        ),
    )
    p.add_argument(
        "--tap",
        type=Path,
        default=BRAVE_STORAGE / "tap_samples/audio_subset/sm57-0.wav",
    )
    p.add_argument(
        "--db-path-y",
        type=Path,
        default=BRAVE_STORAGE / "yt_playlists/babbling_brook/preprocessed",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=BRAVE_STORAGE / "presentation_demos/compare_wf_vs_latent_brook",
    )
    p.add_argument(
        "--domain",
        default="babbling_brook",
        help="Domain name for plot titles (display only)",
    )
    p.add_argument("--segment-sec", type=float, default=3.0)
    p.add_argument("--max-sec", type=float, default=15.0)
    p.add_argument("--silence-sec", type=float, default=0.0)
    p.add_argument("--y-crops", type=int, default=32)
    p.add_argument("--pca-frames", type=int, default=512)
    p.add_argument("--n-signal", type=int, default=65536)
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


@torch.no_grad()
def run_one(
    *,
    label: str,
    cycle_ckpt: Path,
    backbone_x,
    backbone_y,
    tap_segs,
    in_pts,
    out_dir: Path,
    title_prefix: str,
    silence_sec: float,
    segment_sec: float,
    sr: int,
    pca_frames: int,
) -> None:
    warp_xy, _, manifest = load_cyclegan_warps_from_checkpoint(
        str(cycle_ckpt),
        backbone_x=backbone_x,
        backbone_y=backbone_y,
    )
    warp_xy.eval()
    mode = manifest.canonicalizer_type
    print(
        f"{label}: cycle_domain={manifest.cycle_domain} warp={mode} ← {cycle_ckpt.name}"
    )

    def recon_fn(x: torch.Tensor) -> torch.Tensor:
        return transfer_x_to_y(
            x.unsqueeze(0), backbone_x, backbone_y, warp_xy, mode=mode
        )[0]

    def latent_fn(x: torch.Tensor) -> torch.Tensor:
        z_x = da.encode_mean(backbone_x, x)
        return warp_xy(z_x)

    bundle = da.build_tap_recon_segments(
        tap_segs,
        recon_fn=recon_fn,
        latent_fn=latent_fn,
        silence_sec=silence_sec,
        sr=sr,
    )

    clip_dir = out_dir / label
    da.write_tap_recon_clip_bundle(
        clip_dir,
        bundle,
        in_domain_pts=in_pts,
        sr=sr,
        segment_sec=segment_sec,
        title_prefix=f"{title_prefix} — {label.replace('_', ' ').title()} CycleGAN",
        pca_label_b="W_xy(Enc_X(tap))",
        phase_prefix="phase3",
        pca_max_points=pca_frames,
    )
    (clip_dir / "manifest.json").write_text(
        json.dumps(
            {
                "label": label,
                "cycle_ckpt": str(cycle_ckpt),
                "cycle_domain": manifest.cycle_domain,
                "canonicalizer_type": manifest.canonicalizer_type,
                "segment_sec": segment_sec,
                "silence_sec": silence_sec,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  wrote {clip_dir}/phase3_demo.wav + phase3_demo.png + phase3_pca.png")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.gpu else "cpu")
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("--gpu requested but CUDA unavailable")

    man_path = args.waveform_ckpt.parent / "cyclegan_latent.manifest.json"
    man = json.loads(man_path.read_text())

    print("Loading X", man["backbone_x_ckpt"])
    backbone_x = load_model(man["backbone_x_ckpt"], use_gpu=args.gpu)
    print("Loading Y", man["backbone_y_ckpt"])
    backbone_y = load_model(man["backbone_y_ckpt"], use_gpu=args.gpu)
    sr = int(backbone_y.sr)

    x = load_audio(args.tap, backbone_y, device=device)
    x = da.truncate(x, sr, args.max_sec)
    x = da.peak_normalize(x)
    tap_segs = da.split_segments(x, sr, args.segment_sec)
    clip_name = da.clip_stem(args.tap)
    print(f"tap={args.tap.name} ({clip_name}): {len(tap_segs)} × {args.segment_sec}s")

    print("Sampling in-domain Y latents…")
    in_pts = da.collect_in_domain_points(
        backbone_y,
        args.db_path_y,
        n_signal=args.n_signal,
        n_crops=args.y_crops,
        max_points=args.pca_frames,
        device=device,
    )

    title_prefix = da.domain_display_name(args.domain)
    out_dir = args.out_dir / clip_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_one(
        label="waveform",
        cycle_ckpt=args.waveform_ckpt,
        backbone_x=backbone_x,
        backbone_y=backbone_y,
        tap_segs=tap_segs,
        in_pts=in_pts,
        out_dir=out_dir,
        title_prefix=title_prefix,
        silence_sec=args.silence_sec,
        segment_sec=args.segment_sec,
        sr=sr,
        pca_frames=args.pca_frames,
    )
    run_one(
        label="latent",
        cycle_ckpt=args.latent_ckpt,
        backbone_x=backbone_x,
        backbone_y=backbone_y,
        tap_segs=tap_segs,
        in_pts=in_pts,
        out_dir=out_dir,
        title_prefix=title_prefix,
        silence_sec=args.silence_sec,
        segment_sec=args.segment_sec,
        sr=sr,
        pca_frames=args.pca_frames,
    )
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
