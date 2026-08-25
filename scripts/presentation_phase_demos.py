#!/usr/bin/env python3
"""Presentation demos: phase-1/2/3 tap→Y audio + Y-latent PCA plots.

Phase 1: tap → Enc_Y → Dec_Y (zero-shot)
Phase 2: tap → EQ+reverb(preset) → Enc_Y → Dec_Y
Phase 3: tap → Enc_X → W_xy → Dec_Y (CycleGAN)

Demo WAVs interleave fixed-length segments with silence:
  phase1: tap0 || recon0 || tap1 || recon1 || …
  phase2: tap0 || eq0 || recon0 || tap1 || eq1 || recon1 || …
  phase3: tap0 || cyc0 || tap1 || cyc1 || …
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

_BRAVE = Path(__file__).resolve().parents[1]
_LATENT = _BRAVE / "latent_exploration"
_RAVE = _BRAVE / "RAVE"
for p in (_LATENT, _RAVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_model import load_audio, load_model, save_audio  # noqa: E402
from rave.canonicalizer.cycle_inference import (  # noqa: E402
    load_cyclegan_warps_from_checkpoint,
    transfer_x_to_y,
)
from rave.canonicalizer.viz import latent_frames_to_points  # noqa: E402
from rave.dataset import AudioDataset, LazyAudioDataset  # noqa: E402
from rave.dsp import BiquadBank, CausalReverb  # noqa: E402
from rave import transforms  # noqa: E402
import yaml  # noqa: E402

SCRATCH = Path("/data/scratch-fast/p1long/BRAVE")
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
        default=SCRATCH / "yt_playlists",
    )
    p.add_argument(
        "--tap-dir",
        type=Path,
        default=SCRATCH / "tap_samples" / "audio_subset",
    )
    p.add_argument(
        "--tap-backbone",
        type=Path,
        default=(
            SCRATCH / "tap_samples" / "runs"
            / "tap_uncond_run_8e1e614287" / "epoch_1500000.ckpt"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRATCH / "presentation_demos",
    )
    p.add_argument(
        "--cyclegan-hash",
        default="48072c314a",
        help="Suffix of tap_<domain>_wf_cyc_mlp2_<hash> runs",
    )
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--silence-sec", type=float, default=0.25)
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


def clip_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace(".unnormalized", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "clip"


def domain_display_name(domain: str) -> str:
    """``rain_sounds`` → ``Rain Sounds`` (title case, spaces)."""
    return domain.replace("_", " ").title()


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


def truncate(x: torch.Tensor, sr: int, max_sec: float) -> torch.Tensor:
    if max_sec is None or max_sec <= 0:
        return x
    n = int(max_sec * sr)
    return x[..., :n]


def split_segments(
    x: torch.Tensor,
    sr: int,
    segment_sec: float,
) -> List[torch.Tensor]:
    """Non-overlapping full segments of ``segment_sec`` (drop a short tail)."""
    n = max(1, int(round(segment_sec * sr)))
    segs: List[torch.Tensor] = []
    for start in range(0, x.shape[-1] - n + 1, n):
        segs.append(x[..., start:start + n])
    if not segs:
        segs.append(x)
    return segs


def silence_tensor(channels: int, n_samples: int, device, dtype) -> torch.Tensor:
    return torch.zeros(channels, n_samples, device=device, dtype=dtype)


def concat_with_silence(
    segments: Sequence[torch.Tensor],
    silence_sec: float,
    sr: int,
) -> torch.Tensor:
    if not segments:
        raise ValueError("no segments to concatenate")
    device = segments[0].device
    dtype = segments[0].dtype
    ch = segments[0].shape[0]
    gap_n = max(0, int(round(silence_sec * sr)))
    parts: List[torch.Tensor] = []
    for i, seg in enumerate(segments):
        if i and gap_n > 0:
            parts.append(silence_tensor(ch, gap_n, device, dtype))
        parts.append(seg)
    return torch.cat(parts, dim=-1)


def peak_normalize(x: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    m = x.abs().amax().clamp_min(1e-8)
    return x * (peak / m)


def encode_mean(backbone, x: torch.Tensor) -> torch.Tensor:
    """VAE mean latent; x is (B, C, T) or (C, T)."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return backbone.encode_to_latent(x, use_mean=True)


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


def build_lmdb_crop_dataset(db_path: Path, n_signal: int, n_channels: int = 1):
    with open(db_path / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    lazy = bool(meta.get("lazy", False))
    transform = transforms.Compose([
        lambda x: x.astype(np.float32),
        transforms.RandomCrop(n_signal),
        transforms.Dequantize(16),
        lambda x: x.astype(np.float32),
    ])
    if lazy:
        return LazyAudioDataset(
            str(db_path), n_signal, int(meta.get("sr", 44100)), transform, n_channels
        )
    return AudioDataset(
        str(db_path), transforms=transform, n_channels=n_channels, show_progress=False
    )


@torch.no_grad()
def collect_in_domain_points(
    backbone_y,
    db_path: Path,
    *,
    n_signal: int,
    n_crops: int,
    max_points: int,
    device: torch.device,
) -> np.ndarray:
    ds = build_lmdb_crop_dataset(db_path, n_signal, backbone_y.n_channels)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(n_crops, len(ds)), replace=False)
    chunks: List[np.ndarray] = []
    for i in idxs:
        x = torch.from_numpy(np.asarray(ds[int(i)])).float().to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = encode_mean(backbone_y, x)
        chunks.append(latent_frames_to_points(z, max_points=max_points // max(1, n_crops)))
    pts = np.concatenate(chunks, axis=0)
    if pts.shape[0] > max_points:
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts


def fit_pca_project(
    in_domain: np.ndarray,
    ood: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint 2D PCA on in-domain + OOD (same as W&B ``plot_latent_domain_scatter``).

    Fitting on in-domain alone projects OOD onto Y's plane and can fake overlap
    even when 128-D nearest-neighbor distance is large.
    """
    n_in = len(in_domain)
    pca = PCA(n_components=2)
    xy = pca.fit_transform(np.concatenate([in_domain, ood], axis=0))
    return xy[:n_in], xy[n_in:]


def save_pca_scatter(
    in_xy: np.ndarray,
    ood_xy: np.ndarray,
    *,
    title: str,
    path: Path,
    label_a: str = "in-domain Y",
    label_b: str = "OOD (tap)",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="none")
    ax.set_facecolor("none")
    ax.scatter(
        in_xy[:, 0], in_xy[:, 1],
        c="#2a9d8f", alpha=0.45, s=12, label=label_a,
    )
    ax.scatter(
        ood_xy[:, 0], ood_xy[:, 1],
        c="#e76f51", alpha=0.55, s=14, label=label_b,
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    legend = ax.legend(loc="best", framealpha=0.0)
    legend.get_frame().set_facecolor("none")
    legend.get_frame().set_edgecolor("none")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


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
        x = peak_normalize(seg)

        z1 = encode_mean(backbone_y, x)
        y1 = peak_normalize(decode_latent(backbone_y, z1, x.shape[-1])[0])

        x_eq = peak_normalize(
            apply_eq_reverb(
                x, eq=eq, reverb=reverb, eq_gains_db=eq_gains, rev_knobs=rev_knobs
            )
        )
        z2 = encode_mean(backbone_y, x_eq)
        y2 = peak_normalize(decode_latent(backbone_y, z2, x_eq.shape[-1])[0])

        y3 = peak_normalize(
            transfer_x_to_y(
                x.unsqueeze(0),
                backbone_x,
                backbone_y,
                warp_xy,
                mode=mode,  # type: ignore[arg-type]
            )[0]
        )
        z_x = encode_mean(backbone_x, x)
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
        concat_with_silence(p1_parts, silence_sec, sr),
        concat_with_silence(p2_parts, silence_sec, sr),
        concat_with_silence(p3_parts, silence_sec, sr),
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
        stem = clip_stem(tap_path)
        print(f"  clip: {tap_path.name} → {stem}")
        x = load_audio(tap_path, backbone_y, device=device)
        x = truncate(x, sr, args.max_sec)
        x = peak_normalize(x)
        tap_segs = split_segments(x, sr, args.segment_sec)
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

    print(f"  sampling in-domain latents from {db_path}")
    in_pts = collect_in_domain_points(
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

    for phase, ood, label_b in (
        (1, ood1, "Enc_Y(tap)"),
        (2, ood2, "Enc_Y(EQ+reverb(tap))"),
        (3, ood3, "W_xy(Enc_X(tap))"),
    ):
        in_xy, ood_xy = fit_pca_project(in_pts, ood)
        save_pca_scatter(
            in_xy,
            ood_xy,
            title=f"{domain_display_name(domain)} — phase {phase} Y latent PCA",
            path=out_domain / f"pca_phase{phase}.png",
            label_b=label_b,
        )
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
