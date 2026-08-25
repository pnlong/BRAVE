"""Shared audio demo + evaluation visuals for BRAVE tap→Y experiments.

Reusable building blocks:
  - segment tap clips into fixed-length chunks
  - interleave demo WAVs (tap || recon || …)
  - joint in-domain + OOD latent PCA scatter plots
  - stacked mel spectrograms with chunk boundary barlines

Typical clip bundle (phase-1/3 or CycleGAN compare)::

    clip_dir/
      phase3_demo.wav
      phase3_demo.png
      pca.png
      stems/00_tap.wav
      stems/04_cyclegan.wav
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

_BRAVE = Path(__file__).resolve().parents[1]
for p in (_BRAVE / "latent_exploration", _BRAVE / "RAVE"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torchaudio  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

from load_model import save_audio  # noqa: E402
from rave.canonicalizer.viz import latent_frames_to_points  # noqa: E402
from rave.dataset import AudioDataset, LazyAudioDataset  # noqa: E402
from rave import transforms  # noqa: E402


# ---------------------------------------------------------------------------
# Naming / clip helpers
# ---------------------------------------------------------------------------


def clip_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace(".unnormalized", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "clip"


def domain_display_name(domain: str) -> str:
    """``rain_sounds`` → ``Rain Sounds``."""
    return domain.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Audio segment utilities
# ---------------------------------------------------------------------------


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
        segs.append(x[..., start : start + n])
    if not segs:
        segs.append(x)
    return segs


def _silence_tensor(channels: int, n_samples: int, device, dtype) -> torch.Tensor:
    return torch.zeros(channels, n_samples, device=device, dtype=dtype)


def concat_with_silence(
    segments: Sequence[torch.Tensor],
    silence_sec: float,
    sr: int,
) -> torch.Tensor:
    """Concatenate segments; optional silence gap before each segment after the first."""
    if not segments:
        raise ValueError("no segments to concatenate")
    device = segments[0].device
    dtype = segments[0].dtype
    ch = segments[0].shape[0]
    gap_n = max(0, int(round(silence_sec * sr)))
    parts: List[torch.Tensor] = []
    for i, seg in enumerate(segments):
        if i and gap_n > 0:
            parts.append(_silence_tensor(ch, gap_n, device, dtype))
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


# ---------------------------------------------------------------------------
# Latent PCA helpers
# ---------------------------------------------------------------------------


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
        chunks.append(
            latent_frames_to_points(z, max_points=max_points // max(1, n_crops))
        )
    pts = np.concatenate(chunks, axis=0)
    if pts.shape[0] > max_points:
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts


def fit_pca_project(
    in_domain: np.ndarray,
    ood: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint 2D PCA on in-domain + OOD (matches W&B ``plot_latent_domain_scatter``)."""
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


# ---------------------------------------------------------------------------
# Mel spectrogram plots
# ---------------------------------------------------------------------------


def waveform_to_mel_db(
    wave: torch.Tensor,
    sr: int,
    *,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Mono mel spectrogram in dB; shape (n_mels, n_frames)."""
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    mono = wave.mean(dim=0) if wave.shape[0] > 1 else wave[0]
    mono = mono.detach().cpu().float()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(mono.unsqueeze(0))
    mel_db = AF.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=10.0)
    return mel_db.squeeze(0).cpu().numpy()


def _draw_chunk_boundaries(
    axes: Sequence[plt.Axes],
    duration_sec: float,
    segment_sec: float,
) -> None:
    """Vertical dividers aligned with demo chunks (read down a column, then next)."""
    if segment_sec <= 0:
        return
    k = 1
    while k * segment_sec < duration_sec - 1e-6:
        t = k * segment_sec
        for ax in axes:
            ax.axvline(t, color="white", linewidth=1.0, alpha=0.85, zorder=10)
        k += 1


def save_stacked_mel_spectrograms(
    rows: Sequence[Tuple[torch.Tensor, str]],
    path: Path,
    *,
    title: str,
    sr: int,
    segment_sec: float | None = None,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> None:
    """Stack mel spectrograms vertically; shared time (s) on x-axis."""
    if len(rows) < 2:
        raise ValueError("need at least two rows")

    n_samples = min(int(w.shape[-1]) for w, _ in rows)
    trimmed = [w[..., :n_samples] for w, _ in rows]
    duration_sec = n_samples / sr

    mels = [
        waveform_to_mel_db(w, sr, n_fft=n_fft, hop_length=hop_length)
        for w in trimmed
    ]
    n_frames = min(m.shape[1] for m in mels)
    mels = [m[:, :n_frames] for m in mels]
    extent = [0.0, duration_sec, 0.0, float(mels[0].shape[0])]
    vmin = float(min(m.min() for m in mels))
    vmax = float(max(m.max() for m in mels))

    fig_h = max(3.0, 2.2 * len(rows))
    fig, axes = plt.subplots(
        len(rows),
        1,
        figsize=(10.5, fig_h),
        facecolor="none",
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    if len(rows) == 1:
        axes = [axes]

    im = None
    for ax, mel, ylab in zip(axes, mels, [lab for _, lab in rows]):
        ax.set_facecolor("none")
        im = ax.imshow(
            mel,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, duration_sec)

    if segment_sec is not None:
        _draw_chunk_boundaries(axes, duration_sec, segment_sec)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.colorbar(
        im,
        ax=axes,
        location="right",
        shrink=0.88,
        pad=0.04,
        aspect=25,
        label="dB",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=150,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def save_stacked_mel_spectrogram(
    input_w: torch.Tensor,
    recon_w: torch.Tensor,
    path: Path,
    *,
    title: str,
    sr: int,
    segment_sec: float | None = None,
    input_label: str = "input (tap)",
    recon_label: str = "reconstruction",
    hop_length: int = 512,
    n_fft: int = 2048,
) -> None:
    """Two-row mel stack (input vs reconstruction)."""
    save_stacked_mel_spectrograms(
        [(input_w, input_label), (recon_w, recon_label)],
        path,
        title=title,
        sr=sr,
        segment_sec=segment_sec,
        hop_length=hop_length,
        n_fft=n_fft,
    )


# ---------------------------------------------------------------------------
# Tap || recon segment bundles
# ---------------------------------------------------------------------------


@dataclass
class TapReconSegments:
    """Interleaved demo + column-aligned stems from fixed-length tap segments."""

    demo: torch.Tensor
    tap_stem: torch.Tensor
    recon_stem: torch.Tensor
    latent_frames: torch.Tensor


@torch.no_grad()
def build_tap_recon_segments(
    tap_segs: Sequence[torch.Tensor],
    *,
    recon_fn: Callable[[torch.Tensor], torch.Tensor],
    latent_fn: Callable[[torch.Tensor], torch.Tensor],
    silence_sec: float,
    sr: int,
) -> TapReconSegments:
    """Build tap||recon demo and stems; ``latent_fn`` receives peak-normalized tap."""
    parts: List[torch.Tensor] = []
    tap_stem: List[torch.Tensor] = []
    recon_stem: List[torch.Tensor] = []
    z_list: List[torch.Tensor] = []

    for seg in tap_segs:
        x = peak_normalize(seg)
        y = peak_normalize(recon_fn(x))
        parts.extend([x, y])
        tap_stem.append(x)
        recon_stem.append(y)
        z_list.append(latent_fn(x))

    return TapReconSegments(
        demo=concat_with_silence(parts, silence_sec, sr),
        tap_stem=torch.cat(tap_stem, dim=-1),
        recon_stem=torch.cat(recon_stem, dim=-1),
        latent_frames=torch.cat(z_list, dim=-1),
    )


def write_tap_recon_clip_bundle(
    out_dir: Path,
    bundle: TapReconSegments,
    *,
    in_domain_pts: np.ndarray,
    sr: int,
    segment_sec: float,
    title_prefix: str,
    pca_label_b: str,
    phase_prefix: str = "phase3",
    write_stems: bool = True,
    tap_stem_name: str = "00_tap.wav",
    recon_stem_name: str = "04_cyclegan.wav",
    pca_max_points: int = 512,
) -> None:
    """Write demo WAV, mel PNG, PCA PNG, and optional stem WAVs for one clip."""
    out_dir.mkdir(parents=True, exist_ok=True)

    save_audio(out_dir / f"{phase_prefix}_demo.wav", bundle.demo, sr)

    if write_stems:
        stems_dir = out_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        save_audio(stems_dir / tap_stem_name, bundle.tap_stem, sr)
        save_audio(stems_dir / recon_stem_name, bundle.recon_stem, sr)

    save_stacked_mel_spectrogram(
        bundle.tap_stem,
        bundle.recon_stem,
        out_dir / f"{phase_prefix}_demo.png",
        title=f"{title_prefix} — {phase_prefix.replace('_', ' ')} demo",
        sr=sr,
        segment_sec=segment_sec,
    )

    ood = latent_frames_to_points(bundle.latent_frames, max_points=pca_max_points)
    in_xy, ood_xy = fit_pca_project(in_domain_pts, ood)
    save_pca_scatter(
        in_xy,
        ood_xy,
        title=f"{title_prefix} — {phase_prefix.replace('_', ' ')} Y latent PCA",
        path=out_dir / f"{phase_prefix}_pca.png",
        label_b=pca_label_b,
    )


def write_phase_mel_triplet(
    clip_dir: Path,
    *,
    stem_tap: torch.Tensor,
    stem_eq: torch.Tensor,
    stem_y1: torch.Tensor,
    stem_y2: torch.Tensor,
    stem_y3: torch.Tensor,
    title_prefix: str,
    sr: int,
    segment_sec: float,
) -> None:
    """Phase 1/2/3 mel PNGs for presentation-style demos."""
    save_stacked_mel_spectrogram(
        stem_tap,
        stem_y1,
        clip_dir / "phase1_demo.png",
        title=f"{title_prefix} — phase 1 demo",
        sr=sr,
        segment_sec=segment_sec,
    )
    save_stacked_mel_spectrograms(
        [
            (stem_tap, "input (tap)"),
            (stem_eq, "EQ + reverb"),
            (stem_y2, "reconstruction"),
        ],
        clip_dir / "phase2_demo.png",
        title=f"{title_prefix} — phase 2 demo",
        sr=sr,
        segment_sec=segment_sec,
    )
    save_stacked_mel_spectrogram(
        stem_tap,
        stem_y3,
        clip_dir / "phase3_demo.png",
        title=f"{title_prefix} — phase 3 demo",
        sr=sr,
        segment_sec=segment_sec,
    )


def write_domain_phase_pca_plots(
    out_domain: Path,
    *,
    in_pts: np.ndarray,
    ood_by_phase: Sequence[Tuple[int, np.ndarray, str]],
    title_prefix: str,
) -> None:
    """Domain-level PCA scatter for phases 1–3 (aggregated over tap clips)."""
    for phase, ood, label_b in ood_by_phase:
        in_xy, ood_xy = fit_pca_project(in_pts, ood)
        save_pca_scatter(
            in_xy,
            ood_xy,
            title=f"{title_prefix} — phase {phase} Y latent PCA",
            path=out_domain / f"pca_phase{phase}.png",
            label_b=label_b,
        )
