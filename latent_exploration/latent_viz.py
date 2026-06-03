"""Seaborn-based mel + latent visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchaudio

sns.set_theme(style="darkgrid", context="notebook")

DEFAULT_CMAP = "magma"
DEFAULT_PCA_FIDELITY = 0.95
DEFAULT_CLIP_PERCENTILES = (2.0, 98.0)
MEL_N_FFT = 2048
MEL_HOP = 512
MEL_N_MELS = 128


def _save_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    return output_path


def compute_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    n_fft: int = MEL_N_FFT,
    hop_length: int = MEL_HOP,
    n_mels: int = MEL_N_MELS,
) -> np.ndarray:
    """Return log-mel spectrogram as [n_mels, time]."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mono = waveform.mean(dim=0, keepdim=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(mono)
    mel = torch.log1p(mel)
    return mel.squeeze(0).cpu().numpy()


def _as_numpy(z: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(z, torch.Tensor):
        return z.detach().cpu().numpy()
    return z


def latent_distribution_stats(z: torch.Tensor | np.ndarray) -> dict[str, float | int | tuple[int, ...]]:
    """Summary stats over all elements of a latent tensor [L, T] or [1, L, T]."""
    arr = np.asarray(_as_numpy(z)).astype(np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty latent array")
    raw = _as_numpy(z)
    shape = tuple(int(s) for s in raw.shape)
    qs = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "shape": shape,
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
    }


def _format_latent_stats_line(stats: dict[str, float | int | tuple[int, ...]]) -> str:
    return (
        f"shape={stats['shape']}  n={stats['n']}  "
        f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
        f"p01={stats['p01']:.4f}  p50={stats['p50']:.4f}  p99={stats['p99']:.4f}"
    )


def print_latent_distributions(
    z_vae: torch.Tensor,
    z_pca: torch.Tensor | None = None,
    *,
    vae_label: str = "VAE latent (post-mask)",
    pca_label: str = "PCA latent (post-mask)",
) -> None:
    """Print latent value distributions to stdout."""
    print(f"latent distribution [{vae_label}]:")
    print(f"  {_format_latent_stats_line(latent_distribution_stats(z_vae))}")
    if z_pca is not None:
        print(f"latent distribution [{pca_label}]:")
        print(f"  {_format_latent_stats_line(latent_distribution_stats(z_pca))}")


def plot_latent_distribution_histograms(
    z_vae: torch.Tensor,
    output_path: str | Path,
    z_pca: torch.Tensor | None = None,
    *,
    vae_label: str = "VAE latent",
    pca_label: str = "PCA latent",
    bins: int | str = "auto",
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> Path:
    """
    Histogram of latent values (count per bin).

    X-axis limits use ``clip_percentiles`` (same as ``--clip-percentile`` on the CLI).
    Bins are confined to that range so tails do not dominate the view.
    """
    vae_arr = np.asarray(_as_numpy(z_vae)).astype(np.float64).ravel()
    has_pca = z_pca is not None
    lo_pct, hi_pct = clip_percentiles

    if has_pca:
        pca_arr = np.asarray(_as_numpy(z_pca)).astype(np.float64).ravel()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        panels = [(axes[0], vae_arr, vae_label), (axes[1], pca_arr, pca_label)]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        panels = [(ax, vae_arr, vae_label)]

    for ax, arr, title in panels:
        lo, hi = np.percentile(arr, [lo_pct, hi_pct])
        if lo >= hi:
            lo, hi = float(arr.min()), float(arr.max())
        sns.histplot(
            arr,
            ax=ax,
            bins=bins,
            binrange=(lo, hi),
            stat="count",
            edgecolor=None,
        )
        ax.set_xlim(lo, hi)
        ax.set_title(f"{title} (p{lo_pct:g}–p{hi_pct:g})")
        ax.set_xlabel("latent value")
        ax.set_ylabel("count")

    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def _panel_vlim(
    array: np.ndarray,
    *,
    clip_outliers: bool = False,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> tuple[float, float]:
    if not clip_outliers:
        return float(array.min()), float(array.max())
    lo, hi = clip_percentiles
    vmin, vmax = np.percentile(array, [lo, hi])
    if vmin >= vmax:
        return float(array.min()), float(array.max())
    return float(vmin), float(vmax)


def _add_time_ticks(
    ax: plt.Axes,
    n_frames: int,
    seconds_per_frame: float,
    *,
    n_ticks: int = 5,
) -> None:
    if n_frames <= 1:
        return
    tick_idx = np.linspace(0, n_frames - 1, num=min(n_ticks, n_frames), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{i * seconds_per_frame:.2f}" for i in tick_idx])
    ax.set_xlabel("Time (s)")


def _draw_mask_overlay(
    ax: plt.Axes,
    mask_style: Literal["none", "temporal", "latent"] | None,
    mask_start: int | None,
    mask_width: int | None,
    latent_dim: int,
    time_frames: int,
) -> None:
    if mask_style in (None, "none") or mask_start is None or mask_width is None:
        return
    if mask_style == "temporal":
        ax.axvspan(
            mask_start - 0.5,
            mask_start + mask_width - 0.5,
            color="cyan",
            alpha=0.25,
            lw=0,
        )
    elif mask_style == "latent":
        ax.axhspan(
            mask_start - 0.5,
            mask_start + mask_width - 0.5,
            color="cyan",
            alpha=0.25,
            lw=0,
        )


def _plot_latent_panel(
    ax: plt.Axes,
    z: np.ndarray,
    *,
    title: str,
    cmap: str,
    compression_ratio: int,
    sample_rate: int,
    mask_style: Literal["none", "temporal", "latent"] | None = None,
    mask_start: int | None = None,
    mask_width: int | None = None,
    draw_mask: bool = True,
    clip_outliers: bool = False,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> None:
    z_vmin, z_vmax = _panel_vlim(
        z, clip_outliers=clip_outliers, clip_percentiles=clip_percentiles
    )
    sns.heatmap(
        z,
        ax=ax,
        cmap=cmap,
        vmin=z_vmin,
        vmax=z_vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title)
    _add_time_ticks(ax, z.shape[1], compression_ratio / sample_rate)
    if draw_mask:
        _draw_mask_overlay(
            ax,
            mask_style,
            mask_start,
            mask_width,
            z.shape[0],
            z.shape[1],
        )


def plot_mel_latent_and_pca(
    waveform: torch.Tensor,
    latent: torch.Tensor,
    latent_pca: torch.Tensor,
    sample_rate: int,
    compression_ratio: int,
    output_path: str | Path,
    *,
    vae_title: str = "Latent (VAE)",
    pca_title: str | None = None,
    pca_dims: int | None = None,
    pca_fidelity: float = DEFAULT_PCA_FIDELITY,
    cmap: str = DEFAULT_CMAP,
    mask_style: Literal["none", "temporal", "latent"] | None = None,
    mask_start: int | None = None,
    mask_width: int | None = None,
    mask_on_vae: bool = True,
    mask_on_pca: bool = True,
    clip_outliers: bool = False,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> Path:
    """Three-panel figure: input mel | VAE latent | PCA latent (optionally cropped)."""
    mel = compute_mel(waveform, sample_rate)
    z = latent.squeeze(0).detach().cpu().numpy()
    z_pca_full = latent_pca.squeeze(0).detach().cpu().numpy()
    base_pca_title = pca_title if pca_title is not None else "Latent (PCA)"
    if pca_dims is not None and pca_dims < z_pca_full.shape[0]:
        z_pca = z_pca_full[:pca_dims]
        pct = int(round(pca_fidelity * 100))
        pca_title = f"{base_pca_title}, top {pca_dims} @ {pct}% var"
    else:
        z_pca = z_pca_full
        pca_title = base_pca_title
    mel_vmin, mel_vmax = _panel_vlim(
        mel, clip_outliers=clip_outliers, clip_percentiles=clip_percentiles
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sns.heatmap(
        mel,
        ax=axes[0],
        cmap=cmap,
        vmin=mel_vmin,
        vmax=mel_vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title("Input mel spectrogram")
    _add_time_ticks(axes[0], mel.shape[1], MEL_HOP / sample_rate)

    _plot_latent_panel(
        axes[1],
        z,
        title=vae_title,
        cmap=cmap,
        compression_ratio=compression_ratio,
        sample_rate=sample_rate,
        mask_style=mask_style,
        mask_start=mask_start,
        mask_width=mask_width,
        draw_mask=mask_on_vae,
        clip_outliers=clip_outliers,
        clip_percentiles=clip_percentiles,
    )
    _plot_latent_panel(
        axes[2],
        z_pca,
        title=pca_title,
        cmap=cmap,
        compression_ratio=compression_ratio,
        sample_rate=sample_rate,
        mask_style=mask_style,
        mask_start=mask_start,
        mask_width=mask_width,
        draw_mask=mask_on_pca,
        clip_outliers=clip_outliers,
        clip_percentiles=clip_percentiles,
    )

    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved


def plot_mel_and_latent(
    waveform: torch.Tensor,
    latent: torch.Tensor,
    sample_rate: int,
    compression_ratio: int,
    output_path: str | Path,
    *,
    latent_title: str = "Latent (post-encode)",
    cmap: str = DEFAULT_CMAP,
    mask_style: Literal["none", "temporal", "latent"] | None = None,
    mask_start: int | None = None,
    mask_width: int | None = None,
    clip_outliers: bool = False,
    clip_percentiles: tuple[float, float] = DEFAULT_CLIP_PERCENTILES,
) -> Path:
    """Two-panel figure: input mel | latent heatmap."""
    mel = compute_mel(waveform, sample_rate)
    z = latent.squeeze(0).detach().cpu().numpy()
    mel_vmin, mel_vmax = _panel_vlim(
        mel, clip_outliers=clip_outliers, clip_percentiles=clip_percentiles
    )
    z_vmin, z_vmax = _panel_vlim(
        z, clip_outliers=clip_outliers, clip_percentiles=clip_percentiles
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.heatmap(
        mel,
        ax=axes[0],
        cmap=cmap,
        vmin=mel_vmin,
        vmax=mel_vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title("Input mel spectrogram")
    _add_time_ticks(axes[0], mel.shape[1], MEL_HOP / sample_rate)

    sns.heatmap(
        z,
        ax=axes[1],
        cmap=cmap,
        vmin=z_vmin,
        vmax=z_vmax,
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(latent_title)
    _add_time_ticks(
        axes[1],
        z.shape[1],
        compression_ratio / sample_rate,
    )
    _draw_mask_overlay(
        axes[1],
        mask_style,
        mask_start,
        mask_width,
        z.shape[0],
        z.shape[1],
    )

    saved = _save_figure(fig, output_path)
    plt.close(fig)
    return saved
