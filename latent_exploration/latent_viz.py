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


def _panel_vlim(array: np.ndarray) -> tuple[float, float]:
    return float(array.min()), float(array.max())


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
) -> None:
    z_vmin, z_vmax = _panel_vlim(z)
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
    mel_vmin, mel_vmax = _panel_vlim(mel)

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
) -> Path:
    """Two-panel figure: input mel | latent heatmap."""
    mel = compute_mel(waveform, sample_rate)
    z = latent.squeeze(0).detach().cpu().numpy()
    mel_vmin, mel_vmax = _panel_vlim(mel)
    z_vmin, z_vmax = _panel_vlim(z)

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
