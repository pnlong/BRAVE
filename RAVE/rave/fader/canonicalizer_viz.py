"""Validation plots for canonicalizer training: latent scatter + audio."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

try:
    from sklearn.manifold import TSNE

    _HAS_TSNE = True
except ImportError:  # pragma: no cover
    _HAS_TSNE = False


def latent_frames_to_points(z: torch.Tensor, max_points: int = 512) -> np.ndarray:
    """(B, C, T) -> (N, C) subsampled frame vectors."""
    pts = z.detach().reshape(z.shape[0], z.shape[1], -1).permute(0, 2, 1)
    pts = pts.reshape(-1, z.shape[1]).cpu().numpy()
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def plot_latent_domain_scatter(
    in_domain: np.ndarray,
    ood: np.ndarray,
    *,
    method: Literal["pca", "tsne"] = "pca",
    title: str = "Latent space (validation)",
    max_points_per_domain: int = 512,
) -> plt.Figure:
    """
    2D scatter: in-domain vs OOD latents (post-canonicalizer path).
    """
    if in_domain.shape[0] > max_points_per_domain:
        in_domain = in_domain[
            np.random.choice(in_domain.shape[0], max_points_per_domain, replace=False)]
    if ood.shape[0] > max_points_per_domain:
        ood = ood[np.random.choice(ood.shape[0], max_points_per_domain, replace=False)]

    x = np.concatenate([in_domain, ood], axis=0)
    labels = np.array([0] * len(in_domain) + [1] * len(ood))

    if method == "tsne":
        if not _HAS_TSNE:
            method = "pca"
        elif x.shape[0] < 5:
            method = "pca"

    if method == "tsne":
        perplexity = min(30, max(2, x.shape[0] // 4))
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(x)
        xlab = "t-SNE 1"
        ylab = "t-SNE 2"
    else:
        xy = PCA(n_components=2).fit_transform(x)
        xlab = "PC 1"
        ylab = "PC 2"

    fig, ax = plt.subplots(figsize=(7, 6))
    mask_in = labels == 0
    mask_ood = labels == 1
    ax.scatter(
        xy[mask_in, 0], xy[mask_in, 1],
        c="#2a9d8f", alpha=0.45, s=12, label="in-domain",
    )
    ax.scatter(
        xy[mask_ood, 0], xy[mask_ood, 1],
        c="#e76f51", alpha=0.55, s=14, label="OOD",
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    import PIL.Image

    return np.asarray(PIL.Image.open(buf).convert("RGB"))


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def log_wandb_figure(module, key: str, fig: plt.Figure) -> None:
    import wandb

    logger = getattr(module, "logger", None)
    if logger is None or wandb.run is None:
        return
    trainer = getattr(module, "trainer", None)
    step = trainer.global_step if trainer is not None else None
    wandb.log({key: wandb.Image(figure_to_rgb_array(fig))}, step=step)


def log_wandb_audio(module, key: str, waveform: np.ndarray, sample_rate: int) -> None:
    import wandb

    from ..core import log_audio

    log_audio(module.logger, key, waveform, sample_rate, pl_module=module)


@torch.no_grad()
def recon_with_warp(
    trainer_module,
    x: torch.Tensor,
    attr_raw: torch.Tensor,
) -> torch.Tensor:
    """Single-item recon [C, T] through canonicalizer training path."""
    x_b = x.unsqueeze(0)
    attr_b = attr_raw.unsqueeze(0)
    _, _, _, y_raw, _, _, _ = trainer_module._forward_recon(x_b, attr_b)
    return y_raw[0]


def mono_waveform(wav: torch.Tensor) -> np.ndarray:
    """[C, T] -> 1D numpy."""
    if wav.dim() == 1:
        return wav.detach().cpu().numpy()
    return wav.mean(dim=0).detach().cpu().numpy()
