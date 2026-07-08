"""Loss helpers for canonicalizer Stage-1 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import core


def frame_rms_curve(
    x: torch.Tensor,
    n_frames: int,
    n_fft: int = 2048,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable frame-wise RMS envelope aligned to latent time steps.

    Args:
        x: (B, C, T) waveform
        n_frames: target number of frames (typically T_lat)

    Returns:
        (B, n_frames) RMS curve
    """
    mono = x.mean(dim=1)
    hop = max(1, n_fft // 4)
    if mono.shape[-1] < n_fft:
        mono = F.pad(mono, (0, n_fft - mono.shape[-1]))
    frames = mono.unfold(-1, n_fft, hop)
    rms = frames.pow(2).mean(dim=-1).sqrt().clamp_min(eps)
    if rms.shape[-1] != n_frames:
        rms = F.interpolate(
            rms.unsqueeze(1),
            size=n_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
    return rms


def rms_recon_l1(
    y: torch.Tensor,
    target: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    """L1 loss between RMS curves of reconstruction and target waveforms."""
    n_frames = min(n_frames, y.shape[-1], target.shape[-1])
    if n_frames < 1:
        return torch.tensor(0.0, device=y.device, dtype=y.dtype)
    t = min(y.shape[-1], target.shape[-1])
    r_y = frame_rms_curve(y[..., :t], n_frames)
    r_t = frame_rms_curve(target[..., :t], n_frames)
    return F.l1_loss(r_y, r_t)


def resolve_gan_loss(name: str):
    """Map gin string to RAVE GAN loss callable."""
    table = {
        "hinge": core.hinge_gan,
        "ls": core.ls_gan,
        "nonsaturating": core.nonsaturating_gan,
        "logistic": core.nonsaturating_gan,
    }
    if name not in table:
        raise ValueError(f"unknown gan_loss: {name!r}; choose from {list(table)}")
    return table[name]
