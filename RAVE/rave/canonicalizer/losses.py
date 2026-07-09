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


def normalize_loss(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Divide by a fixed reference scale so heterogeneous terms are ~O(1)."""
    return value / max(scale, 1e-8)


def weighted_recon_loss(
    stft: torch.Tensor,
    rms: torch.Tensor,
    *,
    stft_weight: float,
    rms_weight: float,
    stft_scale: float,
    rms_scale: float,
) -> torch.Tensor:
    """Normalized recon: w_stft * (STFT/s_stft) + w_rms * (RMS/s_rms)."""
    return (
        stft_weight * normalize_loss(stft, stft_scale)
        + rms_weight * normalize_loss(rms, rms_scale)
    )


def empirical_loss_scale(values: list[float], min_scale: float = 1e-3) -> float:
    """Mean raw loss over calibration batches; clamped away from zero."""
    if not values:
        raise ValueError("empirical_loss_scale requires at least one value")
    return max(sum(values) / len(values), min_scale)


def empirical_adversarial_loss_scale(
    values: list[float],
    fallback: float,
    min_scale: float = 1e-3,
    *,
    min_mean_fraction: float = 0.1,
) -> float:
    """
    Calibrate GAN/FM scales, but keep gin fallbacks when identity-warp startup
    yields a near-zero adversarial signal (common before the GAN ramp).
    """
    if not values:
        return fallback
    mean = sum(values) / len(values)
    if mean <= min_scale or mean < fallback * min_mean_fraction:
        return fallback
    return max(mean, min_scale)
