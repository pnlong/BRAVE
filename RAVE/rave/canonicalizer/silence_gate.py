"""Hop-aligned RMS, mute / replace-z, and causal loudness sidechain.

Offline diagnosis plus CycleGAN nn~ (``CausalLoudnessSidechain``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_DB = 1e-12


def as_btc(x: torch.Tensor) -> torch.Tensor:
    """``(T,)`` / ``(C, T)`` / ``(B, C, T)`` → ``(B, C, T)``."""
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"expected 1–3D waveform, got {tuple(x.shape)}")


def latent_hop(n_samples: int, n_frames: int) -> int:
    """Audio samples per latent frame (truncate remainder)."""
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    return max(1, int(n_samples) // int(n_frames))


def frame_rms(x: torch.Tensor, hop: int, eps: float = EPS_DB) -> torch.Tensor:
    """Non-overlapping hop RMS. ``x`` is ``(B, C, T)`` or ``(C, T)``.

    Returns ``(B, n_frames)`` with ``n_frames = T // hop``.
    """
    if hop < 1:
        raise ValueError(f"hop must be positive, got {hop}")
    xb = as_btc(x)
    n_frames = xb.shape[-1] // hop
    if n_frames < 1:
        raise ValueError(
            f"waveform length {xb.shape[-1]} shorter than hop {hop}"
        )
    clipped = xb[..., : n_frames * hop]
    frames = clipped.reshape(*clipped.shape[:-1], n_frames, hop)
    return frames.pow(2).mean(dim=(1, -1)).sqrt().clamp_min(eps)


def rms_to_dbfs(rms: torch.Tensor, eps: float = EPS_DB) -> torch.Tensor:
    return 20.0 * torch.log10(rms.clamp_min(eps))


def frame_rms_dbfs(x: torch.Tensor, hop: int, eps: float = EPS_DB) -> torch.Tensor:
    return rms_to_dbfs(frame_rms(x, hop, eps=eps), eps=eps)


def global_rms_dbfs(x: torch.Tensor, eps: float = EPS_DB) -> float:
    xb = as_btc(x)
    rms = xb.pow(2).mean().clamp_min(eps).sqrt()
    return float(rms_to_dbfs(rms).item())


def hold_quiet_mask(quiet: torch.Tensor, hold_frames: int) -> torch.Tensor:
    """Mute where the last ``hold_frames`` frames (ending at t) are all quiet.

    ``quiet`` is boolean ``(B, T)`` or ``(T,)``. ``hold_frames <= 1`` is identity.
    """
    if hold_frames < 1:
        raise ValueError(f"hold_frames must be >= 1, got {hold_frames}")
    q = quiet
    squeezed = False
    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeezed = True
    if hold_frames == 1:
        out = q
    else:
        qf = q.to(dtype=torch.float32)
        k = int(hold_frames)
        padded = F.pad(qf.unsqueeze(1), (k - 1, 0))
        kernel = torch.ones(1, 1, k, device=q.device, dtype=qf.dtype)
        summed = F.conv1d(padded, kernel).squeeze(1)
        out = summed >= (k - 1e-5)
    return out[0] if squeezed else out


def expand_frame_mask(mask: torch.Tensor, hop: int, n_samples: int) -> torch.Tensor:
    """``(B, n_frames)`` or ``(n_frames,)`` → boolean ``(..., n_samples)`` (tail False)."""
    squeezed = mask.dim() == 1
    m = mask.unsqueeze(0) if squeezed else mask
    expanded = m.unsqueeze(-1).expand(*m.shape, hop).reshape(m.shape[0], m.shape[1] * hop)
    if expanded.shape[-1] < n_samples:
        expanded = F.pad(expanded, (0, n_samples - expanded.shape[-1]))
    else:
        expanded = expanded[..., :n_samples]
    return expanded[0] if squeezed else expanded


def mute_by_input_rms(
    y: torch.Tensor,
    x: torch.Tensor,
    hop: int,
    *,
    threshold_db: float = -40.0,
    hold_frames: int = 2,
) -> torch.Tensor:
    """Zero output samples whose corresponding input hop is quiet (with hold).

    ``x`` / ``y`` share time length (or ``y`` is trimmed to ``x``).
    """
    yb = as_btc(y)
    xb = as_btc(x)
    t = min(yb.shape[-1], xb.shape[-1])
    yb = yb[..., :t]
    xb = xb[..., :t]
    rms_db = frame_rms_dbfs(xb, hop)
    quiet = rms_db < threshold_db
    mute = hold_quiet_mask(quiet, hold_frames)
    sample_mute = expand_frame_mask(mute, hop, t)
    while sample_mute.dim() < yb.dim():
        sample_mute = sample_mute.unsqueeze(1)
    out = yb.masked_fill(sample_mute, 0)
    if y.dim() == 2:
        return out[0]
    if y.dim() == 1:
        return out[0, 0]
    return out


def replace_latent_on_quiet(
    z: torch.Tensor,
    z_fill: torch.Tensor,
    rms_db: torch.Tensor,
    *,
    threshold_db: float = -40.0,
    hold_frames: int = 2,
) -> torch.Tensor:
    """Replace quiet latent frames with ``z_fill`` (broadcast over batch/time).

    ``z`` is ``(B, C, T)``; ``z_fill`` is ``(B, C, T)``, ``(B, C, 1)``, or ``(C,)``.
    ``rms_db`` is ``(B, T)`` aligned to ``z`` time (trim to min length).
    """
    t = min(z.shape[-1], rms_db.shape[-1])
    z = z[..., :t]
    rms_db = rms_db[..., :t]
    mute = hold_quiet_mask(rms_db < threshold_db, hold_frames)
    fill = z_fill
    if fill.dim() == 1:
        fill = fill.view(1, -1, 1).expand_as(z)
    elif fill.shape[-1] == 1:
        fill = fill.expand_as(z)
    elif fill.shape[-1] != t:
        fill = fill[..., :t]
    mask = mute.unsqueeze(1)
    return torch.where(mask, fill, z)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def smooth_causal_envelope(rms: torch.Tensor, kernel_frames: int) -> torch.Tensor:
    """Causal boxcar on hop RMS. ``rms`` is ``(B, T)``. ``kernel_frames`` 1 = identity."""
    k = max(1, int(kernel_frames))
    if k == 1:
        return rms
    padded = F.pad(rms.unsqueeze(1), (k - 1, 0))
    kernel = torch.ones(1, 1, k, device=rms.device, dtype=rms.dtype) / k
    return F.conv1d(padded, kernel).squeeze(1)


def upsample_envelope(env: torch.Tensor, hop: int, n_samples: int) -> torch.Tensor:
    """Linear-interpolate hop envelope ``(B, n_frames)`` to ``(B, n_samples)``."""
    if env.dim() != 2:
        raise ValueError(f"expected (B, T) envelope, got {tuple(env.shape)}")
    n_frames = env.shape[-1]
    if n_frames < 1:
        raise ValueError("empty envelope")
    # Place hop values at hop centers, then interpolate to samples.
    stretched = F.interpolate(
        env.unsqueeze(1),
        size=n_frames * hop,
        mode="linear",
        align_corners=True,
    ).squeeze(1)
    if stretched.shape[-1] < n_samples:
        stretched = F.pad(stretched, (0, n_samples - stretched.shape[-1]))
    else:
        stretched = stretched[..., :n_samples]
    return stretched


def sidechain_by_input_loudness(
    y: torch.Tensor,
    x: torch.Tensor,
    hop: int,
    *,
    threshold_db: float = -40.0,
    smooth_frames: int = 8,
) -> torch.Tensor:
    """Scale output by a smoothed input RMS envelope (soft sidechain).

    Gain is ``rms / (rms + knee)`` with ``knee`` at ``threshold_db``, so loud
    taps stay near unity and quiet hops duck without a hard mute. Envelope is
    causally smoothed then linearly upsampled to samples.
    """
    yb = as_btc(y)
    xb = as_btc(x)
    t = min(yb.shape[-1], xb.shape[-1])
    yb = yb[..., :t]
    xb = xb[..., :t]
    rms = frame_rms(xb, hop)
    rms = smooth_causal_envelope(rms, smooth_frames)
    knee = _db_to_lin(threshold_db)
    gain_h = rms / (rms + knee)
    gain = upsample_envelope(gain_h, hop, t)
    while gain.dim() < yb.dim():
        gain = gain.unsqueeze(1)
    out = yb * gain
    if y.dim() == 2:
        return out[0]
    if y.dim() == 1:
        return out[0, 0]
    return out


class CausalLoudnessSidechain(nn.Module):
    """Streaming input-loudness sidechain (Gate C) for nn~ blocks.

    Causal IIR on hop RMS, then ``g = rms / (rms + knee)``. ``env`` carries
    state across 512-sample ``forward`` calls.
    """

    hop: int
    smooth_frames: int

    def __init__(
        self,
        hop: int,
        knee_db: float = -40.0,
        smooth_frames: int = 8,
    ) -> None:
        super().__init__()
        if hop < 1:
            raise ValueError(f"hop must be positive, got {hop}")
        self.hop = int(hop)
        self.smooth_frames = max(1, int(smooth_frames))
        self.register_buffer(
            "knee", torch.tensor(_db_to_lin(knee_db), dtype=torch.float32)
        )
        self.register_buffer("env", torch.zeros(1))

    def reset(self) -> None:
        self.env.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hop = self.hop
        t = min(int(x.shape[-1]), int(y.shape[-1]))
        x = x[..., :t]
        y = y[..., :t]
        n_frames = t // hop
        alpha = 1.0 / float(self.smooth_frames)
        knee = self.knee
        env = self.env[0]
        if n_frames < 1:
            rms = torch.sqrt(x.pow(2).mean() + EPS_DB)
            env = (1.0 - alpha) * env + alpha * rms
            self.env.fill_(env)
            return y * (env / (env + knee))
        mono = x.mean(dim=1)
        clipped = mono[..., : n_frames * hop]
        rms = clipped.reshape(clipped.shape[0], n_frames, hop).pow(2).mean(-1).sqrt()
        gains = []
        for i in range(n_frames):
            r = rms[:, i].mean()
            env = (1.0 - alpha) * env + alpha * r
            gains.append(env / (env + knee))
        self.env.fill_(env)
        g_h = torch.stack(gains)
        g_samp = g_h.repeat_interleave(hop)
        if g_samp.numel() < t:
            g_samp = torch.cat([g_samp, g_h[-1].expand(t - g_samp.numel())])
        return y * g_samp.view(1, 1, -1)

