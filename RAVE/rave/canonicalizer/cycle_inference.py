"""Inference helpers for trained CycleGAN warp checkpoints."""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

from .config import CycleGANManifest, load_cyclegan_checkpoint
from .latent_canonicalizer import LatentCanonicalizer, infer_latent_warp_hparams
from .waveform_canonicalizer import build_waveform_canonicalizer


def build_cyclegan_warps(
    manifest: CycleGANManifest,
    *,
    backbone_x: nn.Module,
    backbone_y: nn.Module,
    warp_xy_state: dict,
    warp_yx_state: dict,
) -> Tuple[nn.Module, nn.Module]:
    mode = manifest.canonicalizer_type
    if mode == "latent":
        xy_h = infer_latent_warp_hparams(warp_xy_state)
        yx_h = infer_latent_warp_hparams(warp_yx_state)
        init_mode = getattr(manifest, "init_mode", "identity") or "identity"
        warp_xy = LatentCanonicalizer(
            latent_size=backbone_x.latent_size, init_mode=init_mode, **xy_h)
        warp_yx = LatentCanonicalizer(
            latent_size=backbone_y.latent_size, init_mode=init_mode, **yx_h)
    elif mode == "waveform":
        warp_xy = build_waveform_canonicalizer(
            sample_rate=backbone_y.sr,
            n_channels=backbone_y.n_channels,
        )
        warp_yx = build_waveform_canonicalizer(
            sample_rate=backbone_x.sr,
            n_channels=backbone_x.n_channels,
        )
    else:
        raise ValueError(f"unknown canonicalizer_type: {mode}")
    warp_xy.load_state_dict(warp_xy_state)
    warp_yx.load_state_dict(warp_yx_state)
    device = next(backbone_x.parameters()).device
    return warp_xy.to(device), warp_yx.to(device)


@torch.no_grad()
def _encode_mean(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z_raw, _ = backbone.encode(x, return_mb=True)
    from .. import blocks

    if isinstance(backbone.encoder, blocks.VariationalEncoder):
        return z_raw.chunk(2, dim=1)[0]
    z, _ = backbone.encoder.reparametrize(z_raw)[:2]
    return z


@torch.no_grad()
def transfer_x_to_y(
    x: torch.Tensor,
    backbone_x: nn.Module,
    backbone_y: nn.Module,
    warp_xy: nn.Module,
    *,
    mode: Literal["latent", "waveform"] = "latent",
) -> torch.Tensor:
    """Map domain-X waveform to domain-Y waveform."""
    if mode == "waveform":
        x_warp = warp_xy(x)
        z = _encode_mean(backbone_y, x_warp)
    else:
        z = warp_xy(_encode_mean(backbone_x, x))
    y_mb = backbone_y.decoder(z)
    if backbone_y.output_mode == "pqmf":
        from ..model import _pqmf_decode

        y = _pqmf_decode(
            backbone_y.pqmf,
            y_mb,
            batch_size=z.shape[:-2],
            n_channels=backbone_y.n_channels,
        )
    else:
        y = y_mb
    return y[..., : x.shape[-1]]


@torch.no_grad()
def transfer_y_to_x(
    y: torch.Tensor,
    backbone_x: nn.Module,
    backbone_y: nn.Module,
    warp_yx: nn.Module,
    *,
    mode: Literal["latent", "waveform"] = "latent",
) -> torch.Tensor:
    """Map domain-Y waveform to domain-X waveform."""
    if mode == "waveform":
        y_warp = warp_yx(y)
        z = _encode_mean(backbone_x, y_warp)
    else:
        z = warp_yx(_encode_mean(backbone_y, y))
    x_mb = backbone_x.decoder(z)
    if backbone_x.output_mode == "pqmf":
        from ..model import _pqmf_decode

        x = _pqmf_decode(
            backbone_x.pqmf,
            x_mb,
            batch_size=z.shape[:-2],
            n_channels=backbone_x.n_channels,
        )
    else:
        x = x_mb
    return x[..., : y.shape[-1]]


@torch.no_grad()
def transfer_waveform_blocked(
    transfer_fn,
    x: torch.Tensor,
    *,
    block_size: int = 512,
    left_context: int = 32768,
) -> torch.Tensor:
    """Chunked causal transfer approximating Max/nn~ ``forward block_size``.

    Training backbones are usually built *without* ``cached_conv`` streaming
    rings. Feeding each block with a long left context and keeping only the
    tail of the output approximates stateful streaming (same idea as overlap
    context for causal nets), so W&B ``val/audio_x_to_y_nn512`` tracks what
    listeners hear at small nn~ buffers.

    ``transfer_fn`` maps ``(B, C, T) -> (B, C, T)`` (e.g. ``G_xy`` waveform).
    ``x`` is ``(C, T)`` or ``(B, C, T)``.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"expected (B, C, T) or (C, T), got {tuple(x.shape)}")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    left_context = max(int(left_context), 0)

    outputs = []
    n = x.shape[-1]
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        valid = end - start
        ctx0 = max(0, start - left_context)
        window = x[..., ctx0:end]
        y_win = transfer_fn(window)
        # Align to window length (decoder may differ by a few samples).
        t = min(y_win.shape[-1], window.shape[-1])
        y_win = y_win[..., :t]
        take = min(valid, y_win.shape[-1])
        outputs.append(y_win[..., -take:])
    return torch.cat(outputs, dim=-1)


def load_cyclegan_warps_from_checkpoint(
    ckpt_path: str,
    *,
    backbone_x: nn.Module,
    backbone_y: nn.Module,
) -> Tuple[nn.Module, nn.Module, CycleGANManifest]:
    warp_xy_state, warp_yx_state, manifest = load_cyclegan_checkpoint(ckpt_path)
    warp_xy, warp_yx = build_cyclegan_warps(
        manifest,
        backbone_x=backbone_x,
        backbone_y=backbone_y,
        warp_xy_state=warp_xy_state,
        warp_yx_state=warp_yx_state,
    )
    return warp_xy, warp_yx, manifest
