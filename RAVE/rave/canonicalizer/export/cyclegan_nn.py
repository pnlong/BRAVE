"""nn~ export: Enc_X → W_xy → Dec_Y (CycleGAN X→Y timbre transfer)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Union

import cached_conv as cc
import nn_tilde
import torch
import torch.nn as nn

import rave
import rave.blocks
import rave.core

from ..config import load_cyclegan_checkpoint
from ..cycle_inference import build_cyclegan_warps
from ..gin_setup import configure_backbone_gin
from ..silence_gate import CausalLoudnessSidechain

_log = logging.getLogger(__name__)


def _ensure_cached_streaming_buffers(root: nn.Module) -> None:
    """Create cached_conv pad/cache buffers before ``torch.jit.script``.

    Those buffers are allocated lazily in ``@torch.jit.unused init_cache``.
    Any submodule that missed the warmup pass then fails scripting with
    ``CachedPadding1d has no attribute 'pad'``.
    """
    from cached_conv.convs import CachedConv1d, CachedConvTranspose1d, CachedPadding1d

    for m in root.modules():
        if isinstance(m, CachedConv1d):
            dummy = torch.zeros(1, m.in_channels, 32)
            if not getattr(m.cache, "initialized", 0):
                m.cache.init_cache(dummy)
            delay = getattr(m, "downsampling_delay", None)
            if delay is not None and not getattr(delay, "initialized", 0):
                delay.init_cache(dummy)
        elif isinstance(m, CachedConvTranspose1d):
            if not getattr(m, "initialized", 0):
                m.init_cache(torch.zeros(1, m.out_channels, 32))
        elif isinstance(m, CachedPadding1d) and not getattr(m, "initialized", 0):
            m.init_cache(torch.zeros(1, 1, max(int(m.padding), 1)))


def _load_backbone(config_path: str, ckpt_path: str, n_channels: int):
    configure_backbone_gin(config_path, n_channels)
    model = rave.RAVE(n_channels=n_channels)
    run = rave.core.search_for_run(ckpt_path)
    if run is None:
        raise FileNotFoundError(f"backbone checkpoint not found: {ckpt_path}")
    model = model.load_from_checkpoint(run)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _remove_weight_norm(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)


def load_cyclegan_xy_for_export(
    ckpt_path: Union[str, Path],
    *,
    n_channels: int = 1,
    streaming: bool = True,
):
    """Load Enc_X, Dec_Y, and W_xy from a ``cyclegan_latent.ckpt``."""
    cc.use_cached_conv(streaming)
    warp_xy_state, warp_yx_state, manifest = load_cyclegan_checkpoint(ckpt_path)
    if manifest.canonicalizer_type != "latent":
        raise ValueError(
            "CycleGAN nn~ export only supports latent warps "
            f"(got canonicalizer_type={manifest.canonicalizer_type!r})"
        )
    backbone_x = _load_backbone(
        manifest.backbone_x_config, manifest.backbone_x_ckpt, n_channels)
    backbone_y = _load_backbone(
        manifest.backbone_y_config, manifest.backbone_y_ckpt, n_channels)
    warp_xy, _warp_yx = build_cyclegan_warps(
        manifest,
        backbone_x=backbone_x,
        backbone_y=backbone_y,
        warp_xy_state=warp_xy_state,
        warp_yx_state=warp_yx_state,
    )
    warp_xy.eval()
    for p in warp_xy.parameters():
        p.requires_grad = False
    _remove_weight_norm(backbone_x)
    _remove_weight_norm(backbone_y)
    return backbone_x, backbone_y, warp_xy, manifest


class ScriptedCycleGANXY(nn_tilde.Module):
    """Streaming X→Y transfer: PQMF_X → Enc_X mean → W_xy → Dec_Y → iPQMF_Y."""

    def __init__(self, backbone_x, backbone_y, warp_xy) -> None:
        super().__init__()
        if not isinstance(backbone_x.encoder, rave.blocks.VariationalEncoder):
            raise ValueError("CycleGAN export requires VariationalEncoder backbones")
        self.encoder = backbone_x.encoder
        self.decoder = backbone_y.decoder
        self.pqmf_x = backbone_x.pqmf
        self.pqmf_y = backbone_y.pqmf
        self.warp_xy = warp_xy
        self.n_channels = backbone_x.n_channels
        self.target_channels = backbone_y.n_channels
        self.latent_size = backbone_x.latent_size
        self.input_mode = backbone_x.input_mode
        self.output_mode = backbone_y.output_mode
        self.sr = backbone_y.sr

        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        if self.pqmf_x is not None:
            self.pqmf_x(torch.zeros(1, 1, x_len))
        if self.pqmf_y is not None:
            self.pqmf_y(torch.zeros(1, 1, x_len))
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]
        self.encode_hop = ratio_encode
        self.loudness_sidechain = CausalLoudnessSidechain(hop=ratio_encode)
        self.register_attribute("sidechain", True)

        self.register_method(
            "encode",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=[
                f"(signal) Channel {d}" for d in range(1, self.n_channels + 1)
            ],
            output_labels=[
                f"(signal) Latent dimension {i + 1}"
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f"(signal) Latent dimension {i + 1}"
                for i in range(self.latent_size)
            ],
            output_labels=[
                f"(signal) Channel {d}"
                for d in range(1, self.target_channels + 1)
            ],
        )
        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f"(signal) Channel {d}" for d in range(1, self.n_channels + 1)
            ],
            output_labels=[
                f"(signal) Channel {d}"
                for d in range(1, self.target_channels + 1)
            ],
        )

    @torch.jit.export
    def get_sidechain(self) -> bool:
        return self.sidechain[0]

    @torch.jit.export
    def set_sidechain(self, sidechain: bool) -> int:
        self.sidechain = (sidechain,)
        return 0

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "pqmf":
            batch_size = x.shape[:-2]
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.pqmf_x(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))
        z_raw = self.encoder(x)
        z = z_raw.chunk(2, dim=1)[0]
        return self.warp_xy(z)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        if self.output_mode == "pqmf":
            batch_size = z.shape[:-2]
            y = y.reshape(y.shape[0] * self.target_channels, -1, y.shape[-1])
            y = self.pqmf_y.inverse(y)
            y = y.reshape(batch_size + (self.target_channels, -1))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.decode(self.encode(x))
        if y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]
        y_sc = self.loudness_sidechain(x, y)
        if self.sidechain[0]:
            return y_sc
        return y


def export_cyclegan_nn(
    ckpt_path: Union[str, Path],
    output_ts: Union[str, Path],
    *,
    n_channels: int = 1,
    streaming: bool = True,
) -> Path:
    """Write a streaming ``model.ts`` for CycleGAN X→Y transfer."""
    ckpt_path = Path(ckpt_path)
    output_ts = Path(output_ts)
    output_ts.parent.mkdir(parents=True, exist_ok=True)

    backbone_x, backbone_y, warp_xy, manifest = load_cyclegan_xy_for_export(
        ckpt_path, n_channels=n_channels, streaming=streaming)
    scripted = ScriptedCycleGANXY(backbone_x, backbone_y, warp_xy)
    x = torch.zeros(1, scripted.n_channels, 2**14)
    y = scripted(x)
    if y.shape[-1] == 0:
        raise RuntimeError("CycleGAN export warmup produced empty audio")
    _ensure_cached_streaming_buffers(scripted)
    # Warmup fills causal pad/cache rings with non-zero state (biases).
    # Serializing that into the .ts makes Max buzz on load / on silence.
    from rave.model import _zero_cached_conv_state

    _zero_cached_conv_state(scripted)

    scripted.export_to_ts(str(output_ts))
    sidecar = ckpt_path.with_suffix(".manifest.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, output_ts.parent / "cyclegan_latent.manifest.json")
    else:
        (output_ts.parent / "cyclegan_latent.manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2)
        )
    _log.info("CycleGAN nn~ export: %s", output_ts)
    return output_ts
