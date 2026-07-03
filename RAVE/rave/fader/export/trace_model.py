"""
JIT-traced Fader model: encode → concat attr → decode.

Used by export_fader_ts.py for TorchScript / plugin deployment.
Strips training-only pieces (latent disc, GAN, stats mutation).

Inference path
--------------
  audio x  → encode → z (B, latent_size, T_lat)
  attr raw → normalize_all → attr_norm (B, D, T_lat)
  decode(cat(z, attr_norm)) → audio y

Port of neurorave realtime/trace.py for BRAVE FaderRAVE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gin
import torch
import torch.nn as nn

from ..attributes import load_attribute_stats


class FaderTraceModel(nn.Module):
    """
    Stripped FaderRAVE for TorchScript: encoder + widened decoder + pqmf.

    Buffers min/max for continuous attrs; discrete attrs use fixed num_classes.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pqmf: nn.Module,
        attribute_names: Sequence[str],
        attribute_kinds: Dict[str, str],
        min_max_features: Dict[str, Tuple[float, float]],
        discrete_num_classes: Dict[str, int],
        latent_size: int,
        sr: int,
        deterministic: bool = True,
        waveform_canonicalizer: Optional[nn.Module] = None,
        latent_canonicalizer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pqmf = pqmf
        self.waveform_canonicalizer = waveform_canonicalizer
        self.latent_canonicalizer = latent_canonicalizer
        self.attribute_names = list(attribute_names)
        self.attribute_kinds = dict(attribute_kinds)
        self.discrete_num_classes = dict(discrete_num_classes)
        self.latent_size = latent_size
        self.sr = sr
        self.deterministic = deterministic

        # --- Register min/max as buffers for JIT normalize ---
        mm = []
        is_cont: List[float] = []
        n_cls_row: List[float] = []
        for name in self.attribute_names:
            if self.attribute_kinds.get(name) == "continuous":
                lo, hi = min_max_features.get(name, (0.0, 1.0))
                mm.append([lo, hi])
                is_cont.append(1.0)
                n_cls_row.append(2.0)
            else:
                mm.append([0.0, 1.0])
                is_cont.append(0.0)
                n_cls_row.append(float(discrete_num_classes.get(name, 2)))
        self.register_buffer(
            "min_max_features",
            torch.tensor(mm, dtype=torch.float32),
        )
        self.register_buffer(
            "is_continuous",
            torch.tensor(is_cont, dtype=torch.float32),
        )
        self.register_buffer(
            "discrete_n_classes",
            torch.tensor(n_cls_row, dtype=torch.float32),
        )
        self.register_buffer(
            "num_attributes",
            torch.tensor(len(self.attribute_names)),
        )
        self.register_buffer(
            "content_latent_size",
            torch.tensor(latent_size),
        )

    @torch.jit.export
    def normalize_all(self, attr: torch.Tensor) -> torch.Tensor:
        """Normalize raw attr (B, D, T) to [-1,1] for decoder concat."""
        lo = self.min_max_features[:, 0].view(1, -1, 1)
        hi = self.min_max_features[:, 1].view(1, -1, 1)
        cont = 2.0 * ((attr - lo) / (hi - lo + 1e-8) - 0.5)

        n_cls = self.discrete_n_classes.view(1, -1, 1).clamp(min=2.0)
        idx = attr.long().clamp(min=0)
        disc = 2.0 * (idx.float() / (n_cls - 1.0)) - 1.0

        mask = self.is_continuous.view(1, -1, 1) > 0.5
        return torch.where(mask, cont, disc)

    def _encode_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.waveform_canonicalizer is not None:
            x = self.waveform_canonicalizer(x)
        # --- PQMF encode → VAE latent (deterministic mean if configured) ---
        if self.pqmf is not None:
            x = self.pqmf(x)
        z = self.encoder(x)
        if z.shape[1] % 2 == 0:
            mean, scale = torch.split(z, z.shape[1] // 2, dim=1)
            std = nn.functional.softplus(scale) + 1e-4
            if self.deterministic:
                z = mean
            else:
                z = mean + torch.randn_like(mean) * std
        else:
            z = z
        if self.latent_canonicalizer is not None:
            z = self.latent_canonicalizer(z)
        return z

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode audio (B, C, T) → content z (B, latent_size, T_lat)."""
        return self._encode_core(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode concatenated z (B, latent_size+D, T_lat) → audio."""
        y = self.decoder(z)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    @torch.jit.export
    def forward(self, x: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        """Encode → concat normalized attr → decode (main realtime entry point)."""
        z = self.encode(x)
        attr_n = self.normalize_all(attr)
        # --- Widened latent: content + control channels ---
        z_c = torch.cat([z, attr_n], dim=1)
        return self.decode(z_c)


def build_trace_model(
    fader_model,
    stats_path: str | Path,
    deterministic: bool = True,
) -> FaderTraceModel:
    """
    Build FaderTraceModel from trained FaderRAVE + attribute_stats.yaml.

    Copies encoder/decoder/pqmf weights; embeds min/max and discrete class
    counts as buffers for JIT-safe normalize_all().
    """
    stats = load_attribute_stats(stats_path)
    names = stats["attribute_names"]
    kinds = stats.get("attribute_kinds", {})
    return FaderTraceModel(
        encoder=fader_model.encoder,
        decoder=fader_model.decoder,
        pqmf=fader_model.pqmf,
        attribute_names=names,
        attribute_kinds=kinds,
        min_max_features=stats["min_max_features"],
        discrete_num_classes=stats.get("discrete_num_classes", {}),
        latent_size=fader_model.latent_size,
        sr=fader_model.sr,
        deterministic=deterministic,
        waveform_canonicalizer=getattr(fader_model, "waveform_canonicalizer", None),
        latent_canonicalizer=getattr(fader_model, "latent_canonicalizer", None),
    )
