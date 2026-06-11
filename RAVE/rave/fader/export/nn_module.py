"""
nn_tilde export wrapper for FaderRAVE — attribute knobs for Max/nn~.

Wraps FaderTraceModel with register_attribute / register_method so nn~ can
drive per-attribute sliders and optional torch-native extraction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import nn_tilde
import torch
import torch.nn as nn

from .torch_descriptors import TorchDescriptorExtract
from .trace_model import FaderTraceModel


def _default_raw_value(
    name: str,
    kind: str,
    min_max_features: Dict[str, Tuple[float, float]],
) -> float:
    if kind == "continuous":
        lo, hi = min_max_features.get(name, (0.0, 1.0))
        return float((lo + hi) * 0.5)
    return 0.0


class ScriptedFaderRAVE(nn_tilde.Module):
    """
    nn~-compatible Fader model with per-attribute knobs.

    Attributes (per row in attribute_stats order):
      {name}           — manual raw value (continuous float or discrete index)
      {name}_scale     — multiplier on normalized row (default 1.0)
      {name}_override  — use manual {name} instead of extracted (default false)
    Global:
      attr_mode        — 0=extract+scale/override, 1=manual-only, 2=extract-only
    """

    def __init__(
        self,
        core: FaderTraceModel,
        min_max_features: Dict[str, Tuple[float, float]],
        continuous_attributes: Sequence[str],
        n_channels: int = 1,
        target_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.n_channels = n_channels
        self.target_channels = target_channels or n_channels
        self.continuous_attributes = list(continuous_attributes)
        self.min_max_features = dict(min_max_features)

        kinds = core.attribute_kinds
        for name in core.attribute_names:
            kind = kinds.get(name, "continuous")
            default = _default_raw_value(name, kind, min_max_features)
            self.register_attribute(name, default)
            self.register_attribute(f"{name}_scale", 1.0)
            self.register_attribute(f"{name}_override", False)

        self.register_attribute("attr_mode", 0)

        self.extractor = TorchDescriptorExtract(
            continuous_attributes=self.continuous_attributes,
            sr=core.sr,
        )

        x_len = 2**14
        x = torch.zeros(1, n_channels, x_len)
        if core.pqmf is not None:
            core.pqmf(torch.zeros(1, 1, x_len))

        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]
        content_size = int(core.content_latent_size.item())
        num_attrs = int(core.num_attributes.item())
        decoder_size = content_size + num_attrs

        self.register_method(
            "encode",
            in_channels=n_channels,
            in_ratio=1,
            out_channels=content_size,
            out_ratio=ratio_encode,
            input_labels=[
                f"(signal) Channel {d}" for d in range(1, n_channels + 1)
            ],
            output_labels=[
                f"(signal) Latent dimension {i + 1}"
                for i in range(content_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=decoder_size,
            in_ratio=ratio_encode,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f"(signal) Latent dimension {i + 1}"
                for i in range(decoder_size)
            ],
            output_labels=[
                f"(signal) Channel {d}"
                for d in range(1, self.target_channels + 1)
            ],
        )
        self.register_method(
            "forward",
            in_channels=n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=1,
            input_labels=[
                f"(signal) Channel {d}" for d in range(1, n_channels + 1)
            ],
            output_labels=[
                f"(signal) Channel {d}"
                for d in range(1, self.target_channels + 1)
            ],
        )

    def _read_attr_scalar(self, name: str) -> torch.Tensor:
        return getattr(self, name)[0]

    def _build_manual_raw(
        self,
        batch: int,
        t_lat: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        d_total = int(self.core.num_attributes.item())
        raw = torch.zeros(batch, d_total, t_lat, device=device, dtype=dtype)
        for i, name in enumerate(self.core.attribute_names):
            val = self._read_attr_scalar(name)
            raw[:, i, :] = val
        return raw

    def _apply_scales(self, attr_norm: torch.Tensor) -> torch.Tensor:
        out = attr_norm.clone()
        for i, name in enumerate(self.core.attribute_names):
            scale = self._read_attr_scalar(f"{name}_scale")
            out[:, i, :] = out[:, i, :] * scale
        return out

    def _merge_raw(
        self,
        x: torch.Tensor,
        t_lat: int,
        manual: torch.Tensor,
    ) -> torch.Tensor:
        mode = int(self.attr_mode[0])
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype

        if mode == 1:
            return manual

        cont_raw = self.extractor(x, t_lat)
        extracted = torch.zeros_like(manual)
        cont_idx = {n: i for i, n in enumerate(self.continuous_attributes)}
        for i, name in enumerate(self.core.attribute_names):
            if name in cont_idx:
                extracted[:, i, :] = cont_raw[:, cont_idx[name], :]

        if mode == 2:
            for i, name in enumerate(self.core.attribute_names):
                kind = self.core.attribute_kinds.get(name, "continuous")
                if kind != "continuous":
                    extracted[:, i, :] = manual[:, i, :]
            return extracted

        raw = extracted.clone()
        for i, name in enumerate(self.core.attribute_names):
            override = self._read_attr_scalar(f"{name}_override") != 0
            if override:
                raw[:, i, :] = manual[:, i, :]
        return raw

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.encode(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.core.decode(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        t_lat = z.shape[-1]
        manual = self._build_manual_raw(
            z.shape[0], t_lat, z.device, z.dtype)
        raw = self._merge_raw(x, t_lat, manual)
        attr_norm = self._apply_scales(self.core.normalize_all(raw))
        z_c = torch.cat([z, attr_norm], dim=1)
        y = self.decode(z_c)
        if self.target_channels < y.shape[1]:
            y = y[:, : self.target_channels]
        return y

    @torch.jit.export
    def get_attr_mode(self) -> int:
        return int(self.attr_mode[0])

    @torch.jit.export
    def set_attr_mode(self, mode: int) -> int:
        self.attr_mode = (int(mode),)
        return 0
