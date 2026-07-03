"""
nn_tilde export wrapper for FaderRAVE — attribute knobs for Max/nn~.

Wraps FaderTraceModel with register_attribute / register_method so nn~ can
drive per-attribute sliders and optional torch-native extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nn_tilde
import torch
import torch.nn as nn

from .torch_descriptors import (
    TORCH_EXTRACTABLE,
    TorchDescriptorExtract,
    calibrate_torch_descriptor_extract,
)
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


def _reader_method_source(
    method_name: str,
    attr_names: Sequence[str],
    *,
    suffix: str = "",
) -> str:
    lines = [f"    def {method_name}(self, idx: int) -> float:"]
    for i, name in enumerate(attr_names):
        lines.append(f"        if idx == {i}:")
        lines.append(f"            return self.{name}{suffix}[0]")
    lines.append("        return 0.0")
    return "\n".join(lines)


def write_scripted_fader_module(
    module_path: Path,
    attr_names: Sequence[str],
) -> None:
    """Write a TorchScript-friendly ScriptedFaderRAVE subclass with literal attr reads."""
    methods = "\n\n".join([
        _reader_method_source("_read_value", attr_names),
        _reader_method_source("_read_scale", attr_names, suffix="_scale"),
        _reader_method_source("_read_override", attr_names, suffix="_override"),
    ])
    module_path.write_text(
        "# Auto-generated for TorchScript export — do not edit.\n"
        "from __future__ import annotations\n\n"
        "import torch\n\n"
        "from rave.fader.export.nn_module import ScriptedFaderRAVEBase\n\n\n"
        f"class ScriptedFaderRAVE(ScriptedFaderRAVEBase):\n{methods}\n",
        encoding="utf-8",
    )


def load_scripted_fader_class(module_path: Path) -> type:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"scripted_fader_{module_path.stem}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generated module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ScriptedFaderRAVE


def create_scripted_fader_rave(
    core: FaderTraceModel,
    min_max_features: Dict[str, Tuple[float, float]],
    continuous_attributes: Sequence[str],
    n_channels: int = 1,
    target_channels: Optional[int] = None,
    *,
    generated_module_path: Optional[Path] = None,
) -> nn_tilde.Module:
    if generated_module_path is None:
        raise ValueError("generated_module_path is required for TorchScript export")
    write_scripted_fader_module(generated_module_path, core.attribute_names)
    cls = load_scripted_fader_class(generated_module_path)
    return cls(
        core,
        min_max_features,
        continuous_attributes,
        n_channels=n_channels,
        target_channels=target_channels,
    )


class ScriptedFaderRAVEBase(nn_tilde.Module):
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
        self._torch_extract_names = [
            n for n in self.continuous_attributes if n in TORCH_EXTRACTABLE
        ]

        kinds = core.attribute_kinds
        for name in core.attribute_names:
            kind = kinds.get(name, "continuous")
            default = _default_raw_value(name, kind, min_max_features)
            self.register_attribute(name, default)
            self.register_attribute(f"{name}_scale", 1.0)
            self.register_attribute(f"{name}_override", False)

        self.register_attribute("attr_mode", 2)

        cont_idx = {n: i for i, n in enumerate(self._torch_extract_names)}
        cont_rows = [cont_idx.get(name, -1) for name in core.attribute_names]
        self.register_buffer(
            "_cont_extract_row",
            torch.tensor(cont_rows, dtype=torch.long),
        )

        self.extractor = TorchDescriptorExtract(
            continuous_attributes=self._torch_extract_names,
            sr=core.sr,
        )
        calibrate_torch_descriptor_extract(self.extractor, sr=core.sr)

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

    def _build_manual_raw(
        self,
        batch: int,
        t_lat: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        d_total = int(self.core.num_attributes.item())
        raw = torch.zeros(batch, d_total, t_lat, device=device, dtype=dtype)
        for i in range(d_total):
            raw[:, i, :] = self._read_value(i)
        return raw

    def _apply_scales(self, attr_norm: torch.Tensor) -> torch.Tensor:
        out = attr_norm.clone()
        d_total = int(self.core.num_attributes.item())
        for i in range(d_total):
            scale = self._read_scale(i)
            out[:, i, :] = out[:, i, :] * scale
        return out

    def _extract_history_ready(self) -> bool:
        return int(self.extractor._hist_len.item()) >= int(self.extractor.max_history)

    def _clamp_raw(self, raw: torch.Tensor) -> torch.Tensor:
        lo = self.core.min_max_features[:, 0].view(1, -1, 1)
        hi = self.core.min_max_features[:, 1].view(1, -1, 1)
        cont = self.core.is_continuous.view(1, -1, 1) > 0.5
        clamped = torch.clamp(raw, lo, hi)
        return torch.where(cont, clamped, raw)

    def _merge_raw(
        self,
        x: torch.Tensor,
        t_lat: int,
        manual: torch.Tensor,
    ) -> torch.Tensor:
        mode = int(self.attr_mode[0])
        d_total = int(self.core.num_attributes.item())

        if mode == 1:
            return manual

        cont_raw = self.extractor(x, t_lat)
        extracted = torch.zeros_like(manual)
        history_ready = self._extract_history_ready()
        for i in range(d_total):
            row = int(self._cont_extract_row[i].item())
            if row >= 0 and history_ready:
                extracted[:, i, :] = cont_raw[:, row, :]
            else:
                extracted[:, i, :] = manual[:, i, :]

        if mode == 2:
            for i in range(d_total):
                if self.core.is_continuous[i] <= 0.5:
                    extracted[:, i, :] = manual[:, i, :]
            return extracted

        raw = extracted.clone()
        for i in range(d_total):
            if self._read_override(i) != 0:
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
        raw = self._clamp_raw(raw)
        attr_norm = self._apply_scales(self.core.normalize_all(raw))
        attr_norm = attr_norm.clamp(-1.0, 1.0)
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


# Backwards-compatible alias; prefer create_scripted_fader_rave() for export.
ScriptedFaderRAVE = ScriptedFaderRAVEBase
