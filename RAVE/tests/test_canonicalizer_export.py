"""Smoke test: FaderTraceModel with latent canonicalizer exports to nn~ .ts."""

import pytest

pytest.importorskip("nn_tilde")
pytest.importorskip("torch")

import torch
import torch.nn as nn

from rave.fader.export.nn_module import ScriptedFaderRAVE
from rave.fader.export.trace_model import FaderTraceModel
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer


def _dummy_trace_with_canonicalizer() -> FaderTraceModel:
    class _Enc(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, :1], x[:, :1]], dim=1)

    class _Dec(nn.Module):
        def forward(self, z):
            return z[:, :1, :]

    canon = LatentCanonicalizer(latent_size=1)
    return FaderTraceModel(
        encoder=_Enc(),
        decoder=_Dec(),
        pqmf=None,
        attribute_names=["rms"],
        attribute_kinds={"rms": "continuous"},
        min_max_features={"rms": (0.0, 1.0)},
        discrete_num_classes={},
        latent_size=1,
        sr=44100,
        latent_canonicalizer=canon,
    )


def test_fader_canonicalizer_nn_export_roundtrip(tmp_path):
    core = _dummy_trace_with_canonicalizer()
    mod = ScriptedFaderRAVE(
        core=core,
        min_max_features={"rms": (0.0, 1.0)},
        continuous_attributes=["rms"],
        n_channels=1,
    )
    x = torch.randn(1, 1, 4096)
    y = mod(x)
    assert y.shape[0] == 1

    out = tmp_path / "model.ts"
    mod.export_to_ts(str(out))
    loaded = torch.jit.load(str(out))
    y2 = loaded(x)
    assert y2.shape == y.shape

    z = loaded.encode(x)
    assert z.shape[1] == 1


def test_scripted_cyclegan_xy_forward():
    import rave.blocks
    from rave.canonicalizer.export.cyclegan_nn import ScriptedCycleGANXY
    from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer

    class _InnerEnc(nn.Module):
        def __init__(self, n_channels=1):
            super().__init__()
            self.n_channels = n_channels

        def forward(self, x):
            return torch.cat([x[:, :1], x[:, :1]], dim=1)

    class _Dec(nn.Module):
        def forward(self, z):
            return z[:, :1, :]

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = rave.blocks.VariationalEncoder(
                encoder=_InnerEnc, n_channels=1)
            self.decoder = _Dec()
            self.pqmf = None
            self.n_channels = 1
            self.latent_size = 1
            self.input_mode = "raw"
            self.output_mode = "raw"
            self.sr = 44100

    bb_x = _Backbone()
    bb_y = _Backbone()
    warp = LatentCanonicalizer(latent_size=1, init_mode="random")
    mod = ScriptedCycleGANXY(bb_x, bb_y, warp)
    x = torch.randn(1, 1, 4096)
    y = mod(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 1
    z = mod.encode(x)
    assert z.shape[1] == 1
    mod.set_sidechain(False)
    y_off = mod(x)
    mod.set_sidechain(True)
    y_on = mod(torch.zeros_like(x))
    # Zeros in + sidechain on should duck vs loud input with sidechain off
    assert y_on.abs().mean() <= y_off.abs().mean() + 1e-4
