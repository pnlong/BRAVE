"""Smoke test for Fader nn~ export module (requires nn_tilde)."""

import pytest

pytest.importorskip("nn_tilde")
pytest.importorskip("torch")

import torch

from rave.fader.export.nn_module import ScriptedFaderRAVE
from rave.fader.export.trace_model import FaderTraceModel


def _dummy_trace() -> FaderTraceModel:
    import torch.nn as nn

    class _Enc(nn.Module):
        def forward(self, x):
            return torch.cat([x[:, :1], x[:, :1]], dim=1)

    class _Dec(nn.Module):
        def forward(self, z):
            return z[:, :1, :]

    return FaderTraceModel(
        encoder=_Enc(),
        decoder=_Dec(),
        pqmf=None,
        attribute_names=["rms", "texture_class"],
        attribute_kinds={"rms": "continuous", "texture_class": "discrete"},
        min_max_features={"rms": (0.0, 1.0)},
        discrete_num_classes={"texture_class": 10},
        latent_size=1,
        sr=44100,
    )


def test_scripted_fader_export_to_ts(tmp_path):
    core = _dummy_trace()
    mod = ScriptedFaderRAVE(
        core=core,
        min_max_features={"rms": (0.0, 1.0)},
        continuous_attributes=["rms"],
        n_channels=1,
    )
    x = torch.randn(1, 1, 4096)
    y = mod(x)
    assert y.shape[0] == 1

    out = tmp_path / "fader.ts"
    mod.export_to_ts(str(out))
    loaded = torch.jit.load(str(out))
    y2 = loaded(x)
    assert y2.shape == y.shape
