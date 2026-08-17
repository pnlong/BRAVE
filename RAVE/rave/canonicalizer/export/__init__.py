"""Export helpers for embedding canonicalizers in realtime bundles."""

from .load import attach_canonicalizer_for_export
from .resolve import (
    CANON_LATENT_NAME,
    CANON_WAVEFORM_NAME,
    CYCLEGAN_LATENT_NAME,
    resolve_canonicalizer_ckpt,
    resolve_cyclegan_ckpt,
)

__all__ = [
    "CANON_LATENT_NAME",
    "CANON_WAVEFORM_NAME",
    "CYCLEGAN_LATENT_NAME",
    "ScriptedCycleGANXY",
    "attach_canonicalizer_for_export",
    "export_cyclegan_nn",
    "resolve_canonicalizer_ckpt",
    "resolve_cyclegan_ckpt",
]


def export_cyclegan_nn(*args, **kwargs):
    from .cyclegan_nn import export_cyclegan_nn as _export

    return _export(*args, **kwargs)


def __getattr__(name: str):
    if name == "ScriptedCycleGANXY":
        from .cyclegan_nn import ScriptedCycleGANXY

        return ScriptedCycleGANXY
    raise AttributeError(name)
