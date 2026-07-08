"""Attach canonicalizer weights to a backbone before TorchScript / nn~ export."""

from __future__ import annotations

from typing import Union
from pathlib import Path

from ..config import CanonicalizerManifest, load_canonicalizer_onto_model


def attach_canonicalizer_for_export(
    model,
    ckpt_path: Union[str, Path],
) -> CanonicalizerManifest:
    """Load warp weights onto a RAVE or FaderRAVE instance."""
    return load_canonicalizer_onto_model(model, ckpt_path)
