"""Resolve canonicalizer checkpoint paths for export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rave

CANON_WAVEFORM_NAME = "waveform_canonicalizer.ckpt"
CANON_LATENT_NAME = "latent_canonicalizer.ckpt"


def resolve_canonicalizer_ckpt(
    model_path: str,
    *,
    mode: str = "auto",
    waveform_canonicalizer: Optional[str] = None,
    latent_canonicalizer: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve canonicalizer checkpoint path.

    ``mode`` is ``auto`` (search run dir), ``none``, or an explicit ckpt path.
    """
    if waveform_canonicalizer and latent_canonicalizer:
        raise ValueError(
            "Specify only one of waveform_canonicalizer or latent_canonicalizer"
        )
    if waveform_canonicalizer:
        return waveform_canonicalizer
    if latent_canonicalizer:
        return latent_canonicalizer
    if mode == "none":
        return None
    if mode not in ("auto", "none"):
        return mode

    run = rave.core.search_for_run(model_path)
    if run is None:
        return None
    run_dir = Path(rave.core.run_dir_from_checkpoint(run))
    wf = run_dir / CANON_WAVEFORM_NAME
    lt = run_dir / CANON_LATENT_NAME
    if wf.is_file():
        return str(wf)
    if lt.is_file():
        return str(lt)
    return None
