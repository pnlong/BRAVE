"""Latent mask builders for exploration."""

from __future__ import annotations

import math

import torch

MASK_STYLES = ("none", "temporal", "latent")


def default_mask_width(axis_len: int) -> int:
    return max(1, math.ceil(axis_len * 0.1))


def build_mask(
    style: str,
    latent_dim: int,
    time_frames: int,
    *,
    start: int = 0,
    width: int | None = None,
) -> torch.Tensor:
    """
    Build a latent mask of shape [1, L, T].

    - none: all ones (identity)
    - temporal: zero columns [start : start+width)
    - latent: zero rows [start : start+width)
    """
    if style not in MASK_STYLES:
        raise ValueError(f"unknown mask style {style!r}; choose from {MASK_STYLES}")

    mask = torch.ones(1, latent_dim, time_frames)

    if style == "none":
        pass
    elif style == "temporal":
        axis_len = time_frames
        w = width if width is not None else default_mask_width(axis_len)
        end = min(start + w, time_frames)
        if start < 0 or start >= time_frames or start >= end:
            raise ValueError(
                f"invalid temporal mask: start={start}, width={w}, time_frames={time_frames}"
            )
        mask[:, :, start:end] = 0.0
    elif style == "latent":
        axis_len = latent_dim
        w = width if width is not None else default_mask_width(axis_len)
        end = min(start + w, latent_dim)
        if start < 0 or start >= latent_dim or start >= end:
            raise ValueError(
                f"invalid latent mask: start={start}, width={w}, latent_dim={latent_dim}"
            )
        mask[:, start:end, :] = 0.0
    else:
        raise ValueError(f"unknown mask style {style!r}; choose from {MASK_STYLES}")

    return mask
