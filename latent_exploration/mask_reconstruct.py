#!/usr/bin/env python3
"""Encode audio, optionally mask latents, reconstruct, save WAVs and latent plot PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from load_model import (
    apply_latent_mask,
    compression_ratio,
    has_pca,
    load_audio,
    load_rave,
    pca_display_dims,
    save_audio,
)
from latent_viz import DEFAULT_PCA_FIDELITY, plot_mel_and_latent, plot_mel_latent_and_pca
from masks import MASK_STYLES, build_mask, default_mask_width
from paths import RECONSTRUCTIONS_DIR

MASK_SPACES = ("vae", "pca")


def default_output_dir(
    input_path: Path,
    *,
    mask_style: str,
    mask_space: str,
    mask_start: int,
    mask_width: int | None,
    latent_dim: int,
    time_frames: int,
) -> Path:
    if mask_style == "none":
        return RECONSTRUCTIONS_DIR / input_path.stem
    w = mask_width
    if w is None:
        axis = time_frames if mask_style == "temporal" else latent_dim
        w = default_mask_width(axis)
    space_prefix = f"{mask_space}_" if mask_space != "vae" else ""
    suffix = f"{space_prefix}{mask_style}_s{mask_start}_w{w}"
    return RECONSTRUCTIONS_DIR / f"{input_path.stem}_{suffix}"


def _mask_overlay_panels(mask_style: str, mask_space: str) -> tuple[bool, bool]:
    if mask_style in ("none",):
        return False, False
    if mask_style == "temporal":
        return True, True
    # latent-style strip: VAE rows vs PCA components
    if mask_space == "pca":
        return False, True
    return True, False


def run_reconstruction(
    model_path: str | Path,
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    mask_style: str = "none",
    mask_space: str = "vae",
    mask_start: int = 0,
    mask_width: int | None = None,
    latent_mode: str = "mean",
    use_gpu: bool = False,
    pca_fidelity: float = DEFAULT_PCA_FIDELITY,
    save_wavs: bool = True,
    save_plot: bool = True,
) -> Path:
    """Encode → optional mask → decode; save reconstruction WAVs and mel/latent PNG."""
    if mask_space not in MASK_SPACES:
        raise ValueError(f"unknown mask_space {mask_space!r}; choose from {MASK_SPACES}")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input audio not found: {input_path}")

    model = load_rave(model_path, use_gpu=use_gpu)
    device = next(model.parameters()).device
    x = load_audio(input_path, model, device=device)

    with torch.no_grad():
        z = model.encode_to_latent(x[None], use_mean=(latent_mode == "mean"))
        latent_dim, time_frames = z.shape[-2], z.shape[-1]
        mask = build_mask(
            mask_style,
            latent_dim,
            time_frames,
            start=mask_start,
            width=mask_width,
        ).to(device)
        z_out, z_pca_masked = apply_latent_mask(
            model, z, mask, mask_space=mask_space
        )
        y = model.decode(z_out)

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = default_output_dir(
            input_path,
            mask_style=mask_style,
            mask_space=mask_space,
            mask_start=mask_start,
            mask_width=mask_width,
            latent_dim=latent_dim,
            time_frames=time_frames,
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_wavs:
        recon_path = out_dir / f"{input_path.stem}_reconstructed.wav"
        orig_path = out_dir / f"{input_path.stem}_original.wav"
        save_audio(recon_path, y.squeeze(0), model.sr)
        save_audio(orig_path, x, model.sr)
        print(f"saved reconstruction: {recon_path}")
        print(f"saved original: {orig_path}")

    print(f"latent shape: {tuple(z.shape)}")
    if mask_style != "none":
        print(f"mask style: {mask_style} (space: {mask_space})")

    if save_plot:
        ratio = compression_ratio(model)
        plot_suffix = "latents" if mask_style == "none" else "mask_plot"
        plot_path = out_dir / f"{input_path.stem}_{plot_suffix}.png"
        w = mask_width
        if w is None and mask_style != "none":
            axis = time_frames if mask_style == "temporal" else latent_dim
            w = default_mask_width(axis)

        mask_kw = {}
        if mask_style != "none":
            mask_kw = {
                "mask_style": mask_style,
                "mask_start": mask_start,
                "mask_width": w,
            }
        mask_on_vae, mask_on_pca = _mask_overlay_panels(mask_style, mask_space)

        if mask_style != "none":
            vae_title = "Latent (VAE, post-mask)"
            pca_title = "Latent (PCA, post-mask)"
        else:
            vae_title = "Latent (VAE)"
            pca_title = "Latent (PCA)"

        if has_pca(model) and z_pca_masked is not None:
            n_pca = pca_display_dims(model, pca_fidelity)
            print(
                f"PCA plot: top {n_pca} of 128 components "
                f"(≥{pca_fidelity:.0%} validation variance)"
            )
            plot_mel_latent_and_pca(
                x,
                z_out,
                z_pca_masked,
                model.sr,
                ratio,
                plot_path,
                vae_title=vae_title,
                pca_title=pca_title,
                pca_dims=n_pca,
                pca_fidelity=pca_fidelity,
                mask_on_vae=mask_on_vae,
                mask_on_pca=mask_on_pca,
                **mask_kw,
            )
        else:
            print("warning: no fitted PCA in checkpoint; saving two-panel plot only")
            plot_mel_and_latent(
                x,
                z_out,
                model.sr,
                ratio,
                plot_path,
                latent_title=vae_title,
                **mask_kw,
            )
        print(f"saved plot: {plot_path}")

    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint run dir or .ckpt path")
    p.add_argument("--input", required=True, help="input audio file")
    p.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: artifacts/reconstructions/<stem>[_<mask>])",
    )
    p.add_argument(
        "--mask-style",
        choices=MASK_STYLES,
        default="none",
        help="latent mask style (default: none = identity)",
    )
    p.add_argument(
        "--mask-space",
        choices=MASK_SPACES,
        default="vae",
        help="vae: mask encoder latents; pca: mask PCA components then inverse-rotate (default: vae)",
    )
    p.add_argument(
        "--mask-start",
        type=int,
        default=0,
        help="start index: time frames (temporal) or row index (latent / PCA component)",
    )
    p.add_argument(
        "--mask-width",
        type=int,
        default=None,
        help="mask strip width (default: ~10%% of axis)",
    )
    p.add_argument(
        "--latent-mode",
        choices=("mean", "sample"),
        default="mean",
        help="use VAE mean (deterministic) or reparametrize sample",
    )
    p.add_argument(
        "--pca-fidelity",
        type=float,
        default=DEFAULT_PCA_FIDELITY,
        help="PCA panel: show leading components until this cumulative variance (default: 0.95)",
    )
    p.add_argument(
        "--no-wavs",
        action="store_true",
        help="skip saving reconstructed/original WAV files",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip saving the mel/latent PNG",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="use CUDA (set CUDA_VISIBLE_DEVICES to pick a GPU; default: CPU)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_reconstruction(
        args.model,
        args.input,
        output_dir=args.output_dir,
        mask_style=args.mask_style,
        mask_space=args.mask_space,
        mask_start=args.mask_start,
        mask_width=args.mask_width,
        latent_mode=args.latent_mode,
        use_gpu=args.gpu,
        pca_fidelity=args.pca_fidelity,
        save_wavs=not args.no_wavs,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
