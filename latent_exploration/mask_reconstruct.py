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
    build_constant_attr,
    compression_ratio,
    extract_normalized_attributes,
    has_pca,
    is_fader_model,
    load_audio,
    load_model,
    load_rave,
    pca_display_dims,
    save_audio,
)
from latent_viz import (
    DEFAULT_PCA_FIDELITY,
    plot_latent_distribution_histograms,
    plot_mel_and_latent,
    plot_mel_latent_and_pca,
    print_latent_distributions,
)
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


MASK_SPACES = ("vae", "pca")
ATTR_MODES = ("extract", "zeros", "swap", "constant")


def _parse_attr_constant(spec: str) -> dict[str, float]:
    """Parse name=value,name2=value2 into dict."""
    out: dict[str, float] = {}
    if not spec.strip():
        return out
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _resolve_attributes(
    model,
    x: torch.Tensor,
    *,
    attr_mode: str,
    swap_x: torch.Tensor | None,
    attr_constant: dict[str, float],
    time_frames: int,
) -> torch.Tensor | None:
    """Return attr_norm [1,D,T] for Fader decode, or None for vanilla RAVE."""
    if not is_fader_model(model):
        return None

    if attr_mode == "zeros":
        d = model.num_attributes
        return torch.zeros(1, d, time_frames, device=x.device)

    if attr_mode == "constant":
        return build_constant_attr(
            model, attr_constant, time_frames=time_frames, device=x.device)

    if attr_mode == "swap":
        if swap_x is None:
            raise ValueError("--attr-mode swap requires --swap-input")
        return extract_normalized_attributes(model, swap_x)

    # --- Default: extract from input audio ---
    return extract_normalized_attributes(model, x)


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
    clip_outliers: bool = False,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    stats_path: str | Path | None = None,
    db_path: str | Path | None = None,
    attr_mode: str = "extract",
    swap_input: str | Path | None = None,
    attr_constant: str = "",
) -> Path:
    """Encode → optional mask → decode; save reconstruction WAVs and mel/latent PNG."""
    if mask_space not in MASK_SPACES:
        raise ValueError(f"unknown mask_space {mask_space!r}; choose from {MASK_SPACES}")

    if attr_mode not in ATTR_MODES:
        raise ValueError(f"unknown attr_mode {attr_mode!r}; choose from {ATTR_MODES}")

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input audio not found: {input_path}")

    model = load_model(
        model_path, use_gpu=use_gpu, stats_path=stats_path, db_path=db_path)
    device = next(model.parameters()).device
    x = load_audio(input_path, model, device=device)
    swap_x = None
    if swap_input:
        swap_x = load_audio(swap_input, model, device=device)

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
        attr_norm = _resolve_attributes(
            model,
            x,
            attr_mode=attr_mode,
            swap_x=swap_x,
            attr_constant=_parse_attr_constant(attr_constant),
            time_frames=time_frames,
        )
        if attr_norm is not None:
            y = model.decode(z_out, attr=attr_norm)
        else:
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

    vae_dist_label = "VAE latent (post-mask)" if mask_style != "none" else "VAE latent"
    pca_dist_label = "PCA latent (post-mask)" if mask_style != "none" else "PCA latent"
    print_latent_distributions(
        z_out,
        z_pca_masked if has_pca(model) else None,
        vae_label=vae_dist_label,
        pca_label=pca_dist_label,
    )

    if save_plot:
        hist_path = out_dir / f"{input_path.stem}_latent_hist.png"
        plot_latent_distribution_histograms(
            z_out,
            hist_path,
            z_pca_masked if has_pca(model) else None,
            vae_label=vae_dist_label,
            pca_label=pca_dist_label,
            clip_percentiles=clip_percentiles,
        )
        print(f"saved latent histogram: {hist_path}")

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
                clip_outliers=clip_outliers,
                clip_percentiles=clip_percentiles,
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
                clip_outliers=clip_outliers,
                clip_percentiles=clip_percentiles,
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
        "--clip-outliers",
        action="store_true",
        help="use percentile clip for heatmap color limits (see --clip-percentile)",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(2.0, 98.0),
        help="percentile range for --clip-outliers color limits (default: 2 98)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="use CUDA (set CUDA_VISIBLE_DEVICES to pick a GPU; default: CPU)",
    )
    p.add_argument(
        "--stats-path",
        default=None,
        help="attribute_stats.yaml for FaderRAVE (auto-search near checkpoint if omitted)",
    )
    p.add_argument(
        "--db-path",
        default=None,
        help="LMDB path to find attribute_stats.yaml for FaderRAVE",
    )
    p.add_argument(
        "--attr-mode",
        choices=ATTR_MODES,
        default="extract",
        help="Fader attribute control: extract from input, zeros, swap, or constant",
    )
    p.add_argument(
        "--swap-input",
        default=None,
        help="second audio file for --attr-mode swap (z from --input, attr from swap)",
    )
    p.add_argument(
        "--attr-constant",
        default="",
        help="comma name=value pairs for --attr-mode constant (raw or class index)",
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
        clip_outliers=args.clip_outliers,
        clip_percentiles=tuple(args.clip_percentile),
        stats_path=args.stats_path,
        db_path=args.db_path,
        attr_mode=args.attr_mode,
        swap_input=args.swap_input,
        attr_constant=args.attr_constant,
    )


if __name__ == "__main__":
    main()
