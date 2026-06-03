#!/usr/bin/env python3
"""Visualize latents — thin wrapper around mask_reconstruct with no mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from latent_viz import DEFAULT_PCA_FIDELITY
from mask_reconstruct import run_reconstruction
from paths import PLOTS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Equivalent to mask_reconstruct.py with --mask-style none.",
    )
    p.add_argument("--model", required=True, help="checkpoint run dir or .ckpt path")
    p.add_argument("--input", required=True, help="input audio file")
    p.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: artifacts/plots/<stem>/)",
    )
    p.add_argument(
        "--pca-fidelity",
        type=float,
        default=DEFAULT_PCA_FIDELITY,
        help="PCA panel: show leading components until this cumulative variance (default: 0.95)",
    )
    p.add_argument(
        "--latent-mode",
        choices=("mean", "sample"),
        default="mean",
        help="use VAE mean (deterministic) or reparametrize sample",
    )
    p.add_argument(
        "--no-wavs",
        action="store_true",
        help="skip saving reconstructed/original WAV files",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else PLOTS_DIR / input_path.stem
    run_reconstruction(
        args.model,
        args.input,
        output_dir=output_dir,
        mask_style="none",
        latent_mode=args.latent_mode,
        use_gpu=args.gpu,
        pca_fidelity=args.pca_fidelity,
        save_wavs=not args.no_wavs,
        save_plot=True,
        clip_outliers=args.clip_outliers,
        clip_percentiles=tuple(args.clip_percentile),
    )


if __name__ == "__main__":
    main()
