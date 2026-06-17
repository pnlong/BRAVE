#!/usr/bin/env python3
"""Precompute backbone latent mean/std for canonicalizer L_latent_stat."""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import argparse
from pathlib import Path

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader

import rave
import rave.dataset
from rave.fader.canonicalizer_config import latent_stats_cache_path
from rave.fader.model import FaderRAVE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Fader gin config")
    p.add_argument("--ckpt", required=True, help="FaderRAVE checkpoint")
    p.add_argument("--db_path", required=True, help="LMDB path")
    p.add_argument("--n_signal", type=int, default=131072)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--output", default=None, help="Output .npz (default: db_path/latent_stats_canonicalizer.npz)")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    gin.parse_config_files_and_bindings([args.config], args.override)

    n_channels = rave.dataset.get_training_channels(args.db_path, 0)
    gin.bind_parameter("RAVE.n_channels", n_channels)

    model = FaderRAVE(n_channels=n_channels)
    run = rave.core.search_for_run(args.ckpt)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    model = model.load_from_checkpoint(run)
    from rave.fader.attributes import resolve_stats_path
    stats_path = resolve_stats_path(args.db_path)
    if stats_path:
        model.load_attribute_stats_from_file(stats_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    ds = rave.dataset.get_dataset(
        args.db_path, model.sr, args.n_signal, n_channels=n_channels)
    train, _ = rave.dataset.split_dataset(ds, 98)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)

    sums = None
    sq_sums = None
    count = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            z = model.encode_to_latent(x, use_mean=True)
            z_flat = z.reshape(z.shape[1], -1)
            if sums is None:
                sums = z_flat.sum(dim=1)
                sq_sums = (z_flat ** 2).sum(dim=1)
            else:
                sums = sums + z_flat.sum(dim=1)
                sq_sums = sq_sums + (z_flat ** 2).sum(dim=1)
            count += z_flat.shape[1]

    latent_mean = (sums / count).cpu().numpy()
    latent_var = (sq_sums / count - (sums / count) ** 2).clamp(min=0)
    latent_std = torch.sqrt(latent_var).cpu().numpy()

    out = Path(args.output) if args.output else latent_stats_cache_path(args.db_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, latent_mean=latent_mean, latent_std=latent_std, count=count)
    print(f"Wrote {out} (count={count})")


if __name__ == "__main__":
    main()
