#!/usr/bin/env python3
"""Minimal repro: water run phase-2 discriminator backward (step ~1M)."""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.join(os.path.dirname(__file__), "..", "RAVE")
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
sys.path.insert(0, _RAVE_ROOT)

import gin
import torch
from torch.utils.data import DataLoader

import rave
import rave.dataset
import rave.training

CKPT = (
    "/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/runs/"
    "water_uncond_run_8e1e614287/epoch-epoch=7343.ckpt"
)
DB_PATH = "/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/preprocessed"
CONFIG = os.path.join(_BRAVE_ROOT, "configs/brave.gin")
BATCH = 8
N_SIGNAL = 131072
TARGET_STEP = 1_000_000


def main() -> None:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    device = torch.device("cuda:0")

    gin.parse_config_file(CONFIG)
    n_channels = rave.dataset.get_training_channels(DB_PATH, 0)
    model = rave.RAVE(n_channels=n_channels)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = model.load_from_checkpoint(CKPT, map_location="cpu")
    model = model.to(device)
    model.train()
    model.automatic_optimization = False
    model.warmed_up = True
    print(f"warmed_up={model.warmed_up} update_discriminator_every={model.update_discriminator_every}")

    dis_opt = torch.optim.Adam(model.discriminator.parameters(), 1e-4, (0.5, 0.9))
    gen_opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), 1e-3, (0.5, 0.9)
    )
    if ckpt.get("optimizer_states"):
        gen_opt.load_state_dict(ckpt["optimizer_states"][0])
        dis_opt.load_state_dict(ckpt["optimizer_states"][1])
        print("Loaded generator + discriminator optimizer state from checkpoint")
    model._optimizers = [gen_opt, dis_opt]
    model.optimizers = lambda: model._optimizers

    ds = rave.dataset.get_dataset(DB_PATH, 44100, N_SIGNAL, n_channels=n_channels)
    train_ds, _ = rave.dataset.split_dataset(ds, 98)
    train_ds = rave.dataset.maybe_reject_silent(train_ds)
    loader = DataLoader(train_ds, BATCH, shuffle=True, drop_last=True, num_workers=0)

    # batch_idx even => discriminator step when warmed_up
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 2 != 0:
            continue
        batch = batch.to(device)
        print(f"batch_idx={batch_idx} shape={tuple(batch.shape)} "
              f"min={batch.min().item():.4f} max={batch.max().item():.4f} "
              f"nan={torch.isnan(batch).any().item()}")

        x_raw = batch
        x_raw.requires_grad = True
        batch_size = x_raw.shape[:-2]
        model.encoder.set_warmed_up(True)
        model.decoder.set_warmed_up(True)

        z, x_multiband = model.encode(x_raw, return_mb=True)
        z, reg = model.encoder.reparametrize(z)[:2]
        y = model.decoder(z)
        y_multiband = y
        y_raw = rave.model._pqmf_decode(model.pqmf, y, batch_size=batch_size, n_channels=model.n_channels)
        y_raw = y_raw[..., : x_raw.shape[-1]]
        y_multiband = y_multiband[..., : x_multiband.shape[-1]]
        print(f"  y_raw shape={tuple(y_raw.shape)} nan={torch.isnan(y_raw).any().item()}")

        xy = torch.cat([x_raw.detach(), y_raw.detach()], 0)
        print(f"  xy shape={tuple(xy.shape)}")
        print("  discriminator forward...", flush=True)
        features = model.discriminator(xy)
        feature_real, feature_fake = model.split_features(features)

        loss_dis = 0
        for scale_real, scale_fake in zip(feature_real, feature_fake):
            _dis, _adv = model.gan_loss(scale_real[-1], scale_fake[-1])
            loss_dis = loss_dis + _dis
            print(f"    scale dis={float(_dis):.6f} nan={torch.isnan(_dis).any().item()}")

        print(f"  loss_dis={float(loss_dis):.6f} — backward...", flush=True)
        dis_opt.zero_grad()
        loss_dis.backward()
        dis_opt.step()
        print("  isolated disc step OK")

    # Full training_step path (matches train.py / Lightning manual loop)
    torch.cuda.synchronize()
    for batch_idx, batch in enumerate(loader):
        if batch_idx != 124:
            continue
        batch = batch.to(device)
        print(f"full training_step batch_idx={batch_idx} shape={tuple(batch.shape)}", flush=True)
        model.training_step(batch, batch_idx)
        torch.cuda.synchronize()
        print("  full training_step OK")
        break
    else:
        print("WARNING: batch_idx 124 not reached in one epoch")

    print("Water phase-2 repro completed without error.")


if __name__ == "__main__":
    main()
