#!/usr/bin/env python3
"""Resume water ckpt and run 3 steps across phase-2 boundary with PL."""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.join(os.path.dirname(__file__), "..", "RAVE")
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
sys.path.insert(0, _RAVE_ROOT)

import gin
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import rave
import rave.dataset

CKPT = (
    "/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/runs/"
    "water_uncond_run_8e1e614287/epoch-epoch=7343.ckpt"
)
DB_PATH = "/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/preprocessed"
CONFIG = os.path.join(_BRAVE_ROOT, "configs/brave.gin")


def main() -> None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    gin.parse_config_file(CONFIG)
    n_channels = rave.dataset.get_training_channels(DB_PATH, 0)
    model = rave.RAVE(n_channels=n_channels)

    ds = rave.dataset.get_dataset(DB_PATH, 44100, 131072, n_channels=n_channels)
    train_ds, val_ds = rave.dataset.split_dataset(ds, 98)
    train_ds = rave.dataset.maybe_reject_silent(train_ds)
    train = DataLoader(train_ds, 8, True, drop_last=True, num_workers=0)
    val = DataLoader(val_ds, 8, False, num_workers=0)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_steps=1_000_002,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        profiler=None,
    )
    print("Fitting from ckpt — should cross step 1M (phase 2)...", flush=True)
    trainer.fit(model, train, val, ckpt_path=CKPT)
    print("SUCCESS: crossed phase 2")


if __name__ == "__main__":
    main()
