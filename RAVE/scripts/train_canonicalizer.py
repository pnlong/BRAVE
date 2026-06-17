#!/usr/bin/env python3
"""Train waveform_canonicalizer or latent_canonicalizer on frozen FaderRAVE."""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import argparse
from pathlib import Path

import gin
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import rave
import rave.dataset
import rave.training
from rave.fader.canonicalizer_config import (
    CanonicalizerManifest,
    _stats_hash,
    build_domain_profile,
    load_latent_stats,
    save_canonicalizer_checkpoint,
)
from rave.fader.canonicalizer_callbacks import CanonicalizerValVizCallback
from rave.fader.canonicalizer_dataset import build_canonicalizer_dataset
from rave.fader.canonicalizer_trainer import CanonicalizerTrainer
from rave.fader.latent_canonicalizer import LatentCanonicalizer
from rave.fader.latent_domain_discriminator import LatentDomainDiscriminator
from rave.fader.model import FaderRAVE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/brave_canonicalizer.gin")
    p.add_argument(
        "--fader_config",
        required=True,
        help="Fader backbone gin (e.g. configs/brave_fader_pitched.gin)",
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--db_path", required=True)
    p.add_argument("--ood_path", required=True, help="Directory of OOD WAV sidecar")
    p.add_argument(
        "--canonicalizer_type",
        choices=("waveform", "latent"),
        required=True,
    )
    p.add_argument("--name", required=True)
    p.add_argument("--out_path", default="runs/")
    p.add_argument("--n_signal", type=int, default=131072)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--val_scatter", choices=("pca", "tsne", "both"), default="both")
    p.add_argument("--val_every", type=int, default=500, help="Validation + viz every N steps")
    p.add_argument("--val_batches", type=int, default=8, help="Max validation batches per check")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb_project", default="brave-canonicalizer")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument(
        "--ir_path",
        default=None,
        help="Directory of room IR .wav files for OOD augmentation",
    )
    p.add_argument(
        "--ir_prob",
        type=float,
        default=None,
        help="Probability of IR aug on OOD clips (default: gin, usually 0.5 when ir_path set)",
    )
    p.add_argument("--no_ir", action="store_true", help="Disable OOD IR augmentation")
    return p.parse_args()


def _add_gin_ext(path: str) -> str:
    return path if path.endswith(".gin") else path + ".gin"


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    fader_cfg = _add_gin_ext(args.fader_config)
    canon_cfg = _add_gin_ext(args.config)
    if not Path(fader_cfg).is_absolute():
        fader_cfg = str(Path(_BRAVE_ROOT) / fader_cfg)
    if not Path(canon_cfg).is_absolute():
        canon_cfg = str(Path(_BRAVE_ROOT) / canon_cfg)

    cfg_dir = str(Path(fader_cfg).parent)
    prev_cwd = os.getcwd()
    os.chdir(cfg_dir)
    try:
        gin.parse_config_file(Path(fader_cfg).name)
        gin.parse_config_file(Path(canon_cfg).name)
        for o in args.override:
            gin.parse_config(o)
    finally:
        os.chdir(prev_cwd)

    n_channels = rave.dataset.get_training_channels(args.db_path, 0)
    gin.bind_parameter("RAVE.n_channels", n_channels)

    profile = build_domain_profile(fader_cfg, args.db_path)

    model = FaderRAVE(n_channels=n_channels)
    run = rave.core.search_for_run(args.ckpt)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    model = model.load_from_checkpoint(run)
    model.load_attribute_stats_from_file(profile.stats_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.canonicalizer_type == "waveform":
        from rave.dsp import BiquadBank, CausalReverb
        from rave.fader.waveform_canonicalizer import WaveformCanonicalizer

        warp = WaveformCanonicalizer(
            eq=BiquadBank(sample_rate=model.sr),
            reverb=CausalReverb(sample_rate=model.sr),
            use_reverb=True,
        )
        ckpt_name = "waveform_canonicalizer.ckpt"
    else:
        warp = LatentCanonicalizer(latent_size=model.latent_size)
        ckpt_name = "latent_canonicalizer.ckpt"

    latent_mean = None
    if profile.latent_stats_path and profile.latent_stats_path.is_file():
        st = load_latent_stats(profile.latent_stats_path)
        latent_mean = torch.tensor(st["latent_mean"], dtype=torch.float32)

    trainer_module = CanonicalizerTrainer(
        fader=model,
        warp=warp,
        canonicalizer_type=args.canonicalizer_type,
        domain_profile=profile,
        latent_mean=latent_mean,
    )

    ds = rave.dataset.get_dataset(
        args.db_path, model.sr, args.n_signal, n_channels=n_channels)
    train_base, val_base = rave.dataset.split_dataset(ds, 98)
    train_wrapped, val_wrapped = rave.training.wrap_training_datasets(
        train_base,
        val_base,
        sampling_rate=model.sr,
        n_signal=args.n_signal,
        db_path=args.db_path,
    )

    from rave.fader.dataset import FaderAttributeDataset
    if not isinstance(train_wrapped, FaderAttributeDataset):
        raise TypeError("Expected FaderAttributeDataset from wrap_fader_dataset")
    if not isinstance(val_wrapped, FaderAttributeDataset):
        raise TypeError("Expected FaderAttributeDataset for validation")

    ratios = list(model.encoder.ratios) if hasattr(model.encoder, "ratios") else [2, 2, 2, 1]
    cont = model.continuous_attributes
    disc = model.discrete_attributes

    ir_kwargs: dict = {}
    if args.no_ir:
        ir_kwargs["ir_prob"] = 0.0
    else:
        if args.ir_path is not None:
            ir_kwargs["ir_path"] = args.ir_path
            if args.ir_prob is None:
                ir_kwargs["ir_prob"] = 0.5
        if args.ir_prob is not None:
            ir_kwargs["ir_prob"] = args.ir_prob

    if ir_kwargs.get("ir_prob", 0.0) > 0.0:
        ir_src = ir_kwargs.get("ir_path") or "(synthetic fallback)"
        print(f"OOD IR augmentation: prob={ir_kwargs['ir_prob']} path={ir_src}")

    mixed = build_canonicalizer_dataset(
        train_dataset=train_wrapped,
        ood_path=args.ood_path,
        sampling_rate=model.sr,
        n_signal=args.n_signal,
        n_bands=16,
        ratios=ratios,
        continuous_attributes=cont,
        discrete_attributes=disc,
        **ir_kwargs,
    )
    val_mixed = build_canonicalizer_dataset(
        train_dataset=val_wrapped,
        ood_path=args.ood_path,
        sampling_rate=model.sr,
        n_signal=args.n_signal,
        n_bands=16,
        ratios=ratios,
        continuous_attributes=cont,
        discrete_attributes=disc,
        in_domain_fraction=0.5,
        ir_prob=0.0,
    )

    num_workers = 0 if sys.platform == "darwin" else args.workers
    loader = DataLoader(
        mixed,
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_mixed,
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    out_dir = Path(args.out_path) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        CanonicalizerValVizCallback(
            out_dir=out_dir,
            scatter_method="pca" if args.val_scatter == "both" else args.val_scatter,
            also_tsne=args.val_scatter == "both",
        ),
    ]

    logger = False
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import hashlib

        gin_hash = hashlib.md5(gin.operative_config_str().encode()).hexdigest()[:10]
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"{args.name}_{gin_hash}",
            save_dir=str(out_dir),
            offline=args.wandb_offline,
        )

    pl_trainer = pl.Trainer(
        max_steps=1 if args.smoke_test else args.max_steps,
        accelerator="gpu" if args.gpu is not None and torch.cuda.is_available() else "cpu",
        devices=[args.gpu] if args.gpu is not None else 1,
        default_root_dir=str(out_dir),
        enable_checkpointing=False,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=1 if args.smoke_test else args.val_every,
        limit_val_batches=1 if args.smoke_test else args.val_batches,
    )
    print("Running initial validation (step 0 baseline: PCA/t-SNE + audio)...")
    pl_trainer.validate(trainer_module, val_loader)
    pl_trainer.fit(trainer_module, loader, val_loader)

    manifest = CanonicalizerManifest(
        canonicalizer_type=args.canonicalizer_type,
        backbone_config=str(Path(fader_cfg).resolve()),
        backbone_ckpt=str(Path(args.ckpt).resolve()),
        db_path=str(Path(args.db_path).resolve()),
        use_reverb=getattr(warp, "use_reverb", False),
        stats_hash=_stats_hash(profile.stats_path),
    )
    save_canonicalizer_checkpoint(
        out_dir / ckpt_name,
        warp.state_dict(),
        manifest,
    )
    print(f"Saved {out_dir / ckpt_name}")


if __name__ == "__main__":
    main()
