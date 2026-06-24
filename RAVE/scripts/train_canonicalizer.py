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
    save_canonicalizer_checkpoint,
)
from rave.fader.canonicalizer_callbacks import CanonicalizerValVizCallback
from rave.fader.canonicalizer_dataset import (
    OodFaderDataset,
    build_canonicalizer_dataset,
    make_ir_augment,
)
from rave.fader.canonicalizer_trainer import CanonicalizerTrainer
from rave.fader.dataset import FaderAttributeDataset
from rave.fader.latent_canonicalizer import LatentCanonicalizer
from rave.fader.latent_domain_discriminator import LatentDomainDiscriminator
from rave.fader.model import FaderRAVE


def _parse_gin_configs(
    fader_cfg: str,
    canon_cfg: str,
    *,
    overrides: list[str] | None = None,
) -> None:
    cfg_dir = str(Path(fader_cfg).parent)
    prev_cwd = os.getcwd()
    os.chdir(cfg_dir)
    try:
        gin.parse_config_file(Path(fader_cfg).name)
        gin.parse_config_file(Path(canon_cfg).name)
        for o in overrides or []:
            gin.parse_config(o)
    finally:
        os.chdir(prev_cwd)


def _parse_fader_gin(fader_cfg: str, *, overrides: list[str] | None = None) -> None:
    """Re-apply fader gin so backbone architecture is not clobbered by brave.gin re-includes."""
    cfg_dir = str(Path(fader_cfg).parent)
    prev_cwd = os.getcwd()
    os.chdir(cfg_dir)
    try:
        gin.parse_config_file(Path(fader_cfg).name)
        for o in overrides or []:
            gin.parse_config(o)
    finally:
        os.chdir(prev_cwd)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/brave_canonicalizer.gin")
    p.add_argument(
        "--fader_config",
        required=True,
        help="Fader backbone gin (e.g. configs/brave_fader_pitched.gin)",
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--db_path", required=True, help="In-domain Fader LMDB")
    p.add_argument(
        "--ood_db_path",
        required=True,
        help="OOD Fader LMDB (same preprocess_fader pipeline as db_path)",
    )
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
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids librosa fork segfaults)")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--val_scatter", choices=("pca", "tsne", "both"), default="both")
    p.add_argument("--val_every", type=int, default=500, help="Validation + viz every N steps")
    p.add_argument("--val_batches", type=int, default=8, help="Max validation batches per check")
    p.add_argument(
        "--val_audio_samples",
        type=int,
        default=8,
        help="Validation clips per domain in W&B audio (input|pre_enc|recon each)",
    )
    p.add_argument(
        "--wandb_project",
        default="brave",
        help="Weights & Biases project name",
    )
    p.add_argument(
        "--wandb_entity",
        default=None,
        help="Weights & Biases entity (team or user)",
    )
    p.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Log to W&B in offline mode",
    )
    p.add_argument(
        "--log_every_n_steps",
        type=int,
        default=None,
        help="Lightning/W&B flush interval (default: min(50, batches per epoch))",
    )
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


def _load_fader_lmdb_pair(
    db_path: str,
    *,
    sr: int,
    n_signal: int,
    n_channels: int,
    split_percent: int = 98,
    reject_silent_train: bool = True,
) -> tuple[FaderAttributeDataset, FaderAttributeDataset]:
    ds = rave.dataset.get_dataset(db_path, sr, n_signal, n_channels=n_channels)
    train_base, val_base = rave.dataset.split_dataset(ds, split_percent)
    if reject_silent_train:
        train_base = rave.dataset.maybe_reject_silent(train_base)
    train_wrapped, val_wrapped = rave.training.wrap_training_datasets(
        train_base,
        val_base,
        sampling_rate=sr,
        n_signal=n_signal,
        db_path=db_path,
    )
    if not isinstance(train_wrapped, FaderAttributeDataset):
        raise TypeError(f"Expected FaderAttributeDataset for {db_path}")
    if not isinstance(val_wrapped, FaderAttributeDataset):
        raise TypeError(f"Expected FaderAttributeDataset for val split of {db_path}")
    return train_wrapped, val_wrapped


def _resolve_ir_augment(args, sr: int) -> object | None:
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

    ir_aug = make_ir_augment(sampling_rate=sr, **ir_kwargs)
    if ir_aug is not None:
        ir_src = ir_kwargs.get("ir_path") or "(synthetic fallback)"
        print(f"OOD IR augmentation: prob={ir_kwargs['ir_prob']} path={ir_src}")
    return ir_aug


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    fader_cfg = _add_gin_ext(args.fader_config)
    canon_cfg = _add_gin_ext(args.config)
    if not Path(fader_cfg).is_absolute():
        fader_cfg = str(Path(_BRAVE_ROOT) / fader_cfg)
    if not Path(canon_cfg).is_absolute():
        canon_cfg = str(Path(_BRAVE_ROOT) / canon_cfg)

    _parse_gin_configs(fader_cfg, canon_cfg, overrides=args.override)

    n_channels = rave.dataset.get_training_channels(args.db_path, 0)
    gin.bind_parameter("RAVE.n_channels", n_channels)

    profile = build_domain_profile(fader_cfg, args.db_path)

    _parse_fader_gin(fader_cfg, overrides=args.override)

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

    _parse_gin_configs(fader_cfg, canon_cfg, overrides=args.override)

    trainer_module = CanonicalizerTrainer(
        fader=model,
        warp=warp,
        canonicalizer_type=args.canonicalizer_type,
        domain_profile=profile,
    )

    train_wrapped, val_wrapped = _load_fader_lmdb_pair(
        args.db_path,
        sr=model.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
    )
    ood_train_wrapped, ood_val_wrapped = _load_fader_lmdb_pair(
        args.ood_db_path,
        sr=model.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
    )

    ir_aug = _resolve_ir_augment(args, model.sr)
    ood_train = OodFaderDataset(ood_train_wrapped, ir_augment=ir_aug)
    ood_val = OodFaderDataset(ood_val_wrapped, ir_augment=None)

    mixed = build_canonicalizer_dataset(
        in_domain_dataset=train_wrapped,
        ood_dataset=ood_train,
    )
    val_mixed = build_canonicalizer_dataset(
        in_domain_dataset=val_wrapped,
        ood_dataset=ood_val,
        in_domain_fraction=0.5,
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
            num_audio_samples=args.val_audio_samples,
        ),
    ]

    import hashlib
    from pytorch_lightning.loggers import WandbLogger

    gin_hash = hashlib.md5(gin.operative_config_str().encode()).hexdigest()[:10]
    wandb_kwargs = dict(
        project=args.wandb_project,
        name=f"{args.name}_{gin_hash}",
        save_dir=str(out_dir),
        offline=args.wandb_offline,
        config={
            "db_path": args.db_path,
            "ood_db_path": args.ood_db_path,
            "batch": args.batch,
            "n_signal": args.n_signal,
            "max_steps": args.max_steps,
            "canonicalizer_type": args.canonicalizer_type,
        },
    )
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    logger = WandbLogger(**wandb_kwargs)

    train_batches = len(loader)
    if train_batches == 0:
        raise SystemExit(
            "No training batches: mixed dataset length "
            f"({len(mixed)}; in-domain={len(train_wrapped)}, ood={len(ood_train)}) "
            f"is smaller than --batch={args.batch} with drop_last=True. "
            "Lower --batch or add more preprocessed LMDB chunks."
        )

    if args.log_every_n_steps is not None:
        log_every_n_steps = max(1, args.log_every_n_steps)
    else:
        log_every_n_steps = min(50, max(1, train_batches))
    if train_batches < 50:
        print(
            f"W&B: log_every_n_steps={log_every_n_steps} "
            f"({train_batches} train batches/epoch; default 50 would skip charts)",
        )

    val_check_kwargs: dict = {}
    if args.smoke_test:
        val_check_kwargs["val_check_interval"] = 1
    elif train_batches >= args.val_every:
        val_check_kwargs["val_check_interval"] = args.val_every
    else:
        nepoch = max(1, args.val_every // train_batches)
        val_check_kwargs["check_val_every_n_epoch"] = nepoch
        print(
            f"val_every={args.val_every} > train batches ({train_batches}); "
            f"validating every {nepoch} epoch(s) instead",
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
        limit_val_batches=1 if args.smoke_test else args.val_batches,
        log_every_n_steps=log_every_n_steps,
        **val_check_kwargs,
    )
    print("Running initial validation (step 0 baseline: PCA/t-SNE + audio)...")
    pl_trainer.validate(trainer_module, val_loader)
    pl_trainer.fit(trainer_module, loader, val_loader)

    manifest = CanonicalizerManifest(
        canonicalizer_type=args.canonicalizer_type,
        backbone_config=str(Path(fader_cfg).resolve()),
        backbone_ckpt=str(Path(args.ckpt).resolve()),
        db_path=str(Path(args.db_path).resolve()),
        ood_db_path=str(Path(args.ood_db_path).resolve()),
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
