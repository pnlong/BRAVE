#!/usr/bin/env python3
"""Train waveform or latent canonicalizer on frozen RAVE / FaderRAVE."""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import argparse
import hashlib
import json
from pathlib import Path

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import rave
import rave.dataset
import rave.training
from rave.canonicalizer.callbacks import CanonicalizerValVizCallback
from rave.canonicalizer.config import (
    CanonicalizerManifest,
    build_training_profile,
    save_canonicalizer_checkpoint,
)
from rave.canonicalizer.dataset import (
    OodAudioDataset,
    OodFaderDataset,
    TaggedAudioDataset,
    TaggedFaderDataset,
    build_canonicalizer_dataloader,
    build_canonicalizer_dataset,
    canonicalizer_collate,
    ddp_aligned_num_batches,
    ddp_batches_per_rank,
)
from rave.canonicalizer.attribute_marginals import build_ood_discrete_marginals
from rave.fader.attributes import load_attribute_stats
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer
from rave.canonicalizer.gin_setup import (
    build_in_domain_discriminator,
    configure_backbone_gin,
    configure_canonicalizer_gin,
    resolve_canon_max_steps,
)
from rave.canonicalizer.trainer import CanonicalizerTrainer
from rave.canonicalizer.callbacks import (
    CanonicalizerGanRampCallback,
    CanonicalizerValVizCallback,
)
from rave.fader.dataset import FaderAttributeDataset
from rave.fader.model import FaderRAVE

# Ensure brave_canonicalizer.gin configurables are registered before parse.
import rave.canonicalizer.callbacks  # noqa: F401
import rave.canonicalizer.in_domain_discriminator  # noqa: F401
import rave.canonicalizer.trainer  # noqa: F401
import rave.canonicalizer.waveform_canonicalizer  # noqa: F401
import rave.canonicalizer.latent_canonicalizer  # noqa: F401
from rave import discriminator, dsp  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/brave_canonicalizer.gin")
    p.add_argument(
        "--backbone_config",
        required=True,
        help="Frozen backbone gin (configs/brave.gin or configs/brave_fader_*.gin)",
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--db_path", required=True, help="In-domain LMDB")
    p.add_argument("--ood_db_path", required=True, help="OOD LMDB")
    p.add_argument(
        "--canonicalizer_type",
        choices=("waveform", "latent"),
        required=True,
    )
    p.add_argument("--name", required=True)
    p.add_argument("--out_path", default="runs/")
    p.add_argument("--n_signal", type=int, default=131072)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Total optimizer steps (default: CANON_MAX_STEPS from gin, else 100000)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", type=int, action="append", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--val_scatter", choices=("pca", "tsne", "both"), default="both")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--val_audio_samples", type=int, default=8)
    p.add_argument("--wandb_project", default="brave")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--log_every_n_steps", type=int, default=None)
    p.add_argument("--calibration_batches", type=int, default=None)
    p.add_argument("--no_calibrate_scales", action="store_true")
    return p.parse_args()


def _add_gin_ext(path: str) -> str:
    return path if path.endswith(".gin") else path + ".gin"


def _load_audio_lmdb_pair(
    db_path: str,
    *,
    sr: int,
    n_signal: int,
    n_channels: int,
    is_fader: bool,
    split_percent: int = 98,
    reject_silent_train: bool = True,
) -> tuple:
    ds = rave.dataset.get_dataset(db_path, sr, n_signal, n_channels=n_channels)
    train_base, val_base = rave.dataset.split_dataset(ds, split_percent)
    if reject_silent_train:
        train_base = rave.dataset.maybe_reject_silent(train_base)

    if is_fader:
        train_wrapped, val_wrapped = rave.training.wrap_training_datasets(
            train_base,
            val_base,
            sampling_rate=sr,
            n_signal=n_signal,
            db_path=db_path,
        )
        if not isinstance(train_wrapped, FaderAttributeDataset):
            raise TypeError(f"Expected FaderAttributeDataset for {db_path}")
        return (
            TaggedFaderDataset(train_wrapped),
            TaggedFaderDataset(val_wrapped),
        )

    return (
        TaggedAudioDataset(train_base),
        TaggedAudioDataset(val_base),
    )


def _load_ood_pair(
    db_path: str,
    *,
    sr: int,
    n_signal: int,
    n_channels: int,
    is_fader: bool,
    split_percent: int = 98,
):
    if is_fader:
        train_wrapped, val_wrapped = _load_audio_lmdb_pair(
            db_path,
            sr=sr,
            n_signal=n_signal,
            n_channels=n_channels,
            is_fader=True,
        )
        return (
            OodFaderDataset(train_wrapped._fader),
            OodFaderDataset(val_wrapped._fader),
        )

    ds = rave.dataset.get_dataset(db_path, sr, n_signal, n_channels=n_channels)
    train_base, val_base = rave.dataset.split_dataset(ds, split_percent)
    train_base = rave.dataset.maybe_reject_silent(train_base)
    return (
        OodAudioDataset(train_base),
        OodAudioDataset(val_base),
    )


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    backbone_cfg = _add_gin_ext(args.backbone_config)
    canon_cfg = _add_gin_ext(args.config)
    if not Path(backbone_cfg).is_absolute():
        backbone_cfg = str(Path(_BRAVE_ROOT) / backbone_cfg)
    if not Path(canon_cfg).is_absolute():
        canon_cfg = str(Path(_BRAVE_ROOT) / canon_cfg)

    n_channels = rave.dataset.get_training_channels(args.db_path, 0)

    profile = build_training_profile(
        backbone_cfg,
        args.db_path,
        ood_db_path=args.ood_db_path,
    )

    configure_backbone_gin(backbone_cfg, n_channels)

    if profile.is_fader:
        model = FaderRAVE(n_channels=n_channels)
        backbone_kind = "FaderRAVE"
    else:
        model = rave.RAVE(n_channels=n_channels)
        backbone_kind = "RAVE"

    run = rave.core.search_for_run(args.ckpt)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    model = model.load_from_checkpoint(run)
    if profile.is_fader and profile.stats_path is not None:
        model.load_attribute_stats_from_file(profile.stats_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    configure_canonicalizer_gin(canon_cfg, n_channels, overrides=args.override)

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else resolve_canon_max_steps()
    )

    warp_kwargs: dict = {}
    if profile.is_fader and getattr(model, "num_attributes", 0) > 0:
        warp_kwargs = {
            "num_attributes": model.num_attributes,
            "num_classes_per_attribute": model.num_classes_per_attribute,
        }

    if args.canonicalizer_type == "waveform":
        from rave.canonicalizer.waveform_canonicalizer import build_waveform_canonicalizer

        warp = build_waveform_canonicalizer(
            sample_rate=model.sr,
            n_channels=n_channels,
            **warp_kwargs,
        )
        ckpt_name = "waveform_canonicalizer.ckpt"
    else:
        warp = LatentCanonicalizer(latent_size=model.latent_size, **warp_kwargs)
        ckpt_name = "latent_canonicalizer.ckpt"

    gin_snapshot = gin.config_str()
    gin_hash = hashlib.md5(gin_snapshot.encode()).hexdigest()[:10]
    run_name = f"{args.name}_{gin_hash}"
    out_dir = Path(args.out_path) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out_dir}")
    with open(out_dir / "config.gin", "w") as config_out:
        config_out.write(gin_snapshot)

    disc_kwargs: dict = {}
    if profile.is_fader and getattr(model, "num_attributes", 0) > 0:
        disc_kwargs = {
            "num_attributes": model.num_attributes,
            "num_classes_per_attribute": model.num_classes_per_attribute,
        }
    in_domain_disc = build_in_domain_discriminator(n_channels, **disc_kwargs)

    discrete_class_labels: dict = {}
    stratify_attr = ""
    stratify_n_classes = 0
    if profile.is_fader and getattr(model, "num_attributes", 0) > 0:
        if profile.stats_path is not None:
            stats = load_attribute_stats(profile.stats_path)
            discrete_class_labels = dict(
                stats.get("discrete_class_labels") or {})
        stratify_attr = (
            profile.discrete_attributes[0]
            if profile.discrete_attributes
            else ""
        )
        if stratify_attr:
            stratify_n_classes = int(
                getattr(model, "discrete_num_classes", {}).get(stratify_attr, 0))

    trainer_module = CanonicalizerTrainer(
        backbone=model,
        warp=warp,
        canonicalizer_type=args.canonicalizer_type,
        in_domain_disc=in_domain_disc,
        discrete_class_labels=discrete_class_labels,
    )
    print(
        f"Training for {max_steps:,} steps "
        f"(recon-only={trainer_module.warmup:,}, "
        f"gan_ramp={trainer_module.gan_ramp_duration:,}, "
        f"full_gan≈{max(0, max_steps - trainer_module.warmup - trainer_module.gan_ramp_duration):,})",
    )
    n_disc = sum(p.numel() for p in in_domain_disc.parameters())
    print(f"InDomainAudioDiscriminator: {n_disc:,} trainable params")

    train_in, val_in = _load_audio_lmdb_pair(
        args.db_path,
        sr=model.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
        is_fader=profile.is_fader,
    )
    ood_train, ood_val = _load_ood_pair(
        args.ood_db_path,
        sr=model.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
        is_fader=profile.is_fader,
    )

    num_workers = 0 if sys.platform == "darwin" else args.workers

    if profile.is_fader and getattr(model, "num_attributes", 0) > 0:
        marginal_np = build_ood_discrete_marginals(
            getattr(model, "attribute_names", []),
            getattr(model, "attribute_kinds", {}),
            getattr(model, "discrete_num_classes", {}),
            args.db_path,
            num_train_indices=len(train_in),
        )
        marginals = {
            name: torch.from_numpy(probs).float()
            for name, probs in marginal_np.items()
        }
        trainer_module.set_ood_discrete_marginals(marginals)
        if marginal_np:
            print("OOD discrete marginals (Y train split):")
            for name, probs in marginal_np.items():
                print(f"  {name}: {np.round(probs, 4).tolist()}")

    loader_kwargs = {}
    if profile.is_fader and stratify_attr and stratify_n_classes > 1:
        loader_kwargs = {
            "class_stratified_batches": True,
            "db_path": args.db_path,
            "stratify_discrete_attr": stratify_attr,
            "stratify_n_classes": stratify_n_classes,
        }

    loader = build_canonicalizer_dataloader(
        in_domain_dataset=train_in,
        ood_dataset=ood_train,
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        **loader_kwargs,
    )
    val_loader = build_canonicalizer_dataloader(
        in_domain_dataset=val_in,
        ood_dataset=ood_val,
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=False,
        drop_last=False,
        stratified_batches=False,
        num_workers=num_workers,
    )

    if args.no_calibrate_scales:
        trainer_module.calibrate_loss_scales = False

    if trainer_module.calibrate_loss_scales and not args.smoke_test:
        cal_device = (
            torch.device(f"cuda:{args.gpu[0]}")
            if args.gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )
        trainer_module.to(cal_device)
        n_cal = args.calibration_batches or trainer_module.calibration_batches
        scales = trainer_module.calibrate_loss_scales_from_loader(
            loader, max_batches=n_cal)
        print(
            f"Calibrated loss scales from {n_cal} stratified train batches "
            "(identity warp, frozen backbone):",
        )
        for key, value in scales.items():
            print(f"  {key}: {value:.6f}")
        with open(out_dir / "loss_scales.json", "w", encoding="utf-8") as scale_out:
            json.dump(
                {
                    **scales,
                    "calibration_batches": n_cal,
                    "calibrated": True,
                },
                scale_out,
                indent=2,
            )

    from pytorch_lightning.loggers import WandbLogger

    callbacks = [
        CanonicalizerGanRampCallback(),
        CanonicalizerValVizCallback(
            out_dir=out_dir,
            scatter_method="pca" if args.val_scatter == "both" else args.val_scatter,
            also_tsne=args.val_scatter == "both",
            num_audio_samples=args.val_audio_samples,
        ),
    ]

    wandb_kwargs = dict(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(out_dir),
        offline=args.wandb_offline,
        config={
            "db_path": args.db_path,
            "ood_db_path": args.ood_db_path,
            "backbone_kind": backbone_kind,
            "batch": args.batch,
            "n_signal": args.n_signal,
            "max_steps": max_steps,
            "canonicalizer_type": args.canonicalizer_type,
            "calibrate_loss_scales": trainer_module.calibrate_loss_scales,
        },
    )
    if trainer_module.loss_scales_calibrated:
        wandb_kwargs["config"].update({
            "stft_loss_scale": trainer_module.stft_loss_scale,
            "rms_loss_scale": trainer_module.rms_loss_scale,
            "gan_loss_scale": trainer_module.gan_loss_scale,
            "fm_loss_scale": trainer_module.fm_loss_scale,
            "latent_spread_scale": trainer_module.latent_spread_scale,
            "latent_in_std_mean": trainer_module.latent_in_std_mean,
        })
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    logger = WandbLogger(**wandb_kwargs)

    train_batches = len(loader)
    if train_batches == 0:
        raise SystemExit(
            "No training batches: in-domain="
            f"{len(train_in)}, ood={len(ood_train)}, "
            f"--batch={args.batch} with drop_last=True."
        )

    n_devices = (
        len(args.gpu)
        if args.gpu and torch.cuda.is_available() and len(args.gpu) > 1
        else 1
    )
    aligned_train_batches = ddp_aligned_num_batches(train_batches, n_devices)
    if aligned_train_batches == 0:
        raise SystemExit(
            "No training batches after DDP alignment: in-domain="
            f"{len(train_in)}, ood={len(ood_train)}, "
            f"--batch={args.batch}, devices={n_devices}."
        )
    batches_per_rank = ddp_batches_per_rank(train_batches, n_devices)
    if n_devices > 1 and aligned_train_batches < train_batches:
        dropped = train_batches - aligned_train_batches
        print(
            f"DDP batch alignment: using {aligned_train_batches}/{train_batches} "
            f"stratified batches ({dropped} dropped) -> {batches_per_rank}/rank",
        )

    if args.log_every_n_steps is not None:
        log_every_n_steps = max(1, args.log_every_n_steps)
    else:
        log_every_n_steps = min(50, max(1, batches_per_rank))

    val_check_kwargs: dict = {}
    if args.smoke_test:
        val_check_kwargs["val_check_interval"] = 1
    elif batches_per_rank >= args.val_every:
        val_check_kwargs["val_check_interval"] = args.val_every
    else:
        nepoch = max(1, args.val_every // batches_per_rank)
        val_check_kwargs["check_val_every_n_epoch"] = nepoch
        print(
            f"val scheduling: {batches_per_rank} batches/rank "
            f"(of {aligned_train_batches} aligned / {train_batches} total), "
            f"check_val_every_n_epoch={nepoch}",
        )

    accelerator = "cpu"
    devices: int | list[int] = 1
    strategy = None
    if args.gpu and torch.cuda.is_available():
        accelerator = "gpu"
        # Always pass explicit GPU ids as a list — devices=0 means "zero GPUs" to PL.
        devices = args.gpu
        if len(args.gpu) > 1:
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=True)
            print(
                f"Multi-GPU DDP (find_unused_parameters=True): {len(args.gpu)} devices, "
                f"per-GPU batch={args.batch}, global batch≈{args.batch * len(args.gpu)}, "
                f"batches/rank={batches_per_rank}",
            )
    elif args.gpu:
        print("CUDA not available; training on CPU")

    pl_trainer = pl.Trainer(
        max_steps=1 if args.smoke_test else max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        default_root_dir=str(out_dir),
        enable_checkpointing=False,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        limit_val_batches=1 if args.smoke_test else args.val_batches,
        log_every_n_steps=log_every_n_steps,
        **val_check_kwargs,
    )
    print("Running initial validation (step 0 baseline)...")
    pl_trainer.validate(trainer_module, val_loader)
    pl_trainer.fit(trainer_module, loader, val_loader)

    manifest = CanonicalizerManifest(
        canonicalizer_type=args.canonicalizer_type,
        backbone_config=str(Path(backbone_cfg).resolve()),
        backbone_ckpt=str(Path(args.ckpt).resolve()),
        db_path=str(Path(args.db_path).resolve()),
        ood_db_path=str(Path(args.ood_db_path).resolve()),
        use_reverb=getattr(warp, "use_reverb", False),
        stats_hash=profile.stats_hash,
        backbone_kind=backbone_kind,
        latent_n_layers=int(getattr(warp, "n_layers", 1)),
        latent_hidden_size=getattr(warp, "hidden_size", None)
        if args.canonicalizer_type == "latent" else None,
    )
    save_canonicalizer_checkpoint(
        out_dir / ckpt_name,
        warp.state_dict(),
        manifest,
    )
    print(f"Saved {out_dir / ckpt_name}")


if __name__ == "__main__":
    main()
