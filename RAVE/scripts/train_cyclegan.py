#!/usr/bin/env python3
"""Train bidirectional CycleGAN (latent or waveform) on dual frozen BRAVE backbones."""

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
import pytorch_lightning as pl
import torch
import rave
import rave.core
import rave.dataset
from rave.canonicalizer.callbacks import (
    CycleGANExportCallback,
    CycleGANRampCallback,
    CycleGANValVizCallback,
)
from rave.canonicalizer.config import (
    CycleGANManifest,
    resolve_cyclegan_lightning_ckpt,
    save_cyclegan_checkpoint,
)
from rave.canonicalizer.cycle_trainer import CycleGANTrainer
from rave.canonicalizer.dataset import (
    DOMAIN_IN,
    DOMAIN_OOD,
    OodAudioDataset,
    TaggedAudioDataset,
    build_canonicalizer_dataloader,
    ddp_aligned_num_batches,
    ddp_batches_per_rank,
)
from rave.canonicalizer.gin_setup import (
    build_in_domain_discriminator,
    build_latent_discriminator,
    configure_backbone_gin,
    configure_cyclegan_gin,
    resolve_cycle_max_steps,
)
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer
from rave.canonicalizer.waveform_canonicalizer import build_waveform_canonicalizer

import rave.canonicalizer.callbacks  # noqa: F401
import rave.canonicalizer.cycle_trainer  # noqa: F401
import rave.canonicalizer.in_domain_discriminator  # noqa: F401
import rave.canonicalizer.waveform_canonicalizer  # noqa: F401
import rave.canonicalizer.latent_canonicalizer  # noqa: F401
from rave import discriminator, dsp  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/brave_cyclegan.gin")
    p.add_argument(
        "--backbone_x_config",
        required=True,
        help="Frozen domain-X (tap) gin, e.g. configs/brave.gin",
    )
    p.add_argument("--ckpt_x", required=True)
    p.add_argument("--db_path_x", required=True, help="Domain-X LMDB (tap)")
    p.add_argument(
        "--backbone_y_config",
        required=True,
        help="Frozen domain-Y (water) gin, e.g. configs/brave.gin",
    )
    p.add_argument("--ckpt_y", required=True)
    p.add_argument("--db_path_y", required=True, help="Domain-Y LMDB (water)")
    p.add_argument(
        "--canonicalizer_type",
        choices=("waveform", "latent"),
        required=True,
    )
    p.add_argument("--name", required=True)
    p.add_argument("--out_path", default="runs/")
    p.add_argument("--n_signal", type=int, default=131072)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--resume",
        default=None,
        help="Run dir or last.ckpt. Default: auto-resume if last.ckpt exists in the run dir.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore last.ckpt and start a new run in the hashed directory.",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Write last.ckpt every N train steps (0 = only at end).",
    )
    p.add_argument("--wandb_run_id", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", type=int, action="append", default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--val_audio_samples", type=int, default=8)
    p.add_argument("--wandb_project", default="brave")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--log_every_n_steps", type=int, default=None)
    p.add_argument(
        "--log_audio_every_n_steps",
        type=int,
        default=1000,
        help="W&B audio at most every N train steps (default: 1000; 0 = every val)",
    )
    p.add_argument("--calibration_batches", type=int, default=None)
    p.add_argument("--no_calibrate_scales", action="store_true")
    return p.parse_args()


def _add_gin_ext(path: str) -> str:
    return path if path.endswith(".gin") else path + ".gin"


def _resolve(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(_BRAVE_ROOT) / p)


def _load_plain_lmdb_pair(
    db_path: str,
    *,
    sr: int,
    n_signal: int,
    n_channels: int,
    domain: str,
    split_percent: int = 98,
):
    train_base, val_base = rave.dataset.split_train_val(
        db_path,
        sr,
        n_signal,
        percent=split_percent,
        n_channels=n_channels,
    )
    train_base = rave.dataset.maybe_reject_silent(train_base)
    return (
        TaggedAudioDataset(train_base, domain=domain),
        TaggedAudioDataset(val_base, domain=domain),
    )


def _load_frozen_backbone(config_path: str, ckpt_path: str, n_channels: int):
    configure_backbone_gin(config_path, n_channels)
    model = rave.RAVE(n_channels=n_channels)
    run = rave.core.search_for_run(ckpt_path)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    model = model.load_from_checkpoint(run)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    backbone_x_cfg = _resolve(_add_gin_ext(args.backbone_x_config))
    backbone_y_cfg = _resolve(_add_gin_ext(args.backbone_y_config))
    cycle_cfg = _resolve(_add_gin_ext(args.config))

    n_channels = rave.dataset.get_training_channels(args.db_path_y, 0)

    backbone_x = _load_frozen_backbone(backbone_x_cfg, args.ckpt_x, n_channels)
    backbone_y = _load_frozen_backbone(backbone_y_cfg, args.ckpt_y, n_channels)

    configure_cyclegan_gin(cycle_cfg, n_channels, overrides=args.override)
    rave.core.bind_log_audio_every_n_steps(args.log_audio_every_n_steps)

    max_steps = (
        args.max_steps if args.max_steps is not None else resolve_cycle_max_steps()
    )

    if args.canonicalizer_type == "waveform":
        print(
            "WARNING: waveform CycleGAN warps are shelved; prefer "
            "--canonicalizer_type latent with CycleGANTrainer.cycle_domain."
        )
        warp_xy = build_waveform_canonicalizer(
            sample_rate=backbone_y.sr,
            n_channels=n_channels,
        )
        warp_yx = build_waveform_canonicalizer(
            sample_rate=backbone_x.sr,
            n_channels=n_channels,
        )
        ckpt_name = "cyclegan_waveform.ckpt"
        warp_init_mode = "identity"
    else:
        warp_init_mode = "random"
        warp_xy = LatentCanonicalizer(
            latent_size=backbone_x.latent_size, init_mode=warp_init_mode)
        warp_yx = LatentCanonicalizer(
            latent_size=backbone_y.latent_size, init_mode=warp_init_mode)
        ckpt_name = "cyclegan_latent.ckpt"

    cycle_domain = None
    for key in (
        "CycleGANTrainer.cycle_domain",
        "rave.canonicalizer.cycle_trainer.CycleGANTrainer.cycle_domain",
    ):
        try:
            cycle_domain = gin.query_parameter(key)
            break
        except ValueError:
            continue

    def _query_float(keys, default: float) -> float:
        for key in keys:
            try:
                return float(gin.query_parameter(key))
            except ValueError:
                continue
        return default

    lambda_latent_gan = _query_float(
        (
            "CycleGANTrainer.lambda_latent_gan",
            "rave.canonicalizer.cycle_trainer.CycleGANTrainer.lambda_latent_gan",
        ),
        0.0,
    )

    disc_latent_x = None
    disc_latent_y = None
    if cycle_domain == "latent":
        disc_x = build_latent_discriminator(backbone_x.latent_size)
        disc_y = build_latent_discriminator(backbone_y.latent_size)
    else:
        disc_x = build_in_domain_discriminator(n_channels)
        disc_y = build_in_domain_discriminator(n_channels)
        # Track B: waveform cycle + audio D + hybrid latent D
        if lambda_latent_gan > 0.0 and args.canonicalizer_type == "latent":
            disc_latent_x = build_latent_discriminator(backbone_x.latent_size)
            disc_latent_y = build_latent_discriminator(backbone_y.latent_size)

    trainer_module = CycleGANTrainer(
        backbone_x=backbone_x,
        backbone_y=backbone_y,
        warp_xy=warp_xy,
        warp_yx=warp_yx,
        canonicalizer_type=args.canonicalizer_type,
        disc_x=disc_x,
        disc_y=disc_y,
        disc_latent_x=disc_latent_x,
        disc_latent_y=disc_latent_y,
    )

    gin_snapshot = gin.config_str()
    gin_hash = hashlib.md5(gin_snapshot.encode()).hexdigest()[:10]
    run_name = f"{args.name}_{gin_hash}"
    out_dir = Path(args.out_path) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out_dir}")
    with open(out_dir / "config.gin", "w") as config_out:
        config_out.write(gin_snapshot)

    lightning_ckpt = resolve_cyclegan_lightning_ckpt(
        out_dir=out_dir,
        resume=_resolve(args.resume) if args.resume else None,
        fresh=args.fresh,
    )
    if lightning_ckpt is not None:
        print(f"Resuming Lightning checkpoint: {lightning_ckpt}")
    elif args.resume:
        raise FileNotFoundError(
            f"No last.ckpt found for --resume={args.resume} (looked in {out_dir})"
        )

    print(
        f"Training for {max_steps:,} steps "
        f"(cycle_warmup={trainer_module.cycle_warmup_duration:,}, "
        f"gan_ramp={trainer_module.gan_ramp_duration:,}, "
        f"full_gan≈{max(0, max_steps - trainer_module.cycle_warmup_duration - trainer_module.gan_ramp_duration):,})",
    )
    print(
        f"Unfreeze: encoders={trainer_module.unfreeze_encoders}, "
        f"decoders={trainer_module.unfreeze_decoders}, "
        f"backbone_lr={trainer_module.backbone_lr}",
    )
    print(
        f"cycle_domain={trainer_module.cycle_domain}, "
        f"gan_domain={trainer_module.gan_domain}, "
        f"latent_cycle_mode={trainer_module.latent_cycle_mode}, "
        f"ae_aware={trainer_module.use_ae_aware_cycle}, "
        f"direct={trainer_module.use_latent_cycle}, "
        f"hybrid_latent_gan={trainer_module.hybrid_latent_gan}, "
        f"audio_polish_start={trainer_module.audio_polish_start_step}, "
        f"warp_init={warp_init_mode}",
    )

    train_x, val_x = _load_plain_lmdb_pair(
        args.db_path_x,
        sr=backbone_x.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
        domain=DOMAIN_OOD,
    )
    train_y, val_y = _load_plain_lmdb_pair(
        args.db_path_y,
        sr=backbone_y.sr,
        n_signal=args.n_signal,
        n_channels=n_channels,
        domain=DOMAIN_IN,
    )

    num_workers = 0 if sys.platform == "darwin" else args.workers

    loader = build_canonicalizer_dataloader(
        in_domain_dataset=train_y,
        ood_dataset=OodAudioDataset(train_x._base),
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = build_canonicalizer_dataloader(
        in_domain_dataset=val_y,
        ood_dataset=OodAudioDataset(val_x._base),
        batch_size=1 if args.smoke_test else args.batch,
        shuffle=False,
        drop_last=False,
        stratified_batches=False,
        num_workers=num_workers,
    )

    if args.no_calibrate_scales or lightning_ckpt is not None:
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
        print(f"Calibrated loss scales from {n_cal} stratified train batches:")
        for key, value in scales.items():
            print(f"  {key}: {value:.6f}")
        with open(out_dir / "loss_scales.json", "w", encoding="utf-8") as scale_out:
            json.dump(
                {**scales, "calibration_batches": n_cal, "calibrated": True},
                scale_out,
                indent=2,
            )

    from pytorch_lightning.loggers import WandbLogger

    manifest = CycleGANManifest(
        canonicalizer_type=args.canonicalizer_type,
        backbone_x_config=str(Path(backbone_x_cfg).resolve()),
        backbone_x_ckpt=str(Path(args.ckpt_x).resolve()),
        backbone_y_config=str(Path(backbone_y_cfg).resolve()),
        backbone_y_ckpt=str(Path(args.ckpt_y).resolve()),
        db_path_x=str(Path(args.db_path_x).resolve()),
        db_path_y=str(Path(args.db_path_y).resolve()),
        use_reverb=getattr(warp_xy, "use_reverb", False),
        latent_n_layers=int(getattr(warp_xy, "n_layers", 1)),
        latent_hidden_size=getattr(warp_xy, "hidden_size", None)
        if args.canonicalizer_type == "latent" else None,
        cycle_domain=trainer_module.cycle_domain,
        latent_cycle_mode=trainer_module.latent_cycle_mode,
        init_mode=warp_init_mode,
    )

    ckpt_callback_kwargs = dict(
        dirpath=str(out_dir),
        filename="unused",
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    if args.smoke_test:
        ckpt_callback_kwargs["every_n_train_steps"] = 1
    elif args.save_every and args.save_every > 0:
        ckpt_callback_kwargs["every_n_train_steps"] = args.save_every
    checkpoint_cb = pl.callbacks.ModelCheckpoint(**ckpt_callback_kwargs)

    callbacks = [
        checkpoint_cb,
        CycleGANRampCallback(),
        CycleGANValVizCallback(
            out_dir=out_dir,
            num_audio_samples=args.val_audio_samples,
        ),
        CycleGANExportCallback(out_dir / ckpt_name, manifest),
        rave.core.SaveWandbRunIdCallback(str(out_dir)),
    ]

    wandb_kwargs = dict(
        project=args.wandb_project,
        name=run_name,
        save_dir=str(out_dir),
        offline=args.wandb_offline,
        config={
            "db_path_x": args.db_path_x,
            "db_path_y": args.db_path_y,
            "batch": args.batch,
            "n_signal": args.n_signal,
            "max_steps": max_steps,
            "canonicalizer_type": args.canonicalizer_type,
            "cycle_domain": trainer_module.cycle_domain,
            "latent_cycle_mode": trainer_module.latent_cycle_mode,
            "warp_init_mode": warp_init_mode,
            "calibrate_loss_scales": trainer_module.calibrate_loss_scales,
        },
    )
    if trainer_module.loss_scales_calibrated:
        wandb_kwargs["config"].update({
            "stft_loss_scale": trainer_module.stft_loss_scale,
            "gan_loss_scale": trainer_module.gan_loss_scale,
            "fm_loss_scale": trainer_module.fm_loss_scale,
        })
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    wandb_run_id = args.wandb_run_id
    if wandb_run_id is None and lightning_ckpt is not None:
        wandb_run_id = rave.core.find_wandb_run_id(str(out_dir))
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
        print(f"W&B: resuming run id={wandb_run_id}")
    logger = WandbLogger(**wandb_kwargs)

    train_batches = len(loader)
    if train_batches == 0:
        raise SystemExit(
            f"No training batches: x={len(train_x)}, y={len(train_y)}, batch={args.batch}"
        )

    n_devices = (
        len(args.gpu)
        if args.gpu and torch.cuda.is_available() and len(args.gpu) > 1
        else 1
    )
    aligned_train_batches = ddp_aligned_num_batches(train_batches, n_devices)
    batches_per_rank = ddp_batches_per_rank(train_batches, n_devices)

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

    accelerator = "cpu"
    devices: int | list[int] = 1
    strategy = None
    if args.gpu and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.gpu
        if len(args.gpu) > 1:
            from pytorch_lightning.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=True)

    pl_trainer = pl.Trainer(
        max_steps=1 if args.smoke_test else max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        default_root_dir=str(out_dir),
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        limit_val_batches=1 if args.smoke_test else args.val_batches,
        log_every_n_steps=log_every_n_steps,
        **val_check_kwargs,
    )

    ckpt_path = str(lightning_ckpt) if lightning_ckpt is not None else None
    if ckpt_path is None:
        print("Running initial validation (step 0 baseline)...")
        pl_trainer.validate(trainer_module, val_loader)
    else:
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pl_trainer.fit_loop.epoch_loop._batches_that_stepped = loaded["global_step"]

    pl_trainer.fit(trainer_module, loader, val_loader, ckpt_path=ckpt_path)

    save_cyclegan_checkpoint(
        out_dir / ckpt_name,
        warp_xy.state_dict(),
        warp_yx.state_dict(),
        manifest,
    )
    print(f"Saved {out_dir / ckpt_name}")


if __name__ == "__main__":
    main()
