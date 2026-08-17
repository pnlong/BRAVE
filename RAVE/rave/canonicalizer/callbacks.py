"""Lightning callbacks for canonicalizer validation monitoring."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import gin

from .backbone import primary_discrete_attr_index
from .dataset import DOMAIN_IN, DOMAIN_OOD
from .viz import (
    concat_val_audio_triplets,
    latent_frames_to_points,
    log_wandb_audio,
    log_wandb_figure,
    plot_latent_domain_scatter,
    plot_ood_class_summary,
    save_figure,
)

AudioTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _unpack_validation_outputs(outputs) -> tuple:
    """Support dict and legacy tuple validation returns."""
    if isinstance(outputs, dict):
        return (
            outputs["z"],
            outputs["domains"],
            outputs["x_raw"],
            outputs["x_enc_in"],
            outputs["y_raw"],
            outputs.get("ood_class_samples") or [],
        )
    z, domains, x_raw, x_enc_in, y_raw = outputs
    return z, domains, x_raw, x_enc_in, y_raw, []


def _class_labels_for_plot(pl_module) -> Dict[int, str]:
    disc_idx = primary_discrete_attr_index(pl_module.backbone)
    if disc_idx is None:
        return {}
    names = getattr(pl_module.backbone, "attribute_names", [])
    attr_name = names[disc_idx] if disc_idx < len(names) else ""
    labels = getattr(pl_module, "discrete_class_labels", {}).get(attr_name) or []
    return {i: str(label) for i, label in enumerate(labels) if label}


def _ddp_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def compute_gan_ramp_factor(
    step: int,
    *,
    delay: int,
    ramp_duration: int,
) -> float:
    """Recon-only until ``delay``, then linear 0→1 over ``ramp_duration`` steps."""
    if step < delay:
        return 0.0
    if ramp_duration <= 0:
        return 1.0
    return min(1.0, (step - delay) / ramp_duration)


@gin.configurable
class CanonicalizerGanRampCallback(pl.Callback):
    """
    Recon-only for ``phase_1_duration`` steps, then linearly ramp adversarial
    loss weight from 0 to 1 over ``gan_ramp_duration`` steps.
    """

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        factor = compute_gan_ramp_factor(
            trainer.global_step,
            delay=int(pl_module.warmup),
            ramp_duration=int(pl_module.gan_ramp_duration),
        )
        pl_module.gan_factor = factor
        pl_module.warmed_up = factor >= 1.0
        if pl_module.spread_active:
            pl_module.spread_factor = compute_gan_ramp_factor(
                trainer.global_step,
                delay=int(pl_module.spread_warmup),
                ramp_duration=int(pl_module.spread_ramp_duration),
            )
            pl_module.spread_warmed_up = pl_module.spread_factor >= 1.0
        else:
            pl_module.spread_factor = 0.0
            pl_module.spread_warmed_up = False


@gin.configurable
class CanonicalizerValVizCallback(pl.Callback):
    """
    On validation epoch end:
      1. PCA / t-SNE scatter — in-domain vs OOD latents (post-warp)
      2. W&B audio per domain: ``input | pre_encoder | recon`` × N samples
      3. Single grouped bar chart of OOD disc/recon by assigned class (Fader)
    """

    def __init__(
        self,
        out_dir: Optional[str | Path] = None,
        scatter_method: str = "pca",
        also_tsne: bool = True,
        max_points_per_domain: int = 512,
        num_audio_samples: int = 8,
        plot_ood_by_class: bool = True,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.scatter_method = scatter_method
        self.also_tsne = also_tsne
        self.max_points_per_domain = max_points_per_domain
        self.num_audio_samples = num_audio_samples
        self.plot_ood_by_class = plot_ood_by_class
        self._in_domain_pts: List[np.ndarray] = []
        self._ood_pts: List[np.ndarray] = []
        self._ood_audio: List[AudioTriplet] = []
        self._in_domain_audio: List[AudioTriplet] = []
        self._ood_disc_by_class: Dict[int, List[float]] = defaultdict(list)
        self._ood_recon_by_class: Dict[int, List[float]] = defaultdict(list)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._in_domain_pts.clear()
        self._ood_pts.clear()
        self._ood_audio.clear()
        self._in_domain_audio.clear()
        self._ood_disc_by_class.clear()
        self._ood_recon_by_class.clear()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if not trainer.is_global_zero or outputs is None:
            return
        z, domains, x_raw, x_pre_enc, y_raw, ood_class_samples = (
            _unpack_validation_outputs(outputs))
        for sample in ood_class_samples:
            class_id = int(sample["class_id"])
            self._ood_recon_by_class[class_id].append(float(sample["recon"]))
            if "disc_logit" in sample:
                self._ood_disc_by_class[class_id].append(float(sample["disc_logit"]))

        for i, dom in enumerate(domains):
            pts = latent_frames_to_points(z[i:i + 1], max_points=self.max_points_per_domain)
            if dom == DOMAIN_IN:
                self._in_domain_pts.append(pts)
            else:
                self._ood_pts.append(pts)

            triplet = (x_raw[i].cpu(), x_pre_enc[i].cpu(), y_raw[i].cpu())
            if dom == DOMAIN_OOD and len(self._ood_audio) < self.num_audio_samples:
                self._ood_audio.append(triplet)
            elif dom == DOMAIN_IN and len(self._in_domain_audio) < self.num_audio_samples:
                self._in_domain_audio.append(triplet)

    def _log_domain_audio(
        self,
        pl_module,
        *,
        prefix: str,
        samples: List[AudioTriplet],
        step: int,
    ) -> None:
        if not samples:
            return
        sr = pl_module.backbone.sr
        wav = concat_val_audio_triplets(samples, max_samples=self.num_audio_samples)
        logged = log_wandb_audio(pl_module, f"val/audio_{prefix}", wav, sr)
        if self.out_dir is None or not logged:
            return
        import soundfile as sf

        viz_dir = self.out_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(viz_dir / f"{prefix}_val_step{step}.wav"), wav, sr)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            if self._in_domain_pts and self._ood_pts:
                in_pts = np.concatenate(self._in_domain_pts, axis=0)
                ood_pts = np.concatenate(self._ood_pts, axis=0)
                step = trainer.global_step

                methods = [self.scatter_method]
                if self.also_tsne and self.scatter_method != "tsne":
                    methods.append("tsne")

                for method in methods:
                    fig = plot_latent_domain_scatter(
                        in_pts,
                        ood_pts,
                        method=method,
                        title=f"Canonicalizer val latents ({method.upper()})",
                        max_points_per_domain=self.max_points_per_domain,
                    )
                    key = f"val/canonicalizer_latent_{method}"
                    log_wandb_figure(pl_module, key, fig)
                    if self.out_dir is not None:
                        save_figure(
                            fig,
                            self.out_dir / "viz" / f"latent_{method}_step{step}.png",
                        )
                    import matplotlib.pyplot as plt
                    plt.close(fig)

                self._log_domain_audio(
                    pl_module, prefix="ood", samples=self._ood_audio, step=step)
                self._log_domain_audio(
                    pl_module,
                    prefix="indomain",
                    samples=self._in_domain_audio,
                    step=step,
                )

                if self.plot_ood_by_class and self._ood_recon_by_class:
                    class_fig = plot_ood_class_summary(
                        self._ood_disc_by_class,
                        self._ood_recon_by_class,
                        class_labels=_class_labels_for_plot(pl_module),
                    )
                    if class_fig is not None:
                        log_wandb_figure(pl_module, "val/ood_by_class", class_fig)
                        if self.out_dir is not None:
                            save_figure(
                                class_fig,
                                self.out_dir / "viz" / f"ood_by_class_step{step}.png",
                            )
                        import matplotlib.pyplot as plt
                        plt.close(class_fig)

        _ddp_barrier()


@gin.configurable
class CycleGANRampCallback(pl.Callback):
    """
    Cycle-only warmup (no adversarial / no D steps), then linear GAN ramp.

    During warmup ``gan_factor=0``; latent spread ramps with GAN when active.
    """

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        factor = compute_gan_ramp_factor(
            trainer.global_step,
            delay=int(pl_module.cycle_warmup_duration),
            ramp_duration=int(pl_module.gan_ramp_duration),
        )
        pl_module.gan_factor = factor
        pl_module.cycle_warmed_up = trainer.global_step >= int(
            pl_module.cycle_warmup_duration)
        if getattr(pl_module, "spread_active", False):
            pl_module.spread_factor = factor
        else:
            pl_module.spread_factor = 0.0


class CycleGANExportCallback(pl.Callback):
    """Write inference ``cyclegan_*.ckpt`` whenever Lightning saves ``last.ckpt``."""

    def __init__(self, path, manifest) -> None:
        super().__init__()
        self.path = Path(path)
        self.manifest = manifest

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        from .config import save_cyclegan_checkpoint

        save_cyclegan_checkpoint(
            self.path,
            pl_module.warp_xy.state_dict(),
            pl_module.warp_yx.state_dict(),
            self.manifest,
        )


@gin.configurable
class CycleGANValVizCallback(pl.Callback):
    """
    Validation monitoring for CycleGAN:

    - ``val/audio_x`` / ``val/audio_y``: ``input | transfer | cycle``
    - ``val/audio_x_to_y`` / ``val/audio_y_to_x``: transfer only
    - ``val/latent_x_pca``: Enc_X(x) vs G_yx latent, colored by domain x/y
    - ``val/latent_y_pca``: Enc_Y(y) vs G_xy latent, colored by domain x/y
    """

    def __init__(
        self,
        out_dir: Optional[str | Path] = None,
        num_audio_samples: int = 8,
        max_points_per_domain: int = 512,
        also_tsne: bool = False,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.num_audio_samples = num_audio_samples
        self.max_points_per_domain = max_points_per_domain
        self.also_tsne = also_tsne
        self._x_audio: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._y_audio: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # X-space: real Enc_X(x) vs transferred z_yx (from y)
        self._x_space_from_x: List[np.ndarray] = []
        self._x_space_from_y: List[np.ndarray] = []
        # Y-space: real Enc_Y(y) vs transferred z_xy (from x)
        self._y_space_from_y: List[np.ndarray] = []
        self._y_space_from_x: List[np.ndarray] = []

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._x_audio.clear()
        self._y_audio.clear()
        self._x_space_from_x.clear()
        self._x_space_from_y.clear()
        self._y_space_from_y.clear()
        self._y_space_from_x.clear()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if not trainer.is_global_zero or outputs is None:
            return
        domains = outputs["domains"]
        fwd = outputs["fwd"]

        if fwd["z_x"].shape[0] > 0:
            self._x_space_from_x.append(
                latent_frames_to_points(
                    fwd["z_x"], max_points=self.max_points_per_domain))
        if fwd["z_yx"].shape[0] > 0:
            self._x_space_from_y.append(
                latent_frames_to_points(
                    fwd["z_yx"], max_points=self.max_points_per_domain))
        if fwd["z_y"].shape[0] > 0:
            self._y_space_from_y.append(
                latent_frames_to_points(
                    fwd["z_y"], max_points=self.max_points_per_domain))
        if fwd["z_xy"].shape[0] > 0:
            self._y_space_from_x.append(
                latent_frames_to_points(
                    fwd["z_xy"], max_points=self.max_points_per_domain))

        x_ptr = 0
        y_ptr = 0
        for dom in domains:
            if dom == DOMAIN_OOD and x_ptr < fwd["x"].shape[0]:
                if len(self._x_audio) < self.num_audio_samples:
                    # x | G_xy(x) | x_cycle
                    triplet = (
                        fwd["x"][x_ptr].cpu(),
                        fwd["y_fake"][x_ptr].cpu(),
                        fwd["x_cycle"][x_ptr].cpu(),
                    )
                    self._x_audio.append(triplet)
                x_ptr += 1
            elif dom == DOMAIN_IN and y_ptr < fwd["y"].shape[0]:
                if len(self._y_audio) < self.num_audio_samples:
                    # y | G_yx(y) | y_cycle
                    triplet = (
                        fwd["y"][y_ptr].cpu(),
                        fwd["x_fake"][y_ptr].cpu(),
                        fwd["y_cycle"][y_ptr].cpu(),
                    )
                    self._y_audio.append(triplet)
                y_ptr += 1

    def _write_wav(self, name: str, wav: np.ndarray, sr: int, step: int) -> None:
        if self.out_dir is None or wav.size == 0:
            return
        import soundfile as sf

        viz_dir = self.out_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(viz_dir / f"{name}_val_step{step}.wav"), wav, sr)

    def _log_triplets(
        self,
        pl_module,
        *,
        key: str,
        file_stem: str,
        samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        step: int,
    ) -> None:
        if not samples:
            return
        sr = pl_module.backbone_y.sr
        wav = concat_val_audio_triplets(samples, max_samples=self.num_audio_samples)
        logged = log_wandb_audio(pl_module, key, wav, sr)
        if logged:
            self._write_wav(file_stem, wav, sr, step)

    def _log_transfers(
        self,
        pl_module,
        *,
        key: str,
        file_stem: str,
        samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        step: int,
    ) -> None:
        """Log middle channel of each triplet (transfer only)."""
        if not samples:
            return
        from .viz import mono_waveform

        sr = pl_module.backbone_y.sr
        chunks = [
            mono_waveform(transfer)
            for _, transfer, _ in samples[: self.num_audio_samples]
        ]
        wav = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        logged = log_wandb_audio(pl_module, key, wav, sr)
        if logged:
            self._write_wav(file_stem, wav, sr, step)

    def _log_latent_space(
        self,
        pl_module,
        *,
        space: str,
        from_x: List[np.ndarray],
        from_y: List[np.ndarray],
        step: int,
    ) -> None:
        if not from_x or not from_y:
            return
        pts_x = np.concatenate(from_x, axis=0)
        pts_y = np.concatenate(from_y, axis=0)
        methods = ["pca"]
        if self.also_tsne:
            methods.append("tsne")
        for method in methods:
            # Args: label_a = first array, label_b = second.
            # For X-space: real x vs transferred-from-y.
            # For Y-space: transferred-from-x vs real y — keep color meaning
            # consistent (x=orange-ish second / y=teal first) by always
            # passing (from_y, from_x) so label_a=y, label_b=x.
            fig = plot_latent_domain_scatter(
                pts_y,
                pts_x,
                method=method,
                title=f"CycleGAN {space}-space latents ({method.upper()})",
                max_points_per_domain=self.max_points_per_domain,
                label_a="domain y",
                label_b="domain x",
            )
            key = f"val/latent_{space}_{method}"
            log_wandb_figure(pl_module, key, fig)
            if self.out_dir is not None:
                save_figure(
                    fig,
                    self.out_dir / "viz" / f"latent_{space}_{method}_step{step}.png",
                )
            import matplotlib.pyplot as plt
            plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            step = trainer.global_step
            self._log_triplets(
                pl_module,
                key="val/audio_x",
                file_stem="x",
                samples=self._x_audio,
                step=step,
            )
            self._log_triplets(
                pl_module,
                key="val/audio_y",
                file_stem="y",
                samples=self._y_audio,
                step=step,
            )
            self._log_transfers(
                pl_module,
                key="val/audio_x_to_y",
                file_stem="x_to_y",
                samples=self._x_audio,
                step=step,
            )
            self._log_transfers(
                pl_module,
                key="val/audio_y_to_x",
                file_stem="y_to_x",
                samples=self._y_audio,
                step=step,
            )
            # X decoder space: Enc_X(real x) vs G_yx warp (from y)
            self._log_latent_space(
                pl_module,
                space="x",
                from_x=self._x_space_from_x,
                from_y=self._x_space_from_y,
                step=step,
            )
            # Y decoder space: Enc_Y(real y) vs G_xy warp (from x)
            self._log_latent_space(
                pl_module,
                space="y",
                from_x=self._y_space_from_x,
                from_y=self._y_space_from_y,
                step=step,
            )
        _ddp_barrier()
