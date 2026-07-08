"""Lightning callbacks for canonicalizer validation monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import gin

from .dataset import DOMAIN_IN, DOMAIN_OOD
from .viz import (
    concat_val_audio_triplets,
    latent_frames_to_points,
    log_wandb_audio,
    log_wandb_figure,
    plot_latent_domain_scatter,
    save_figure,
)

AudioTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@gin.configurable
class CanonicalizerValVizCallback(pl.Callback):
    """
    On validation epoch end:
      1. PCA / t-SNE scatter — in-domain vs OOD latents (post-warp)
      2. W&B audio per domain: ``input | pre_encoder | recon`` × N samples
    """

    def __init__(
        self,
        out_dir: Optional[str | Path] = None,
        scatter_method: str = "pca",
        also_tsne: bool = True,
        max_points_per_domain: int = 512,
        num_audio_samples: int = 8,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.scatter_method = scatter_method
        self.also_tsne = also_tsne
        self.max_points_per_domain = max_points_per_domain
        self.num_audio_samples = num_audio_samples
        self._in_domain_pts: List[np.ndarray] = []
        self._ood_pts: List[np.ndarray] = []
        self._ood_audio: List[AudioTriplet] = []
        self._in_domain_audio: List[AudioTriplet] = []

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._in_domain_pts.clear()
        self._ood_pts.clear()
        self._ood_audio.clear()
        self._in_domain_audio.clear()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if outputs is None:
            return
        z, domains, x_raw, x_pre_enc, y_raw = outputs
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
        log_wandb_audio(pl_module, f"val/audio_{prefix}", wav, sr)
        if self.out_dir is None:
            return
        import soundfile as sf

        viz_dir = self.out_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(viz_dir / f"{prefix}_val_step{step}.wav"), wav, sr)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._in_domain_pts or not self._ood_pts:
            return

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
                save_figure(fig, self.out_dir / "viz" / f"latent_{method}_step{step}.png")
            import matplotlib.pyplot as plt
            plt.close(fig)

        self._log_domain_audio(pl_module, prefix="ood", samples=self._ood_audio, step=step)
        self._log_domain_audio(
            pl_module, prefix="indomain", samples=self._in_domain_audio, step=step)
