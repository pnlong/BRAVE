"""Lightning callbacks for canonicalizer validation monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch

from .canonicalizer_dataset import DOMAIN_IN, DOMAIN_OOD
from .canonicalizer_viz import (
    latent_frames_to_points,
    log_wandb_audio,
    log_wandb_figure,
    mono_waveform,
    plot_latent_domain_scatter,
    recon_with_warp,
    save_figure,
)


class CanonicalizerValVizCallback(pl.Callback):
    """
    On validation epoch end:
      1. PCA / t-SNE scatter — in-domain vs OOD latents (post-warp)
      2. Audio — OOD recon
      3. Audio — in-domain recon
    """

    def __init__(
        self,
        out_dir: Optional[str | Path] = None,
        scatter_method: str = "pca",
        also_tsne: bool = True,
        max_points_per_domain: int = 512,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.scatter_method = scatter_method
        self.also_tsne = also_tsne
        self.max_points_per_domain = max_points_per_domain
        self._in_domain_pts: List[np.ndarray] = []
        self._ood_pts: List[np.ndarray] = []
        self._ood_audio: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._train_audio: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._in_domain_pts.clear()
        self._ood_pts.clear()
        self._ood_audio = None
        self._train_audio = None

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
        z, domains = outputs
        for i, dom in enumerate(domains):
            pts = latent_frames_to_points(z[i:i + 1], max_points=self.max_points_per_domain)
            if dom == DOMAIN_IN:
                self._in_domain_pts.append(pts)
            else:
                self._ood_pts.append(pts)

        x_raw, attr_raw, domain = batch
        for i, dom in enumerate(domain):
            if dom == DOMAIN_OOD and self._ood_audio is None:
                self._ood_audio = (x_raw[i].cpu(), attr_raw[i].cpu())
            if dom == DOMAIN_IN and self._train_audio is None:
                self._train_audio = (x_raw[i].cpu(), attr_raw[i].cpu())

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

        sr = pl_module.fader.sr
        if self._ood_audio is not None:
            x, attr = self._ood_audio
            x = x.to(pl_module.device)
            attr = attr.to(pl_module.device)
            y = recon_with_warp(pl_module, x, attr)
            log_wandb_audio(pl_module, "val/audio_ood_recon", mono_waveform(y), sr)
            if self.out_dir is not None:
                import soundfile as sf
                p = self.out_dir / "viz" / f"ood_recon_step{step}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), mono_waveform(y), sr)

        if self._train_audio is not None:
            x, attr = self._train_audio
            x = x.to(pl_module.device)
            attr = attr.to(pl_module.device)
            y = recon_with_warp(pl_module, x, attr)
            log_wandb_audio(pl_module, "val/audio_indomain_recon", mono_waveform(y), sr)
            if self.out_dir is not None:
                import soundfile as sf
                p = self.out_dir / "viz" / f"indomain_recon_step{step}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), mono_waveform(y), sr)
