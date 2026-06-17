"""Stage-1 Lightning trainer for waveform / latent canonicalizers."""

from __future__ import annotations

from typing import Optional, Tuple

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import hinge_gan
from ..model import _pqmf_decode
from .attributes import compute_descriptor_matrix
from .canonicalizer_config import DomainProfile
from .canonicalizer_dataset import DOMAIN_IN, DOMAIN_OOD
from .canonicalizer_losses import rms_recon_l1
from .latent_domain_discriminator import LatentDomainDiscriminator


@gin.configurable
class CanonicalizerTrainer(pl.LightningModule):
    """
    Train waveform_canonicalizer or latent_canonicalizer with frozen FaderRAVE
    (encoder optionally unfrozen). Optional latent-domain adversary on z.
    """

    automatic_optimization = False

    def __init__(
        self,
        fader: nn.Module,
        warp: nn.Module,
        canonicalizer_type: str,
        domain_profile: DomainProfile,
        latent_mean: Optional[torch.Tensor] = None,
        latent_domain_disc: Optional[LatentDomainDiscriminator] = None,
        lambda_identity: float = 10.0,
        lambda_latent_stat: float = 1.0,
        lambda_descriptor: float = 0.5,
        lambda_latent_adv: float = 0.1,
        lambda_rms_recon: float = 1.0,
        recon_ood_mode: str = "rms",
        recon_in_domain_mode: str = "stft",
        lr: float = 1e-3,
        disc_lr: float = 2e-4,
        unfreeze_encoder: bool = False,
        encoder_lr: float = 1e-5,
    ) -> None:
        super().__init__()
        if canonicalizer_type not in ("waveform", "latent"):
            raise ValueError("canonicalizer_type must be waveform or latent")

        self.fader = fader
        self.warp = warp
        self.canonicalizer_type = canonicalizer_type
        self.domain_profile = domain_profile
        self.latent_domain_disc = latent_domain_disc
        self.lambda_identity = lambda_identity
        self.lambda_latent_stat = lambda_latent_stat
        self.lambda_descriptor = lambda_descriptor
        self.lambda_latent_adv = lambda_latent_adv
        self.lambda_rms_recon = lambda_rms_recon
        if recon_ood_mode not in ("stft", "rms", "both"):
            raise ValueError("recon_ood_mode must be stft, rms, or both")
        if recon_in_domain_mode not in ("stft", "rms", "both"):
            raise ValueError("recon_in_domain_mode must be stft, rms, or both")
        self.recon_ood_mode = recon_ood_mode
        self.recon_in_domain_mode = recon_in_domain_mode
        self.lr = lr
        self.disc_lr = disc_lr
        self.unfreeze_encoder = unfreeze_encoder
        self.encoder_lr = encoder_lr

        self.register_buffer(
            "latent_mean",
            latent_mean if latent_mean is not None else torch.zeros(fader.latent_size),
        )
        desc_means = domain_profile.descriptor_mean_vector()
        self.register_buffer("descriptor_means", desc_means)

        for p in self.fader.parameters():
            p.requires_grad = False
        if self.unfreeze_encoder:
            for p in self.fader.encoder.parameters():
                p.requires_grad = True

    def _set_fader_train_mode(self) -> None:
        if self.unfreeze_encoder:
            self.fader.encoder.train()
            self.fader.decoder.eval()
            if self.latent_domain_disc is not None:
                self.latent_domain_disc.eval()
        else:
            self.fader.eval()

    def configure_optimizers(self):
        warp_groups = [{"params": self.warp.parameters(), "lr": self.lr}]
        if self.unfreeze_encoder:
            warp_groups.append({
                "params": [p for p in self.fader.encoder.parameters() if p.requires_grad],
                "lr": self.encoder_lr,
            })
        warp_opt = torch.optim.Adam(warp_groups, betas=(0.5, 0.9))
        if self.latent_domain_disc is None:
            return warp_opt
        disc_opt = torch.optim.Adam(
            self.latent_domain_disc.parameters(),
            lr=self.disc_lr,
            betas=(0.5, 0.9),
        )
        return [warp_opt, disc_opt]

    def _optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        opts = self.optimizers()
        if self.latent_domain_disc is None:
            return opts, None
        warp_opt, disc_opt = opts
        return warp_opt, disc_opt

    def _forward_recon(
        self,
        x_raw: torch.Tensor,
        attr_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x_raw.shape[:-2]
        attr, _ = self.fader._prepare_attributes(attr_raw)

        if self.canonicalizer_type == "waveform":
            x_enc_in = self.warp(x_raw)
            z, x_multiband = self.fader.encode(x_enc_in, return_mb=True)
            z, _z_pre = self.fader.encoder.reparametrize(z)[:2]
            x_compare = x_raw
            z_identity_ref = None
        else:
            x_enc_in = x_raw
            z, x_multiband = self.fader.encode(x_raw, return_mb=True)
            z, _z_pre = self.fader.encoder.reparametrize(z)[:2]
            z_identity_ref = z
            z = self.warp(z)
            x_compare = x_raw

        z_cond = torch.cat([z, attr], dim=1) if self.fader.num_attributes else z
        y_multiband = self.fader.decoder(z_cond)
        y_raw = y_multiband
        if self.fader.output_mode == "pqmf":
            y_raw = _pqmf_decode(
                self.fader.pqmf, y_multiband,
                batch_size=batch_size, n_channels=self.fader.n_channels)

        t = min(x_compare.shape[-1], y_raw.shape[-1])
        x_compare = x_compare[..., :t]
        y_raw = y_raw[..., :t]
        x_multiband = x_multiband[..., :x_multiband.shape[-1]]
        y_multiband = y_multiband[..., :x_multiband.shape[-1]]
        return z, x_compare, x_multiband, y_raw, y_multiband, x_enc_in, z_identity_ref

    def _stft_recon_loss(
        self,
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        x_cmp: torch.Tensor,
        y_raw: torch.Tensor,
    ) -> torch.Tensor:
        mb_dist = self.fader.multiband_audio_distance(x_mb, y_mb)
        fb_dist = self.fader.audio_distance(x_cmp, y_raw)
        return sum(mb_dist.values()) + sum(fb_dist.values())

    def _recon_loss_for_mask(
        self,
        mask: torch.Tensor,
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        x_cmp: torch.Tensor,
        y_raw: torch.Tensor,
        z: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (stft_recon, rms_recon) for samples selected by mask."""
        zero = torch.tensor(0.0, device=z.device)
        if not mask.any():
            return zero, zero
        n_frames = z.shape[-1]
        loss_stft = zero
        loss_rms = zero
        if mode in ("stft", "both"):
            loss_stft = self._stft_recon_loss(
                x_mb[mask], y_mb[mask], x_cmp[mask], y_raw[mask])
        if mode in ("rms", "both"):
            loss_rms = rms_recon_l1(y_raw[mask], x_cmp[mask], n_frames)
        return loss_stft, loss_rms

    def _descriptor_loss(self, x_warped: torch.Tensor) -> torch.Tensor:
        attrs = self.domain_profile.descriptor_loss_attrs
        if not attrs:
            return torch.tensor(0.0, device=x_warped.device)
        mono = x_warped.mean(dim=1)[0].detach().cpu().numpy()
        mat = compute_descriptor_matrix(mono, self.fader.sr, attrs, latent_length=32)
        vec = torch.tensor(mat.mean(axis=1), device=x_warped.device, dtype=x_warped.dtype)
        return F.mse_loss(vec, self.descriptor_means.to(x_warped.device))

    def _latent_domain_adv(
        self,
        z: torch.Tensor,
        in_mask: torch.Tensor,
        ood_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (discriminator_loss, generator_adversarial_loss)."""
        zero = torch.tensor(0.0, device=z.device)
        if (
            self.latent_domain_disc is None
            or not in_mask.any()
            or not ood_mask.any()
        ):
            return zero, zero

        z_real = z[in_mask].detach()
        z_fake = z[ood_mask]
        score_real = self.latent_domain_disc(z_real)
        score_fake_d = self.latent_domain_disc(z_fake.detach())
        loss_d, _ = hinge_gan(score_real, score_fake_d)

        score_fake_g = self.latent_domain_disc(z_fake)
        loss_adv = -score_fake_g.mean()
        return loss_d, loss_adv

    def training_step(self, batch, batch_idx):
        warp_opt, disc_opt = self._optimizers()
        self._set_fader_train_mode()
        if self.latent_domain_disc is not None:
            self.latent_domain_disc.train()

        x_raw, attr_raw, domain = batch
        x_raw = x_raw.to(self.device)
        attr_raw = attr_raw.to(self.device)
        is_in = [d == DOMAIN_IN for d in domain]
        in_mask = torch.tensor(is_in, device=self.device)
        ood_mask = ~in_mask

        z, x_cmp, x_mb, y_raw, y_mb, x_enc_in, z_identity_ref = self._forward_recon(
            x_raw, attr_raw)

        loss_recon_stft = torch.tensor(0.0, device=self.device)
        loss_recon_rms = torch.tensor(0.0, device=self.device)
        stft_in, rms_in = self._recon_loss_for_mask(
            in_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_in_domain_mode)
        stft_ood, rms_ood = self._recon_loss_for_mask(
            ood_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_ood_mode)
        loss_recon_stft = stft_in + stft_ood
        loss_recon_rms = rms_in + rms_ood
        loss_recon = loss_recon_stft + self.lambda_rms_recon * loss_recon_rms

        loss_id = torch.tensor(0.0, device=self.device)
        loss_lat = torch.tensor(0.0, device=self.device)
        loss_desc = torch.tensor(0.0, device=self.device)

        if in_mask.any():
            if self.canonicalizer_type == "waveform":
                loss_id = F.l1_loss(x_enc_in[in_mask], x_raw[in_mask])
            else:
                loss_id = F.mse_loss(z[in_mask], z_identity_ref[in_mask])

        if ood_mask.any():
            z_ood = z[ood_mask].mean(dim=(0, 2))
            loss_lat = F.mse_loss(z_ood, self.latent_mean)
            if self.canonicalizer_type == "waveform":
                loss_desc = self._descriptor_loss(x_enc_in[ood_mask])

        loss_d, loss_adv = self._latent_domain_adv(z, in_mask, ood_mask)

        if disc_opt is not None and loss_d.requires_grad:
            disc_opt.zero_grad()
            self.manual_backward(loss_d)
            disc_opt.step()

        loss = (
            loss_recon
            + self.lambda_identity * loss_id
            + self.lambda_latent_stat * loss_lat
            + self.lambda_descriptor * loss_desc
            + self.lambda_latent_adv * loss_adv
        )

        warp_opt.zero_grad()
        self.manual_backward(loss)
        warp_opt.step()

        self.log("canon/loss", loss, prog_bar=True)
        self.log("canon/recon", loss_recon)
        self.log("canon/recon_stft", loss_recon_stft)
        self.log("canon/recon_rms", loss_recon_rms)
        self.log("canon/identity", loss_id)
        self.log("canon/latent_stat", loss_lat)
        self.log("canon/descriptor", loss_desc)
        self.log("canon/latent_adv", loss_adv)
        self.log("canon/latent_disc", loss_d)
        return loss

    def validation_step(self, batch, batch_idx):
        self.fader.eval()
        x_raw, attr_raw, domain = batch
        x_raw = x_raw.to(self.device)
        attr_raw = attr_raw.to(self.device)

        with torch.no_grad():
            z, x_cmp, _, y_raw, _, _, _ = self._forward_recon(x_raw, attr_raw)

        loss = self.fader.audio_distance(x_cmp, y_raw)
        val_loss = sum(loss.values())
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        domains = list(domain) if isinstance(domain, (list, tuple)) else [domain]
        return z.detach(), domains
