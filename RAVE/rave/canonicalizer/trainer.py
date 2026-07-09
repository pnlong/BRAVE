"""Stage-1 Lightning trainer for waveform / latent canonicalizers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..core import mean_difference
from ..model import _pqmf_decode
from .backbone import prepare_decode_attributes
from .dataset import DOMAIN_IN, DOMAIN_OOD
from .in_domain_discriminator import InDomainAudioDiscriminator
from .losses import (
    empirical_adversarial_loss_scale,
    empirical_loss_scale,
    normalize_loss,
    resolve_gan_loss,
    rms_recon_l1,
    weighted_recon_loss,
)


@gin.configurable
class CanonicalizerTrainer(pl.LightningModule):
    """
    Train waveform or latent canonicalizer on a frozen RAVE / FaderRAVE backbone.

    One-way CycleGAN-style audio GAN (X=OOD → Y=in-domain) plus optional recon.
    """

    automatic_optimization = False

    def __init__(
        self,
        backbone: nn.Module,
        warp: nn.Module,
        canonicalizer_type: str,
        in_domain_disc: Optional[InDomainAudioDiscriminator] = None,
        lambda_gan: float = 1.0,
        lambda_rec: float = 0.0,
        recon_stft_weight: float = 0.9,
        recon_rms_weight: float = 0.1,
        stft_loss_scale: float = 45.0,
        rms_loss_scale: float = 0.3,
        gan_loss_scale: float = 1.0,
        fm_loss_scale: float = 0.5,
        calibrate_loss_scales: bool = True,
        calibration_batches: int = 16,
        loss_scale_min: float = 1e-3,
        lambda_feature_matching: float = 10.0,
        recon_ood_mode: str = "rms",
        recon_in_domain_mode: str = "rms",
        gan_loss: str = "hinge",
        lr: float = 1e-3,
        disc_lr: float = 2e-4,
        phase_1_duration: int = 500,
        gan_ramp_duration: int = 5000,
        update_discriminator_every: int = 2,
        num_skipped_features: int = 1,
        unfreeze_encoder: bool = False,
        encoder_lr: float = 1e-5,
        encode_use_mean: bool = True,
        # Deprecated aliases (ignored)
        fader: Optional[nn.Module] = None,
        domain_profile=None,
        latent_domain_disc=None,
        lambda_identity: float = 0.0,
        lambda_descriptor: float = 0.0,
        lambda_latent_adv: float = 0.0,
    ) -> None:
        super().__init__()
        if canonicalizer_type not in ("waveform", "latent"):
            raise ValueError("canonicalizer_type must be waveform or latent")

        if fader is not None and backbone is None:
            backbone = fader
        self.backbone = backbone
        self.warp = warp
        self.canonicalizer_type = canonicalizer_type
        if in_domain_disc is not None and isinstance(in_domain_disc, type):
            in_domain_disc = in_domain_disc(n_channels=backbone.n_channels)
        self.in_domain_disc = in_domain_disc
        self.lambda_gan = lambda_gan
        self.lambda_rec = lambda_rec
        self.recon_stft_weight = recon_stft_weight
        self.recon_rms_weight = recon_rms_weight
        self.stft_loss_scale = stft_loss_scale
        self.rms_loss_scale = rms_loss_scale
        self.gan_loss_scale = gan_loss_scale
        self.fm_loss_scale = fm_loss_scale
        self.calibrate_loss_scales = calibrate_loss_scales
        self.calibration_batches = calibration_batches
        self.loss_scale_min = loss_scale_min
        self.loss_scales_calibrated = False
        self.lambda_feature_matching = lambda_feature_matching
        if recon_ood_mode not in ("stft", "rms", "both"):
            raise ValueError("recon_ood_mode must be stft, rms, or both")
        if recon_in_domain_mode not in ("stft", "rms", "both"):
            raise ValueError("recon_in_domain_mode must be stft, rms, or both")
        self.recon_ood_mode = recon_ood_mode
        self.recon_in_domain_mode = recon_in_domain_mode
        self.gan_loss_fn = resolve_gan_loss(gan_loss)
        self.lr = lr
        self.disc_lr = disc_lr
        self.warmup = phase_1_duration
        self.gan_ramp_duration = gan_ramp_duration
        self.gan_factor = 0.0
        self.warmed_up = False
        self.update_discriminator_every = update_discriminator_every
        self.num_skipped_features = num_skipped_features
        self.unfreeze_encoder = unfreeze_encoder
        self.encoder_lr = encoder_lr
        self.encode_use_mean = encode_use_mean

        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.unfreeze_encoder:
            for p in self.backbone.encoder.parameters():
                p.requires_grad = True

    @property
    def fader(self):
        """Backward-compatible alias used by validation callbacks."""
        return self.backbone

    def _set_backbone_train_mode(self) -> None:
        if self.unfreeze_encoder:
            self.backbone.encoder.train()
            self.backbone.decoder.eval()
            if self.in_domain_disc is not None:
                self.in_domain_disc.eval()
        else:
            self.backbone.eval()

    def configure_optimizers(self):
        warp_groups = [{"params": self.warp.parameters(), "lr": self.lr}]
        if self.unfreeze_encoder:
            warp_groups.append({
                "params": [
                    p for p in self.backbone.encoder.parameters() if p.requires_grad
                ],
                "lr": self.encoder_lr,
            })
        warp_opt = torch.optim.Adam(warp_groups, betas=(0.5, 0.9))
        if self.in_domain_disc is None:
            return warp_opt
        disc_opt = torch.optim.Adam(
            self.in_domain_disc.parameters(),
            lr=self.disc_lr,
            betas=(0.5, 0.9),
        )
        return [warp_opt, disc_opt]

    def _optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        opts = self.optimizers()
        if self.in_domain_disc is None:
            return opts, None
        warp_opt, disc_opt = opts
        return warp_opt, disc_opt

    def _encode_latent(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to content latent; optionally use VAE mean (no sampling)."""
        z_raw, x_multiband = self.backbone.encode(x, return_mb=True)
        if self.encode_use_mean:
            from .. import blocks

            if isinstance(self.backbone.encoder, blocks.VariationalEncoder):
                z = z_raw.chunk(2, dim=1)[0]
            else:
                z, _ = self.backbone.encoder.reparametrize(z_raw)[:2]
        else:
            z, _ = self.backbone.encoder.reparametrize(z_raw)[:2]
        return z, x_multiband

    def _forward_recon(
        self,
        x_raw: torch.Tensor,
        attr_raw: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        batch_size = x_raw.shape[:-2]
        attr = prepare_decode_attributes(self.backbone, attr_raw)

        if self.canonicalizer_type == "waveform":
            x_enc_in = self.warp(x_raw)
            z, x_multiband = self._encode_latent(x_enc_in)
            x_compare = x_raw
        else:
            x_enc_in = x_raw
            z, x_multiband = self._encode_latent(x_raw)
            z = self.warp(z)
            x_compare = x_raw

        z_cond = torch.cat([z, attr], dim=1) if attr is not None else z
        y_multiband = self.backbone.decoder(z_cond)
        y_raw = y_multiband
        if self.backbone.output_mode == "pqmf":
            y_raw = _pqmf_decode(
                self.backbone.pqmf, y_multiband,
                batch_size=batch_size, n_channels=self.backbone.n_channels)

        t = min(x_compare.shape[-1], y_raw.shape[-1])
        x_compare = x_compare[..., :t]
        y_raw = y_raw[..., :t]
        x_multiband = x_multiband[..., :x_multiband.shape[-1]]
        y_multiband = y_multiband[..., :x_multiband.shape[-1]]
        return z, x_compare, x_multiband, y_raw, y_multiband, x_enc_in

    def _stft_recon_loss(
        self,
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        x_cmp: torch.Tensor,
        y_raw: torch.Tensor,
    ) -> torch.Tensor:
        mb_dist = self.backbone.multiband_audio_distance(x_mb, y_mb)
        fb_dist = self.backbone.audio_distance(x_cmp, y_raw)
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

    def _disc_features(
        self,
        y_real: torch.Tensor,
        y_fake: torch.Tensor,
        *,
        detach: bool,
    ) -> tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        if detach:
            y_real = y_real.detach()
            y_fake = y_fake.detach()
        feat_real = self.in_domain_disc(y_real)
        feat_fake = self.in_domain_disc(y_fake)
        return feat_real, feat_fake

    def _audio_gan_d(
        self,
        feat_real: List[List[torch.Tensor]],
        feat_fake: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        zero = torch.tensor(0.0, device=feat_real[0][-1].device)
        if self.in_domain_disc is None:
            return zero
        loss_d, _ = InDomainAudioDiscriminator.gan_losses(
            feat_real, feat_fake, self.gan_loss_fn)
        return loss_d

    def _audio_gan_g(
        self,
        feat_fake: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        zero = torch.tensor(0.0, device=feat_fake[0][-1].device)
        if self.in_domain_disc is None or not feat_fake:
            return zero
        loss_g = torch.tensor(0.0, device=feat_fake[0][-1].device)
        for scale in feat_fake:
            _, g = self.gan_loss_fn(scale[-1].detach(), scale[-1])
            loss_g = loss_g + g
        return loss_g / max(len(feat_fake), 1)

    def _feature_matching_loss(
        self,
        feat_real: List[List[torch.Tensor]],
        feat_fake: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        zero = torch.tensor(0.0, device=feat_fake[0][-1].device)
        if self.in_domain_disc is None or not feat_fake:
            return zero
        loss_fm = zero
        n_scales = len(feat_real)
        for scale_real, scale_fake in zip(feat_real, feat_fake):
            real_layers = scale_real[self.num_skipped_features:]
            fake_layers = scale_fake[self.num_skipped_features:]
            if not real_layers:
                continue
            current = sum(
                mean_difference(r.detach(), f, norm="L1")
                for r, f in zip(real_layers, fake_layers)
            ) / len(real_layers)
            loss_fm = loss_fm + current
        return loss_fm / max(n_scales, 1)

    @staticmethod
    def _mean_fake_logit(feat_fake: List[List[torch.Tensor]]) -> torch.Tensor:
        return sum(scale[-1].mean() for scale in feat_fake) / max(len(feat_fake), 1)

    def _parse_batch(self, batch):
        if len(batch) == 3:
            x_raw, attr_raw, domain = batch
        else:
            raise ValueError("batch must be (audio, attr|None, domain)")
        return x_raw, attr_raw, domain

    def _domain_masks(self, domain) -> tuple[torch.Tensor, torch.Tensor]:
        is_in = [d == DOMAIN_IN for d in domain]
        in_mask = torch.tensor(is_in, device=self.device)
        return in_mask, ~in_mask

    def _domain_recon_loss(
        self,
        stft: torch.Tensor,
        rms: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_recon_loss(
            stft,
            rms,
            stft_weight=self.recon_stft_weight,
            rms_weight=self.recon_rms_weight,
            stft_scale=self.stft_loss_scale,
            rms_scale=self.rms_loss_scale,
        )

    def _uses_stft_recon(self) -> bool:
        return (
            self.recon_ood_mode in ("stft", "both")
            or self.recon_in_domain_mode in ("stft", "both")
        )

    def _uses_rms_recon(self) -> bool:
        return (
            self.recon_ood_mode in ("rms", "both")
            or self.recon_in_domain_mode in ("rms", "both")
        )

    @torch.no_grad()
    def _batch_raw_losses(
        self,
        batch,
        *,
        include_adversarial: bool,
    ) -> Dict[str, Optional[float]]:
        """Collect unnormalized loss components for one batch."""
        self.backbone.eval()
        self.warp.eval()
        if self.in_domain_disc is not None:
            self.in_domain_disc.eval()

        x_raw, attr_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        if attr_raw is not None:
            attr_raw = attr_raw.to(self.device)
        in_mask, ood_mask = self._domain_masks(domain)

        z, x_cmp, x_mb, y_raw, y_mb, _ = self._forward_recon(x_raw, attr_raw)
        stft_in, rms_in = self._recon_loss_for_mask(
            in_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_in_domain_mode)
        stft_ood, rms_ood = self._recon_loss_for_mask(
            ood_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_ood_mode)

        out: Dict[str, Optional[float]] = {
            "stft": None,
            "rms": None,
            "gan": None,
            "fm": None,
        }
        if self._uses_stft_recon():
            out["stft"] = float((stft_in + stft_ood).detach().cpu())
        if self._uses_rms_recon():
            out["rms"] = float((rms_in + rms_ood).detach().cpu())

        if (
            include_adversarial
            and self.in_domain_disc is not None
            and in_mask.any()
            and ood_mask.any()
        ):
            y_real = y_raw[in_mask]
            y_fake = y_raw[ood_mask]
            feat_real, feat_fake = self._disc_features(y_real, y_fake, detach=True)
            loss_gan = self._audio_gan_g(feat_fake)
            loss_fm = self._feature_matching_loss(feat_real, feat_fake)
            out["gan"] = float(loss_gan.detach().cpu())
            out["fm"] = float(loss_fm.detach().cpu())
        return out

    @torch.no_grad()
    def calibrate_loss_scales_from_loader(
        self,
        dataloader: Iterable,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Set loss scales from mean raw losses at identity init on stratified batches.

        Measures E[L_raw] over the first ``max_batches`` train batches (warp +
        frozen backbone at initialization). STFT/RMS scales use the measured mean
        (clamped to ``loss_scale_min``). GAN/FM keep gin fallbacks when the
        identity-warp startup signal is too weak to measure (near-zero adversarial
        loss before the GAN ramp).
        """
        if max_batches is None:
            max_batches = self.calibration_batches
        max_batches = max(1, max_batches)

        buckets: Dict[str, List[float]] = {
            "stft": [],
            "rms": [],
            "gan": [],
            "fm": [],
        }
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            raw = self._batch_raw_losses(batch, include_adversarial=True)
            for key, value in raw.items():
                if value is not None:
                    buckets[key].append(value)

        scales: Dict[str, float] = {
            "stft_loss_scale": self.stft_loss_scale,
            "rms_loss_scale": self.rms_loss_scale,
            "gan_loss_scale": self.gan_loss_scale,
            "fm_loss_scale": self.fm_loss_scale,
        }
        if buckets["stft"]:
            scales["stft_loss_scale"] = empirical_loss_scale(
                buckets["stft"], self.loss_scale_min)
            self.stft_loss_scale = scales["stft_loss_scale"]
        if buckets["rms"]:
            scales["rms_loss_scale"] = empirical_loss_scale(
                buckets["rms"], self.loss_scale_min)
            self.rms_loss_scale = scales["rms_loss_scale"]
        if buckets["gan"]:
            scales["gan_loss_scale"] = empirical_adversarial_loss_scale(
                buckets["gan"],
                self.gan_loss_scale,
                self.loss_scale_min,
            )
            self.gan_loss_scale = scales["gan_loss_scale"]
        if buckets["fm"]:
            scales["fm_loss_scale"] = empirical_adversarial_loss_scale(
                buckets["fm"],
                self.fm_loss_scale,
                self.loss_scale_min,
            )
            self.fm_loss_scale = scales["fm_loss_scale"]

        self.loss_scales_calibrated = True
        return scales

    def _log_train_metrics(
        self,
        *,
        loss: torch.Tensor,
        loss_gan: torch.Tensor,
        loss_gan_norm: torch.Tensor,
        loss_recon: torch.Tensor,
        loss_recon_stft: torch.Tensor,
        loss_recon_rms: torch.Tensor,
        loss_recon_in: torch.Tensor,
        loss_recon_ood: torch.Tensor,
        loss_d: torch.Tensor,
        loss_fm: torch.Tensor,
        loss_fm_norm: torch.Tensor,
        batch_size: int,
        log_audio_disc: bool = False,
    ) -> None:
        # Normalized terms (~1 at calibration) — use these when comparing λ weights.
        self.log("canon/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("canon/recon_norm", loss_recon, batch_size=batch_size)
        self.log("canon/recon_in_norm", loss_recon_in, batch_size=batch_size)
        self.log("canon/recon_ood_norm", loss_recon_ood, batch_size=batch_size)
        self.log("canon/gan_norm", loss_gan_norm, batch_size=batch_size)
        self.log("canon/fm_norm", loss_fm_norm, batch_size=batch_size)
        # Raw terms — diagnostic magnitudes before scale division.
        self.log("canon/recon_stft_raw", loss_recon_stft, batch_size=batch_size)
        self.log("canon/recon_rms_raw", loss_recon_rms, batch_size=batch_size)
        self.log("canon/gan_raw", loss_gan, batch_size=batch_size)
        self.log("canon/fm_raw", loss_fm, batch_size=batch_size)
        if log_audio_disc:
            self.log("canon/audio_disc", loss_d, batch_size=batch_size)
        self.log("canon/gan_factor", float(self.gan_factor), batch_size=batch_size)
        self.log("canon/warmed_up", float(self.warmed_up), batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        warp_opt, disc_opt = self._optimizers()
        self._set_backbone_train_mode()
        if self.in_domain_disc is not None:
            self.in_domain_disc.train()

        x_raw, attr_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        if attr_raw is not None:
            attr_raw = attr_raw.to(self.device)
        batch_size = x_raw.size(0)
        in_mask, ood_mask = self._domain_masks(domain)

        z, x_cmp, x_mb, y_raw, y_mb, _ = self._forward_recon(x_raw, attr_raw)

        stft_in, rms_in = self._recon_loss_for_mask(
            in_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_in_domain_mode)
        stft_ood, rms_ood = self._recon_loss_for_mask(
            ood_mask, x_mb, y_mb, x_cmp, y_raw, z, self.recon_ood_mode)
        loss_recon_stft = stft_in + stft_ood
        loss_recon_rms = rms_in + rms_ood
        loss_recon_in = self._domain_recon_loss(stft_in, rms_in)
        loss_recon_ood = self._domain_recon_loss(stft_ood, rms_ood)
        loss_recon = loss_recon_in + loss_recon_ood

        zero = torch.tensor(0.0, device=self.device)
        loss_d = zero
        loss_gan = zero
        loss_fm = zero
        loss_gan_norm = zero
        loss_fm_norm = zero

        has_mixed = in_mask.any() and ood_mask.any()
        gan_active = (
            self.gan_factor > 0.0
            and self.in_domain_disc is not None
            and has_mixed
        )
        is_disc_step = (
            gan_active
            and not (batch_idx % self.update_discriminator_every)
        )

        if is_disc_step:
            y_real = y_raw[in_mask]
            y_fake = y_raw[ood_mask]
            feat_real, feat_fake = self._disc_features(y_real, y_fake, detach=True)
            loss_d = self._audio_gan_d(feat_real, feat_fake)
            if disc_opt is not None and loss_d.requires_grad:
                disc_opt.zero_grad()
                self.manual_backward(loss_d)
                disc_opt.step()
            self._log_train_metrics(
                loss=loss_d,
                loss_gan=loss_gan,
                loss_gan_norm=loss_gan_norm,
                loss_recon=loss_recon,
                loss_recon_stft=loss_recon_stft,
                loss_recon_rms=loss_recon_rms,
                loss_recon_in=loss_recon_in,
                loss_recon_ood=loss_recon_ood,
                loss_d=loss_d,
                loss_fm=loss_fm,
                loss_fm_norm=loss_fm_norm,
                batch_size=batch_size,
                log_audio_disc=True,
            )
            return loss_d

        if gan_active:
            y_real = y_raw[in_mask]
            y_fake = y_raw[ood_mask]
            feat_real, feat_fake = self._disc_features(y_real, y_fake, detach=False)
            loss_gan = self._audio_gan_g(feat_fake)
            loss_fm = self._feature_matching_loss(feat_real, feat_fake)
            loss_gan_norm = normalize_loss(loss_gan, self.gan_loss_scale)
            loss_fm_norm = normalize_loss(loss_fm, self.fm_loss_scale)

        loss = (
            self.lambda_rec * loss_recon
            + self.gan_factor * self.lambda_gan * loss_gan_norm
            + self.gan_factor * self.lambda_feature_matching * loss_fm_norm
        )

        warp_opt.zero_grad()
        if loss.requires_grad:
            self.manual_backward(loss)
            warp_opt.step()

        self._log_train_metrics(
            loss=loss,
            loss_gan=loss_gan,
            loss_gan_norm=loss_gan_norm,
            loss_recon=loss_recon,
            loss_recon_stft=loss_recon_stft,
            loss_recon_rms=loss_recon_rms,
            loss_recon_in=loss_recon_in,
            loss_recon_ood=loss_recon_ood,
            loss_d=loss_d,
            loss_fm=loss_fm,
            loss_fm_norm=loss_fm_norm,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.backbone.eval()
        x_raw, attr_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        if attr_raw is not None:
            attr_raw = attr_raw.to(self.device)
        batch_size = x_raw.size(0)
        in_mask, ood_mask = self._domain_masks(domain)

        with torch.no_grad():
            z, x_cmp, x_mb, y_raw, y_mb, x_enc_in = self._forward_recon(
                x_raw, attr_raw)

            if in_mask.any():
                self.log(
                    "val/recon_in",
                    self._stft_recon_loss(
                        x_mb[in_mask], y_mb[in_mask],
                        x_cmp[in_mask], y_raw[in_mask]),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            if ood_mask.any():
                loss_ood = self._stft_recon_loss(
                    x_mb[ood_mask], y_mb[ood_mask],
                    x_cmp[ood_mask], y_raw[ood_mask])
                self.log(
                    "val/recon_ood",
                    loss_ood,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                self.log(
                    "val/rms_ood",
                    rms_recon_l1(
                        y_raw[ood_mask], x_cmp[ood_mask], z.shape[-1]),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                if self.in_domain_disc is not None:
                    feat_fake = self.in_domain_disc(y_raw[ood_mask])
                    self.log(
                        "val/disc_ood",
                        self._mean_fake_logit(feat_fake),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

        domains = list(domain) if isinstance(domain, (list, tuple)) else [domain]
        return z.detach(), domains, x_raw.detach(), x_enc_in.detach(), y_raw.detach()
