"""Bidirectional CycleGAN trainer on dual frozen BRAVE backbones."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import gin
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..model import _pqmf_decode
from .dataset import DOMAIN_IN, DOMAIN_OOD
from .gan_utils import audio_gan_d, audio_gan_g, feature_matching_loss, mean_fake_logit
from .in_domain_discriminator import InDomainAudioDiscriminator
from .losses import (
    empirical_adversarial_loss_scale,
    empirical_loss_scale,
    latent_per_channel_variance,
    normalize_loss,
    resolve_gan_loss,
    resolve_latent_spread_loss,
    rms_recon_l1,
    weighted_recon_loss,
)


@gin.configurable
class CycleGANTrainer(pl.LightningModule):
    """
    Full CycleGAN between domain X (tap) and domain Y (water) on frozen BRAVE.

    Supports latent or waveform warps for G_{X→Y} and G_{Y→X}.

    Cycle consistency can be measured in waveform space (STFT+RMS), latent space
    (``z ≈ W_rev(W_fwd(z))``, no re-encode), or both — see ``use_waveform_cycle`` /
    ``use_latent_cycle``.
    """

    automatic_optimization = False

    def __init__(
        self,
        backbone_x: nn.Module,
        backbone_y: nn.Module,
        warp_xy: nn.Module,
        warp_yx: nn.Module,
        canonicalizer_type: str,
        disc_x: Optional[InDomainAudioDiscriminator] = None,
        disc_y: Optional[InDomainAudioDiscriminator] = None,
        lambda_cycle: float = 10.0,
        lambda_gan: float = 1.0,
        lambda_feature_matching: float = 0.5,
        lambda_identity: float = 0.0,
        cycle_stft_weight: float = 0.9,
        cycle_rms_weight: float = 0.1,
        stft_loss_scale: float = 45.0,
        rms_loss_scale: float = 0.3,
        gan_loss_scale: float = 0.5,
        fm_loss_scale: float = 0.25,
        calibrate_loss_scales: bool = True,
        calibration_batches: int = 16,
        loss_scale_min: float = 1e-3,
        cycle_mode: str = "both",
        use_waveform_cycle: bool = True,
        use_latent_cycle: bool = False,
        lambda_latent_cycle: float = 10.0,
        latent_cycle_loss_scale: float = 1.0,
        gan_loss: str = "hinge",
        lr: float = 3e-4,
        disc_lr: float = 2e-5,
        cycle_warmup_duration: int = 50000,
        gan_ramp_duration: int = 5000,
        update_discriminator_every: int = 2,
        num_skipped_features: int = 1,
        encode_use_mean: bool = True,
        lambda_latent_spread: float = 0.25,
        latent_spread_mode: str = "var_match",
        latent_spread_scale: float = 1.0,
        latent_spread_use_ema_ref: bool = True,
        latent_spread_ema_decay: float = 0.99,
        unfreeze_encoders: bool = False,
        unfreeze_decoders: bool = False,
        backbone_lr: float = 1e-5,
    ) -> None:
        super().__init__()
        if canonicalizer_type not in ("waveform", "latent"):
            raise ValueError("canonicalizer_type must be waveform or latent")
        if cycle_mode not in ("stft", "rms", "both"):
            raise ValueError("cycle_mode must be stft, rms, or both")
        if use_latent_cycle and canonicalizer_type != "latent":
            raise ValueError(
                "use_latent_cycle=True requires canonicalizer_type='latent' "
                "(warps must map Z_X ↔ Z_Y)"
            )
        if not use_waveform_cycle and not use_latent_cycle:
            raise ValueError(
                "at least one of use_waveform_cycle / use_latent_cycle must be True"
            )

        self.backbone_x = backbone_x
        self.backbone_y = backbone_y
        self.warp_xy = warp_xy
        self.warp_yx = warp_yx
        self.canonicalizer_type = canonicalizer_type

        for disc, name in ((disc_x, "disc_x"), (disc_y, "disc_y")):
            if disc is not None and isinstance(disc, type):
                disc = disc(n_channels=backbone_x.n_channels)
            setattr(self, name, disc)

        self.lambda_cycle = lambda_cycle
        self.lambda_gan = lambda_gan
        self.lambda_feature_matching = lambda_feature_matching
        self.lambda_identity = lambda_identity
        self.cycle_stft_weight = cycle_stft_weight
        self.cycle_rms_weight = cycle_rms_weight
        self.stft_loss_scale = stft_loss_scale
        self.rms_loss_scale = rms_loss_scale
        self.gan_loss_scale = gan_loss_scale
        self.fm_loss_scale = fm_loss_scale
        self.calibrate_loss_scales = calibrate_loss_scales
        self.calibration_batches = calibration_batches
        self.loss_scale_min = loss_scale_min
        self.loss_scales_calibrated = False
        self.cycle_mode = cycle_mode
        self.use_waveform_cycle = use_waveform_cycle
        self.use_latent_cycle = use_latent_cycle
        self.lambda_latent_cycle = lambda_latent_cycle
        self.latent_cycle_loss_scale = latent_cycle_loss_scale
        self.gan_loss_fn = resolve_gan_loss(gan_loss)
        self.lr = lr
        self.disc_lr = disc_lr
        self.backbone_lr = backbone_lr
        self.cycle_warmup_duration = cycle_warmup_duration
        self.gan_ramp_duration = gan_ramp_duration
        self.gan_factor = 0.0
        self.cycle_warmed_up = False
        self.update_discriminator_every = update_discriminator_every
        self.num_skipped_features = num_skipped_features
        self.encode_use_mean = encode_use_mean
        self.lambda_latent_spread = lambda_latent_spread
        if latent_spread_mode not in ("var_match", "var_floor"):
            raise ValueError("latent_spread_mode must be var_match or var_floor")
        self.latent_spread_fn = resolve_latent_spread_loss(latent_spread_mode)
        self.latent_spread_scale = latent_spread_scale
        self.latent_spread_use_ema_ref = latent_spread_use_ema_ref
        self.latent_spread_ema_decay = latent_spread_ema_decay
        self.spread_factor = 0.0
        self.latent_var_ref_x: Optional[torch.Tensor] = None
        self.latent_var_ref_y: Optional[torch.Tensor] = None
        self.spread_active = (
            canonicalizer_type == "latent" and self.lambda_latent_spread > 0.0
        )
        self.unfreeze_encoders = unfreeze_encoders
        self.unfreeze_decoders = unfreeze_decoders

        for backbone in (self.backbone_x, self.backbone_y):
            for p in backbone.parameters():
                p.requires_grad = False
            if self.unfreeze_encoders:
                for p in backbone.encoder.parameters():
                    p.requires_grad = True
            if self.unfreeze_decoders:
                for p in backbone.decoder.parameters():
                    p.requires_grad = True

    @property
    def warmup(self) -> int:
        """Alias for cycle warmup (used by ramp callback)."""
        return self.cycle_warmup_duration

    def configure_optimizers(self):
        warp_groups = [
            {
                "params": list(self.warp_xy.parameters()) + list(self.warp_yx.parameters()),
                "lr": self.lr,
            }
        ]
        backbone_params = []
        for backbone in (self.backbone_x, self.backbone_y):
            if self.unfreeze_encoders:
                backbone_params.extend(
                    p for p in backbone.encoder.parameters() if p.requires_grad)
            if self.unfreeze_decoders:
                backbone_params.extend(
                    p for p in backbone.decoder.parameters() if p.requires_grad)
        if backbone_params:
            warp_groups.append({"params": backbone_params, "lr": self.backbone_lr})

        warp_opt = torch.optim.Adam(warp_groups, betas=(0.5, 0.9))
        disc_params = []
        if self.disc_x is not None:
            disc_params.extend(self.disc_x.parameters())
        if self.disc_y is not None:
            disc_params.extend(self.disc_y.parameters())
        if not disc_params:
            return warp_opt
        disc_opt = torch.optim.Adam(disc_params, lr=self.disc_lr, betas=(0.5, 0.9))
        return [warp_opt, disc_opt]

    def _optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]:
        opts = self.optimizers()
        if self.disc_x is None and self.disc_y is None:
            return opts, None
        warp_opt, disc_opt = opts
        return warp_opt, disc_opt

    def _set_backbone_train_mode(self) -> None:
        for backbone in (self.backbone_x, self.backbone_y):
            backbone.eval()
            if self.unfreeze_encoders:
                backbone.encoder.train()
            if self.unfreeze_decoders:
                backbone.decoder.train()
        if self.disc_x is not None:
            self.disc_x.train()
        if self.disc_y is not None:
            self.disc_y.train()

    def _encode_latent(
        self,
        backbone: nn.Module,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_raw, x_multiband = backbone.encode(x, return_mb=True)
        if self.encode_use_mean:
            from .. import blocks

            if isinstance(backbone.encoder, blocks.VariationalEncoder):
                z = z_raw.chunk(2, dim=1)[0]
            else:
                z, _ = backbone.encoder.reparametrize(z_raw)[:2]
        else:
            z, _ = backbone.encoder.reparametrize(z_raw)[:2]
        return z, x_multiband

    def _decode(
        self,
        backbone: nn.Module,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # _pqmf_decode expects an iterable of leading dims (e.g. (B,)), not an int.
        batch_size = z.shape[:-2]
        y_multiband = backbone.decoder(z)
        y_raw = y_multiband
        if backbone.output_mode == "pqmf":
            y_raw = _pqmf_decode(
                backbone.pqmf,
                y_multiband,
                batch_size=batch_size,
                n_channels=backbone.n_channels,
            )
        return y_raw, y_multiband

    def _apply_warp(self, warp: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        return warp(tensor)

    def _forward_g_xy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """G_{X→Y}(x) → waveform, multiband, latent (post-warp)."""
        if self.canonicalizer_type == "waveform":
            x_warp = self._apply_warp(self.warp_xy, x)
            z, x_mb = self._encode_latent(self.backbone_y, x_warp)
        else:
            z_src, x_mb = self._encode_latent(self.backbone_x, x)
            z = self._apply_warp(self.warp_xy, z_src)
        y_raw, y_mb = self._decode(self.backbone_y, z)
        return self._align_waveforms(x, y_raw), y_mb, z

    def _forward_g_yx(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """G_{Y→X}(y) → waveform, multiband, latent (post-warp)."""
        if self.canonicalizer_type == "waveform":
            y_warp = self._apply_warp(self.warp_yx, y)
            z, y_mb = self._encode_latent(self.backbone_x, y_warp)
        else:
            z_src, y_mb = self._encode_latent(self.backbone_y, y)
            z = self._apply_warp(self.warp_yx, z_src)
        x_raw, x_mb = self._decode(self.backbone_x, z)
        return self._align_waveforms(y, x_raw), x_mb, z

    @staticmethod
    def _align_waveforms(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        t = min(ref.shape[-1], other.shape[-1])
        return other[..., :t]

    def _stft_recon_loss(
        self,
        backbone: nn.Module,
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        x_cmp: torch.Tensor,
        y_raw: torch.Tensor,
    ) -> torch.Tensor:
        t_mb = min(x_mb.shape[-1], y_mb.shape[-1])
        t_raw = min(x_cmp.shape[-1], y_raw.shape[-1])
        mb_dist = backbone.multiband_audio_distance(
            x_mb[..., :t_mb], y_mb[..., :t_mb])
        fb_dist = backbone.audio_distance(
            x_cmp[..., :t_raw], y_raw[..., :t_raw])
        return sum(mb_dist.values()) + sum(fb_dist.values())

    def _cycle_loss(
        self,
        backbone: nn.Module,
        target_raw: torch.Tensor,
        recon_raw: torch.Tensor,
        target_mb: torch.Tensor,
        recon_mb: torch.Tensor,
        n_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=target_raw.device)
        loss_stft = zero
        loss_rms = zero
        if self.cycle_mode in ("stft", "both"):
            loss_stft = self._stft_recon_loss(
                backbone, target_mb, recon_mb, target_raw, recon_raw)
        if self.cycle_mode in ("rms", "both"):
            loss_rms = rms_recon_l1(recon_raw, target_raw, n_frames)
        return loss_stft, loss_rms

    def _weighted_cycle(
        self,
        stft: torch.Tensor,
        rms: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_recon_loss(
            stft,
            rms,
            stft_weight=self.cycle_stft_weight,
            rms_weight=self.cycle_rms_weight,
            stft_scale=self.stft_loss_scale,
            rms_scale=self.rms_loss_scale,
        )

    def _disc_features(
        self,
        disc: InDomainAudioDiscriminator,
        real: torch.Tensor,
        fake: torch.Tensor,
        *,
        detach: bool,
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        if detach:
            real = real.detach()
            fake = fake.detach()
        return disc(real), disc(fake)

    def _parse_batch(self, batch):
        if len(batch) == 3:
            x_raw, _attr_raw, domain = batch
        else:
            raise ValueError("batch must be (audio, attr|None, domain)")
        return x_raw, domain

    def _domain_masks(self, domain) -> Tuple[torch.Tensor, torch.Tensor]:
        is_y = [d == DOMAIN_IN for d in domain]
        y_mask = torch.tensor(is_y, device=self.device)
        return y_mask, ~y_mask

    def _forward_batch(
        self,
        x_raw: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero_idx = torch.zeros(0, dtype=torch.long, device=self.device)
        outputs: Dict[str, torch.Tensor] = {}

        if x_mask.any():
            x_samples = x_raw[x_mask]
            z_x, x_mb = self._encode_latent(self.backbone_x, x_samples)
            y_fake, y_fake_mb, z_xy = self._forward_g_xy(x_samples)
            x_cycle, x_cycle_mb, _ = self._forward_g_yx(y_fake)
            z_x_cycle = (
                self._apply_warp(
                    self.warp_yx, self._apply_warp(self.warp_xy, z_x))
                if self.use_latent_cycle else z_x[:0]
            )
            outputs.update({
                "x": x_samples,
                "x_mb": x_mb,
                "y_fake": y_fake,
                "y_fake_mb": y_fake_mb,
                "x_cycle": x_cycle,
                "x_cycle_mb": x_cycle_mb,
                "z_x": z_x,
                "z_xy": z_xy,
                "z_x_cycle": z_x_cycle,
                "x_idx": torch.where(x_mask)[0],
            })
        else:
            empty = x_raw[:0]
            outputs.update({
                "x": empty,
                "x_mb": empty,
                "y_fake": empty,
                "y_fake_mb": empty,
                "x_cycle": empty,
                "x_cycle_mb": empty,
                "z_x": empty,
                "z_xy": empty,
                "z_x_cycle": empty,
                "x_idx": zero_idx,
            })

        if y_mask.any():
            y_samples = x_raw[y_mask]
            z_y, y_mb = self._encode_latent(self.backbone_y, y_samples)
            x_fake, x_fake_mb, z_yx = self._forward_g_yx(y_samples)
            y_cycle, y_cycle_mb, _ = self._forward_g_xy(x_fake)
            z_y_cycle = (
                self._apply_warp(
                    self.warp_xy, self._apply_warp(self.warp_yx, z_y))
                if self.use_latent_cycle else z_y[:0]
            )
            outputs.update({
                "y": y_samples,
                "y_mb": y_mb,
                "x_fake": x_fake,
                "x_fake_mb": x_fake_mb,
                "y_cycle": y_cycle,
                "y_cycle_mb": y_cycle_mb,
                "z_y": z_y,
                "z_yx": z_yx,
                "z_y_cycle": z_y_cycle,
                "y_idx": torch.where(y_mask)[0],
            })
        else:
            empty = x_raw[:0]
            outputs.update({
                "y": empty,
                "y_mb": empty,
                "x_fake": empty,
                "x_fake_mb": empty,
                "y_cycle": empty,
                "y_cycle_mb": empty,
                "z_y": empty,
                "z_yx": empty,
                "z_y_cycle": empty,
                "y_idx": zero_idx,
            })

        return outputs

    def _latent_cycle_loss(
        self,
        z: torch.Tensor,
        z_cycle: torch.Tensor,
    ) -> torch.Tensor:
        """L1 cycle in latent space: ``‖z − W_rev(W_fwd(z))‖₁`` (no re-encode)."""
        if z.numel() == 0 or z_cycle.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        t = min(z.shape[-1], z_cycle.shape[-1])
        return torch.mean(torch.abs(z[..., :t] - z_cycle[..., :t]))

    def _waveform_cycle_bundle(
        self,
        backbone: nn.Module,
        target_raw: torch.Tensor,
        recon_raw: torch.Tensor,
        target_mb: torch.Tensor,
        recon_mb: torch.Tensor,
        n_frames: int,
    ) -> torch.Tensor:
        if not self.use_waveform_cycle:
            return torch.tensor(0.0, device=self.device)
        stft, rms = self._cycle_loss(
            backbone, target_raw, recon_raw, target_mb, recon_mb, n_frames)
        return self._weighted_cycle(stft, rms)

    def _identity_losses(
        self,
        fwd: Dict[str, torch.Tensor],
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.tensor(0.0, device=self.device)
        if self.lambda_identity <= 0.0:
            return zero

        loss = zero
        if y_mask.any():
            y_id, y_id_mb, _ = self._forward_g_xy(fwd["y"])
            stft, rms = self._cycle_loss(
                self.backbone_y,
                fwd["y"],
                y_id,
                fwd["y_mb"],
                y_id_mb,
                fwd["z_yx"].shape[-1] if fwd["z_yx"].numel() else 1,
            )
            loss = loss + self._weighted_cycle(stft, rms)
        if x_mask.any():
            x_id, x_id_mb, _ = self._forward_g_yx(fwd["x"])
            stft, rms = self._cycle_loss(
                self.backbone_x,
                fwd["x"],
                x_id,
                fwd["x_mb"],
                x_id_mb,
                fwd["z_xy"].shape[-1] if fwd["z_xy"].numel() else 1,
            )
            loss = loss + self._weighted_cycle(stft, rms)
        return loss

    def _update_latent_var_refs(
        self,
        x_raw: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> None:
        if not self.spread_active:
            return
        with torch.no_grad():
            if x_mask.any():
                z_x, _ = self._encode_latent(self.backbone_x, x_raw[x_mask])
                batch_var = latent_per_channel_variance(z_x)
                if self.latent_var_ref_x is None or not self.latent_spread_use_ema_ref:
                    self.latent_var_ref_x = batch_var.detach()
                else:
                    d = self.latent_spread_ema_decay
                    self.latent_var_ref_x = d * self.latent_var_ref_x + (1 - d) * batch_var
            if y_mask.any():
                z_y, _ = self._encode_latent(self.backbone_y, x_raw[y_mask])
                batch_var = latent_per_channel_variance(z_y)
                if self.latent_var_ref_y is None or not self.latent_spread_use_ema_ref:
                    self.latent_var_ref_y = batch_var.detach()
                else:
                    d = self.latent_spread_ema_decay
                    self.latent_var_ref_y = d * self.latent_var_ref_y + (1 - d) * batch_var

    def _latent_spread_loss(self, fwd: Dict[str, torch.Tensor]) -> torch.Tensor:
        zero = torch.tensor(0.0, device=self.device)
        if not self.spread_active or self.spread_factor <= 0.0:
            return zero
        loss = zero
        if fwd["z_xy"].numel() and self.latent_var_ref_y is not None:
            loss = loss + self.latent_spread_fn(fwd["z_xy"], self.latent_var_ref_y)
        if fwd["z_yx"].numel() and self.latent_var_ref_x is not None:
            loss = loss + self.latent_spread_fn(fwd["z_yx"], self.latent_var_ref_x)
        return loss

    @torch.no_grad()
    def _batch_raw_losses(
        self,
        batch,
        *,
        include_adversarial: bool = False,
    ) -> Dict[str, Optional[float]]:
        x_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        y_mask, x_mask = self._domain_masks(domain)
        fwd = self._forward_batch(x_raw, x_mask, y_mask)

        out: Dict[str, Optional[float]] = {
            "cycle_x": None,
            "cycle_y": None,
            "latent_cycle_x": None,
            "latent_cycle_y": None,
            "gan_y": None,
            "gan_x": None,
            "fm_y": None,
            "fm_x": None,
            "spread": None,
        }

        if x_mask.any():
            if self.use_waveform_cycle:
                stft, rms = self._cycle_loss(
                    self.backbone_x,
                    fwd["x"],
                    fwd["x_cycle"],
                    fwd["x_mb"],
                    fwd["x_cycle_mb"],
                    fwd["z_xy"].shape[-1],
                )
                out["cycle_x"] = float((stft + rms).detach().cpu())
            if self.use_latent_cycle and fwd["z_x_cycle"].numel():
                out["latent_cycle_x"] = float(
                    self._latent_cycle_loss(fwd["z_x"], fwd["z_x_cycle"]).detach().cpu())

        if y_mask.any():
            if self.use_waveform_cycle:
                stft, rms = self._cycle_loss(
                    self.backbone_y,
                    fwd["y"],
                    fwd["y_cycle"],
                    fwd["y_mb"],
                    fwd["y_cycle_mb"],
                    fwd["z_yx"].shape[-1],
                )
                out["cycle_y"] = float((stft + rms).detach().cpu())
            if self.use_latent_cycle and fwd["z_y_cycle"].numel():
                out["latent_cycle_y"] = float(
                    self._latent_cycle_loss(fwd["z_y"], fwd["z_y_cycle"]).detach().cpu())

        if include_adversarial and self.disc_y is not None and x_mask.any() and y_mask.any():
            feat_real_y, feat_fake_y = self._disc_features(
                self.disc_y, fwd["y"], fwd["y_fake"], detach=True)
            feat_real_x, feat_fake_x = self._disc_features(
                self.disc_x, fwd["x"], fwd["x_fake"], detach=True)
            out["gan_y"] = float(audio_gan_g(feat_fake_y, self.gan_loss_fn).detach().cpu())
            out["gan_x"] = float(audio_gan_g(feat_fake_x, self.gan_loss_fn).detach().cpu())
            out["fm_y"] = float(feature_matching_loss(
                feat_real_y, feat_fake_y,
                num_skipped_features=self.num_skipped_features).detach().cpu())
            out["fm_x"] = float(feature_matching_loss(
                feat_real_x, feat_fake_x,
                num_skipped_features=self.num_skipped_features).detach().cpu())

        if self.spread_active and fwd["z_xy"].numel() and self.latent_var_ref_y is not None:
            out["spread"] = float(
                self.latent_spread_fn(fwd["z_xy"], self.latent_var_ref_y).detach().cpu())

        return out

    @torch.no_grad()
    def calibrate_loss_scales_from_loader(
        self,
        dataloader: Iterable,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        if max_batches is None:
            max_batches = self.calibration_batches
        max_batches = max(1, max_batches)

        buckets: Dict[str, List[float]] = {
            "cycle_x": [],
            "cycle_y": [],
            "latent_cycle_x": [],
            "latent_cycle_y": [],
            "gan_y": [],
            "gan_x": [],
            "fm_y": [],
            "fm_x": [],
            "spread": [],
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
            "latent_spread_scale": self.latent_spread_scale,
            "latent_cycle_loss_scale": self.latent_cycle_loss_scale,
        }

        cycle_vals = buckets["cycle_x"] + buckets["cycle_y"]
        if cycle_vals:
            mean_cycle = sum(cycle_vals) / len(cycle_vals)
            scales["stft_loss_scale"] = empirical_loss_scale([mean_cycle], self.loss_scale_min)
            self.stft_loss_scale = scales["stft_loss_scale"]

        lat_cycle_vals = buckets["latent_cycle_x"] + buckets["latent_cycle_y"]
        if lat_cycle_vals:
            scales["latent_cycle_loss_scale"] = empirical_loss_scale(
                lat_cycle_vals, self.loss_scale_min)
            self.latent_cycle_loss_scale = scales["latent_cycle_loss_scale"]

        gan_vals = [v for v in buckets["gan_y"] + buckets["gan_x"] if v is not None]
        if gan_vals:
            scales["gan_loss_scale"] = empirical_adversarial_loss_scale(
                gan_vals, self.gan_loss_scale, self.loss_scale_min)
            self.gan_loss_scale = scales["gan_loss_scale"]

        fm_vals = buckets["fm_y"] + buckets["fm_x"]
        if fm_vals:
            scales["fm_loss_scale"] = empirical_adversarial_loss_scale(
                fm_vals, self.fm_loss_scale, self.loss_scale_min)
            self.fm_loss_scale = scales["fm_loss_scale"]

        if buckets["spread"]:
            scales["latent_spread_scale"] = empirical_loss_scale(
                buckets["spread"], self.loss_scale_min)
            self.latent_spread_scale = scales["latent_spread_scale"]

        self.loss_scales_calibrated = True
        return scales

    def _log_train_metrics(
        self,
        *,
        loss: torch.Tensor,
        loss_cycle: torch.Tensor,
        loss_cycle_x: torch.Tensor,
        loss_cycle_y: torch.Tensor,
        loss_latent_cycle: torch.Tensor,
        loss_adv: torch.Tensor,
        loss_fm: torch.Tensor,
        loss_d: torch.Tensor,
        loss_spread: torch.Tensor,
        batch_size: int,
        log_disc: bool = False,
    ) -> None:
        self.log("cycle/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("cycle/cycle_norm", loss_cycle, batch_size=batch_size)
        self.log("cycle/cycle_x_norm", loss_cycle_x, batch_size=batch_size)
        self.log("cycle/cycle_y_norm", loss_cycle_y, batch_size=batch_size)
        self.log("cycle/gan_factor", float(self.gan_factor), batch_size=batch_size)
        self.log("cycle/cycle_warmed_up", float(self.cycle_warmed_up), batch_size=batch_size)
        if self.use_latent_cycle:
            self.log("cycle/latent_cycle_norm", loss_latent_cycle, batch_size=batch_size)
        if self.gan_factor > 0.0:
            self.log("cycle/adv_norm", loss_adv, batch_size=batch_size)
            self.log("cycle/fm_norm", loss_fm, batch_size=batch_size)
        if log_disc:
            self.log("cycle/disc_loss", loss_d, batch_size=batch_size)
        if self.spread_active and loss_spread.requires_grad:
            self.log("cycle/spread_norm", loss_spread, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        warp_opt, disc_opt = self._optimizers()
        self._set_backbone_train_mode()

        x_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        batch_size = x_raw.size(0)
        y_mask, x_mask = self._domain_masks(domain)

        self._update_latent_var_refs(x_raw, x_mask, y_mask)
        fwd = self._forward_batch(x_raw, x_mask, y_mask)

        zero = torch.tensor(0.0, device=self.device)
        loss_cycle_x = zero
        loss_cycle_y = zero
        loss_latent_cycle = zero

        if x_mask.any():
            loss_cycle_x = self._waveform_cycle_bundle(
                self.backbone_x,
                fwd["x"],
                fwd["x_cycle"],
                fwd["x_mb"],
                fwd["x_cycle_mb"],
                fwd["z_xy"].shape[-1],
            )
            if self.use_latent_cycle:
                loss_latent_cycle = loss_latent_cycle + normalize_loss(
                    self._latent_cycle_loss(fwd["z_x"], fwd["z_x_cycle"]),
                    self.latent_cycle_loss_scale,
                )

        if y_mask.any():
            loss_cycle_y = self._waveform_cycle_bundle(
                self.backbone_y,
                fwd["y"],
                fwd["y_cycle"],
                fwd["y_mb"],
                fwd["y_cycle_mb"],
                fwd["z_yx"].shape[-1],
            )
            if self.use_latent_cycle:
                loss_latent_cycle = loss_latent_cycle + normalize_loss(
                    self._latent_cycle_loss(fwd["z_y"], fwd["z_y_cycle"]),
                    self.latent_cycle_loss_scale,
                )

        loss_cycle = loss_cycle_x + loss_cycle_y
        loss_adv = zero
        loss_fm = zero
        loss_d = zero
        loss_spread = zero

        has_both = x_mask.any() and y_mask.any()
        gan_active = self.gan_factor > 0.0 and has_both
        is_disc_step = (
            gan_active
            and disc_opt is not None
            and not (batch_idx % self.update_discriminator_every)
        )

        if is_disc_step:
            feat_real_y, feat_fake_y = self._disc_features(
                self.disc_y, fwd["y"], fwd["y_fake"], detach=True)
            feat_real_x, feat_fake_x = self._disc_features(
                self.disc_x, fwd["x"], fwd["x_fake"], detach=True)
            loss_d_y = audio_gan_d(feat_real_y, feat_fake_y, self.gan_loss_fn)
            loss_d_x = audio_gan_d(feat_real_x, feat_fake_x, self.gan_loss_fn)
            loss_d = loss_d_y + loss_d_x
            disc_opt.zero_grad()
            self.manual_backward(loss_d)
            disc_opt.step()
            self._log_train_metrics(
                loss=loss_d,
                loss_cycle=loss_cycle,
                loss_cycle_x=loss_cycle_x,
                loss_cycle_y=loss_cycle_y,
                loss_latent_cycle=loss_latent_cycle,
                loss_adv=loss_adv,
                loss_fm=loss_fm,
                loss_d=loss_d,
                loss_spread=loss_spread,
                batch_size=batch_size,
                log_disc=True,
            )
            return loss_d

        if gan_active:
            feat_real_y, feat_fake_y = self._disc_features(
                self.disc_y, fwd["y"], fwd["y_fake"], detach=False)
            feat_real_x, feat_fake_x = self._disc_features(
                self.disc_x, fwd["x"], fwd["x_fake"], detach=False)
            loss_gan_y = audio_gan_g(feat_fake_y, self.gan_loss_fn)
            loss_gan_x = audio_gan_g(feat_fake_x, self.gan_loss_fn)
            loss_fm_y = feature_matching_loss(
                feat_real_y, feat_fake_y,
                num_skipped_features=self.num_skipped_features)
            loss_fm_x = feature_matching_loss(
                feat_real_x, feat_fake_x,
                num_skipped_features=self.num_skipped_features)
            loss_adv = normalize_loss(
                loss_gan_y + loss_gan_x, self.gan_loss_scale)
            loss_fm = normalize_loss(loss_fm_y + loss_fm_x, self.fm_loss_scale)

        if self.spread_active and self.spread_factor > 0.0:
            loss_spread_raw = self._latent_spread_loss(fwd)
            loss_spread = normalize_loss(loss_spread_raw, self.latent_spread_scale)

        loss_identity = self._identity_losses(fwd, x_mask, y_mask)

        loss = (
            self.lambda_cycle * loss_cycle
            + self.lambda_latent_cycle * loss_latent_cycle
            + self.gan_factor * self.lambda_gan * loss_adv
            + self.gan_factor * self.lambda_feature_matching * loss_fm
            + self.lambda_identity * loss_identity
            + self.spread_factor * self.lambda_latent_spread * loss_spread
        )

        warp_opt.zero_grad()
        if loss.requires_grad:
            self.manual_backward(loss)
            warp_opt.step()

        self._log_train_metrics(
            loss=loss,
            loss_cycle=loss_cycle,
            loss_cycle_x=loss_cycle_x,
            loss_cycle_y=loss_cycle_y,
            loss_latent_cycle=loss_latent_cycle,
            loss_adv=loss_adv,
            loss_fm=loss_fm,
            loss_d=loss_d,
            loss_spread=loss_spread,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.backbone_x.eval()
        self.backbone_y.eval()
        x_raw, domain = self._parse_batch(batch)
        x_raw = x_raw.to(self.device)
        batch_size = x_raw.size(0)
        y_mask, x_mask = self._domain_masks(domain)

        with torch.no_grad():
            fwd = self._forward_batch(x_raw, x_mask, y_mask)

            if x_mask.any():
                if self.use_waveform_cycle:
                    self.log(
                        "val/cycle_x",
                        self._waveform_cycle_bundle(
                            self.backbone_x,
                            fwd["x"],
                            fwd["x_cycle"],
                            fwd["x_mb"],
                            fwd["x_cycle_mb"],
                            fwd["z_xy"].shape[-1],
                        ),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                if self.use_latent_cycle and fwd["z_x_cycle"].numel():
                    self.log(
                        "val/latent_cycle_x",
                        normalize_loss(
                            self._latent_cycle_loss(fwd["z_x"], fwd["z_x_cycle"]),
                            self.latent_cycle_loss_scale,
                        ),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                if self.disc_y is not None and fwd["y_fake"].shape[0] > 0:
                    feat_fake = self.disc_y(fwd["y_fake"])
                    self.log(
                        "val/disc_y_fake",
                        mean_fake_logit(feat_fake),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

            if y_mask.any():
                if self.use_waveform_cycle:
                    self.log(
                        "val/cycle_y",
                        self._waveform_cycle_bundle(
                            self.backbone_y,
                            fwd["y"],
                            fwd["y_cycle"],
                            fwd["y_mb"],
                            fwd["y_cycle_mb"],
                            fwd["z_yx"].shape[-1],
                        ),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                if self.use_latent_cycle and fwd["z_y_cycle"].numel():
                    self.log(
                        "val/latent_cycle_y",
                        normalize_loss(
                            self._latent_cycle_loss(fwd["z_y"], fwd["z_y_cycle"]),
                            self.latent_cycle_loss_scale,
                        ),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                if self.disc_x is not None and fwd["x_fake"].shape[0] > 0:
                    feat_fake = self.disc_x(fwd["x_fake"])
                    self.log(
                        "val/disc_x_fake",
                        mean_fake_logit(feat_fake),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

        domains = list(domain) if isinstance(domain, (list, tuple)) else [domain]
        return {
            "domains": domains,
            "x_raw": x_raw.detach(),
            "fwd": {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in fwd.items()},
        }
