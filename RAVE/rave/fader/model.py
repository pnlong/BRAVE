"""
FaderRAVE: conditional RAVE with latent discriminator (neurorave FadeRAVE port).

Architecture overview
---------------------
Each training batch is (audio, attr_raw) where attr_raw is (B, D_total, T_lat).

Two derived tensors drive different paths (see _prepare_attributes):

  attr_norm  — float in [-1, 1], concatenated to z for **decoder** control
  attr_cls   — integer class indices for **latent discriminator** CE only

Training phases (inherits RAVE phase_1_duration / warmed_up):

  Phase 1 (not warmed_up):
    - lat_dis_step: train attribute classifier on z (+CE)
    - Generator: fool classifier (-CE on z) + conditional recon
  Phase 2 (warmed_up):
    - Encoder frozen (z.detach); skip latent adversarial entirely
    - Audio GAN + feature matching on recon (standard RAVE phase 2)

Decoder input width = latent_size + D_total (gin DECODER_LATENT_SIZE).

See neurorave: raving_fader/models/fader/faderave.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gin
import numpy as np
import torch
import torch.nn as nn

from ..model import RAVE, _pqmf_decode, _pqmf_encode
from .attributes import (
    attribute_classification_loss,
    compute_bins,
    discrete_index_to_decoder_float,
    load_attribute_stats,
    per_attribute_accuracies,
    per_attribute_ce_losses,
    quantify,
    resolve_attribute_config,
    save_attribute_stats,
    validate_stats_against_config,
)
from .latent_discriminator import LatentDiscriminator


@gin.configurable
class FaderRAVE(RAVE):
    """
    RAVE + Fader Networks conditioning and latent adversarial disentanglement.

    Training batch: (audio, attr_raw) with attr_raw shape (B, D_total, T_lat).
    """

    def __init__(
        self,
        continuous_attributes: Sequence[str] = (),
        discrete_attributes: Sequence[str] = (),
        discrete_num_classes: Optional[Dict[str, int]] = None,
        num_classes: int = 16,
        num_lat_dis_layers: int = 2,
        lambda_inf: float = 0.5,
        lambda_delay: int = 15000,
        n_lat_dis_steps: int = 1,
        rave_mode: bool = False,
        attribute_stats_path: Optional[str] = None,
        **kwargs,
    ):
        # --- Base RAVE (encoder, widened decoder via gin, audio discriminator) ---
        super().__init__(**kwargs)

        # --- Attribute lists from gin: continuous first, discrete appended ---
        cont, disc, names, kinds = resolve_attribute_config(
            continuous_attributes=continuous_attributes,
            discrete_attributes=discrete_attributes,
        )
        self.continuous_attributes = cont
        self.discrete_attributes = disc
        self.attribute_names = names
        self.attribute_kinds = kinds

        self.num_attributes = len(self.attribute_names)
        self.num_classes = num_classes  # quantile bins for continuous latent CE
        self.n_lat_dis_steps = n_lat_dis_steps
        self.rave_mode = rave_mode  # ablation: zero attr_norm, keep attr_cls
        self.lambda_inf = lambda_inf
        self.lambda_delay = lambda_delay
        self.lambda_factor = 0.0  # ramped by LambdaWarmupCallback
        self.attribute_stats_path = attribute_stats_path

        # --- Discrete heads need per-name class counts (default binary=2) ---
        self.discrete_num_classes: Dict[str, int] = dict(discrete_num_classes or {})
        for name in self.discrete_attributes:
            self.discrete_num_classes.setdefault(name, 2)

        # --- Row indices into attr_raw for kind-specific processing ---
        self.continuous_indices = [
            i for i, n in enumerate(self.attribute_names)
            if self.attribute_kinds[n] == "continuous"
        ]
        self.discrete_indices = [
            i for i, n in enumerate(self.attribute_names)
            if self.attribute_kinds[n] == "discrete"
        ]

        # --- Latent disc: nb_bins for continuous rows, num_classes for discrete ---
        self.num_classes_per_attribute = []
        for name in self.attribute_names:
            if self.attribute_kinds[name] == "discrete":
                self.num_classes_per_attribute.append(
                    self.discrete_num_classes.get(name, 2))
            else:
                self.num_classes_per_attribute.append(num_classes)

        # --- Latent attribute classifier operates on pure z (128-D), not cat(z, attr) ---
        self.latent_discriminator = LatentDiscriminator(
            latent_size=self.latent_size,
            num_attributes=self.num_attributes,
            num_classes=num_classes,
            num_classes_per_attribute=self.num_classes_per_attribute,
            num_layers=num_lat_dis_layers,
        )

        # --- Buffers filled by load_attribute_stats_from_file ---
        # --- Quantile bin edges (continuous) from attribute_stats.yaml ---
        max_bins = max(self.num_classes_per_attribute) if self.num_classes_per_attribute else num_classes
        self.register_buffer(
            "bin_values",
            torch.zeros(self.num_attributes, max_bins),
        )
        self.min_max_features: Dict[str, Tuple[float, float]] = {}
        self._attribute_stats_loaded = False

        if "latent_adversarial" not in self.weights:
            self.weights["latent_adversarial"] = 1.0

    def configure_optimizers(self):
        """Three optimizers: generator (enc+dec), audio disc, latent disc."""
        gen_dict, dis_dict = super().configure_optimizers()
        lat_dis_opt = torch.optim.Adam(
            self.latent_discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9),
            weight_decay=1e-4,
        )
        return [gen_dict, dis_dict, {"optimizer": lat_dis_opt}]

    def load_attribute_stats_from_file(self, path: Union[str, Path]) -> None:
        """Load min/max and bin edges from attribute_stats.yaml."""
        stats = load_attribute_stats(path)
        self._apply_stats(stats)

    def _apply_stats(self, stats: Dict) -> None:
        """Copy offline min/max and bin edges into model buffers."""
        # --- Continuous attrs: min/max for decoder normalize; bins for latent CE ---
        self.min_max_features = stats["min_max_features"]
        bv = torch.tensor(stats["bin_values"], dtype=torch.float32)
        self.bin_values.zero_()
        rows = min(bv.shape[0], self.bin_values.shape[0])
        cols = min(bv.shape[1], self.bin_values.shape[1])
        self.bin_values[:rows, :cols].copy_(bv[:rows, :cols])
        # --- Optional discrete class counts from stats file ---
        if stats.get("discrete_num_classes"):
            self.discrete_num_classes.update(stats["discrete_num_classes"])
        self._attribute_stats_loaded = True

    def set_attribute_stats(
        self,
        min_max_features: Dict[str, Tuple[float, float]],
        bin_values: np.ndarray,
        save_path: Optional[str] = None,
        nb_bins: Optional[int] = None,
        latent_length: Optional[int] = None,
    ) -> None:
        """Inject stats computed offline or from precompute_descriptors.py."""
        self.min_max_features = min_max_features
        self.bin_values.zero_()
        bv = torch.tensor(bin_values, dtype=torch.float32)
        rows = min(bv.shape[0], self.bin_values.shape[0])
        cols = min(bv.shape[1], self.bin_values.shape[1])
        self.bin_values[:rows, :cols].copy_(bv[:rows, :cols])
        self._attribute_stats_loaded = True
        if save_path is not None:
            save_attribute_stats(
                save_path,
                attribute_names=self.attribute_names,
                min_max_features=min_max_features,
                bin_values=bin_values,
                nb_bins=nb_bins or self.num_classes,
                latent_length=latent_length or 0,
                sr=self.sr,
                continuous_attributes=self.continuous_attributes,
                discrete_attributes=self.discrete_attributes,
                attribute_kinds=self.attribute_kinds,
                discrete_num_classes=self.discrete_num_classes,
            )

    def decode(self, z: torch.Tensor, attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent; optionally concat normalized attributes first.

        Args:
            z: (B, latent_size, T_lat)
            attr: (B, D_total, T_lat) normalized control (optional)
        """
        batch_size = z.shape[:-2]
        if attr is not None:
            # --- Widened decode: z_c = cat(content_z, attr_norm) → (B, 128+D, T_lat) ---
            z = torch.cat([z, attr], dim=1)
        y = self.decoder(z)
        if self.output_mode == "pqmf":
            y = _pqmf_decode(self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        return y

    def _prepare_attributes(
        self,
        attr_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Raw batch attributes -> (attr_norm, attr_cls).

        Produces two parallel representations from provider output attr_raw:

          attr_norm → decoder concat (continuous: min/max; discrete: index→float)
          attr_cls  → latent discriminator targets (continuous: quantile bins;
                       discrete: native class index, no re-bucketing)

        rave_mode zeros attr_norm only (decoder ablation); attr_cls unchanged.
        """
        b, d, t = attr_raw.shape
        attr_cls = torch.zeros(b, d, t, device=attr_raw.device, dtype=torch.long)
        attr_norm = torch.zeros(b, d, t, device=attr_raw.device, dtype=torch.float32)

        # --- Per-attribute row: branch on kind from attribute_kinds ---
        for i, name in enumerate(self.attribute_names):
            if i >= d:
                break
            if self.attribute_kinds[name] == "continuous":
                # --- Quantile CE targets from raw continuous values ---
                raw_i = attr_raw[:, i:i + 1, :]
                cls_i = quantify(raw_i, self.bin_values[i:i + 1]).long()
                attr_cls[:, i, :] = cls_i[:, 0, :]
                lo, hi = self.min_max_features.get(name, (0.0, 1.0))
                attr_norm[:, i, :] = 2.0 * (
                    (attr_raw[:, i, :] - lo) / (hi - lo + 1e-8) - 0.5)
            else:
                # --- Native discrete indices (no re-bucketing) ---
                idx = attr_raw[:, i, :].long()
                attr_cls[:, i, :] = idx
                n_cls = self.discrete_num_classes.get(name, 2)
                attr_norm[:, i, :] = discrete_index_to_decoder_float(idx, n_cls)

        if self.rave_mode:
            # --- Ablation: decoder sees zero controls; latent disc still gets attr_cls ---
            attr_norm = torch.zeros_like(attr_norm)
        return attr_norm, attr_cls

    def _log_per_attribute_metrics(
        self,
        attr_cls_pred: List[torch.Tensor],
        attr_cls: torch.Tensor,
        prefix: str,
    ) -> None:
        """Log per-attribute CE and accuracy to W&B."""
        ce = per_attribute_ce_losses(
            attr_cls_pred, attr_cls, self.attribute_names)
        acc = per_attribute_accuracies(
            attr_cls_pred, attr_cls, self.attribute_names)
        from ..train_logging import log_train

        for name, val in ce.items():
            log_train(self, f"fader/{prefix}_ce_{name}", val)
        for name, val in acc.items():
            log_train(self, f"fader/{prefix}_acc_{name}", val)

    def lat_dis_step(
        self,
        x_raw: torch.Tensor,
        attr_cls: torch.Tensor,
        lat_dis_opt: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Train latent discriminator (+CE). Phase 1 only; updates lat_dis_opt."""
        # --- Generator frozen: classifier learns to read attributes from z ---
        self.encoder.eval()
        self.decoder.eval()
        self.latent_discriminator.train()

        # --- Fixed z as classifier input (no grad to encoder here) ---
        with torch.no_grad():
            z = self.encode(x_raw, return_mb=False)
            z, _ = self.encoder.reparametrize(z)[:2]

        # --- +CE: improve attribute prediction from content latent ---
        attr_cls_pred = self.latent_discriminator(z)
        latent_dis = attribute_classification_loss(attr_cls_pred, attr_cls)
        self._log_per_attribute_metrics(attr_cls_pred, attr_cls, "latent")

        lat_dis_opt.zero_grad()
        latent_dis.backward()
        lat_dis_opt.step()
        return latent_dis.detach()

    def training_step(self, batch, batch_idx):
        """
        Fader training step: latent adversary + conditional decode.

        Optimizer order: [gen_opt, dis_opt, lat_dis_opt].
        """
        gen_opt, dis_opt, lat_dis_opt = self.optimizers()

        # --- Unpack (audio, attr_raw) from FaderAttributeDataset ---
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x_raw, attr_raw = batch
        else:
            raise ValueError(
                "FaderRAVE expects batches (audio, attr). "
                "Enable wrap_fader_dataset in gin config."
            )

        if not self._attribute_stats_loaded:
            raise RuntimeError(
                "Attribute stats not loaded. Run precompute_descriptors.py or "
                "call load_attribute_stats_from_file before training."
            )

        x_raw = x_raw.to(self.device)
        attr_raw = attr_raw.to(self.device)
        # --- Split raw attrs into decoder float + latent CE integer targets ---
        attr, attr_cls = self._prepare_attributes(attr_raw)

        from ..train_logging import log_train

        if self.rave_mode:
            log_train(self, "fader/rave_mode_active", 1.0)

        # --- Phase 1 pre-step: train latent disc (+CE) before generator ---
        lat_dis_loss_dis = torch.tensor(0.0, device=self.device)
        if not self.warmed_up:
            for _ in range(self.n_lat_dis_steps):
                lat_dis_loss_dis = lat_dis_loss_dis + self.lat_dis_step(
                    x_raw, attr_cls, lat_dis_opt
                ) / self.n_lat_dis_steps

        x_raw.requires_grad = True
        batch_size = x_raw.shape[:-2]
        self.encoder.set_warmed_up(self.warmed_up)
        self.decoder.set_warmed_up(self.warmed_up)

        # --- Latent disc eval during gen forward; encoder train only in phase 1 ---
        self.latent_discriminator.eval()
        if self.warmed_up:
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()

        z, x_multiband = self.encode(x_raw, return_mb=True)
        z, reg = self.encoder.reparametrize(z)[:2]

        # --- Phase 2: stop encoder gradients through z ---
        if self.warmed_up:
            z = z.detach()
            reg = reg.detach()

        lat_dis_loss = torch.tensor(0.0, device=self.device)
        if not self.warmed_up:
            # --- Phase 1: encoder adversary via -CE on z ---
            attr_cls_pred = self.latent_discriminator(z)
            lat_dis_loss = -attribute_classification_loss(attr_cls_pred, attr_cls)
            self._log_per_attribute_metrics(attr_cls_pred, attr_cls, "latent_gen")
        else:
            # --- Phase 2: latent adversarial finished; skip -CE (z detached, no encoder grad) ---
            log_train(self, "fader/latent_adversarial_active", 0.0)

        # --- Conditional decode: cat(content z, normalized attr) → widened decoder ---
        z_cond = torch.cat([z, attr], dim=1)
        y = self.decoder(z_cond)

        # --- PQMF multiband path for distance losses (same as base RAVE) ---
        if self.output_mode == "pqmf":
            y_multiband = y
            y_raw = _pqmf_decode(
                self.pqmf, y, batch_size=batch_size, n_channels=self.n_channels)
        else:
            y_raw = y
            y_multiband = _pqmf_encode(self.pqmf, y)

        y_raw = y_raw[..., :x_raw.shape[-1]]
        y_multiband = y_multiband[..., :x_multiband.shape[-1]]

        if self.valid_signal_crop and self.receptive_field.sum():
            import rave.core
            x_multiband = rave.core.valid_signal_crop(
                x_multiband, *self.receptive_field)
            y_multiband = rave.core.valid_signal_crop(
                y_multiband, *self.receptive_field)

        distances = {}
        # --- Multiband + fullband reconstruction (VAE losses) ---
        multiband_distance = self.multiband_audio_distance(
            x_multiband, y_multiband)
        for k, v in multiband_distance.items():
            distances[f"multiband_{k}"] = (
                self.weights["multiband_audio_distance"] * v)

        fullband_distance = self.audio_distance(x_raw, y_raw)
        for k, v in fullband_distance.items():
            distances[f"fullband_{k}"] = self.weights["audio_distance"] * v

        feature_matching_distance = 0.0

        # --- Phase 2 only: audio discriminator + feature matching ---
        if self.warmed_up:
            xy = torch.cat([x_raw, y_raw], 0)
            features = self.discriminator(xy)
            feature_real, feature_fake = self.split_features(features)

            loss_dis = 0
            loss_adv = 0
            pred_real = 0
            pred_fake = 0

            for scale_real, scale_fake in zip(feature_real, feature_fake):
                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:],
                        scale_fake[self.num_skipped_features:],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = (
                    feature_matching_distance + current_feature_distance)

                _dis, _adv = self.gan_loss(scale_real[-1], scale_fake[-1])
                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()
                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            feature_matching_distance = (
                feature_matching_distance / len(feature_real))
        else:
            pred_real = torch.tensor(0.0, device=self.device)
            pred_fake = torch.tensor(0.0, device=self.device)
            loss_dis = torch.tensor(0.0, device=self.device)
            loss_adv = torch.tensor(0.0, device=self.device)

        loss_gen = {}
        loss_gen.update(distances)

        if reg.item():
            loss_gen["regularization"] = reg * self.beta_factor

        # --- Latent adversarial term: phase 1 only (weighted by lambda_factor) ---
        if not self.warmed_up:
            loss_gen["latent_adversarial"] = (
                self.lambda_factor * self.weights.get("latent_adversarial", 1.0)
                * lat_dis_loss)

        # --- GAN generator terms: phase 2 only ---
        if self.warmed_up:
            loss_gen["feature_matching"] = (
                self.weights["feature_matching"] * feature_matching_distance)
            loss_gen["adversarial"] = self.weights["adversarial"] * loss_adv

        # --- Alternating optimizers: audio disc every N steps in phase 2 ---
        is_gen_step = not (
            not (batch_idx % self.update_discriminator_every) and self.warmed_up)
        if not is_gen_step:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
        else:
            # --- Generator step: recon + (phase1) latent_adv + (phase2) GAN ---
            gen_opt.zero_grad()
            loss_gen_value = 0.0
            for k, v in loss_gen.items():
                loss_gen_value = loss_gen_value + v * self.weights.get(k, 1.0)
            loss_gen_value.backward()
            gen_opt.step()

        from ..train_logging import log_generator_losses, log_train, log_train_dict

        log_generator_losses(self, loss_gen, is_gen_step)

        log_train(self, "beta_factor", self.beta_factor)
        log_train(self, "fader/lambda_factor", self.lambda_factor)
        if not self.warmed_up:
            log_train(self, "fader/latent_dis_loss_dis", lat_dis_loss_dis)
            log_train(self, "fader/latent_adversarial_active", 1.0)

        if self.warmed_up:
            log_train(self, "loss_dis", loss_dis)
            log_train(self, "pred_real", pred_real.mean())
            log_train(self, "pred_fake", pred_fake.mean())

        log_train_dict(self, loss_gen)

    def validation_step(self, batch, batch_idx):
        """Validation with conditional decode when attributes are in the batch."""
        from .. import blocks

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, attr_raw = batch
            if self._attribute_stats_loaded:
                attr_raw = attr_raw.to(x.device)
                # --- Match training: conditional decode with attr_norm ---
                attr, _ = self._prepare_attributes(attr_raw)
            else:
                attr = None
        else:
            x = batch
            attr = None

        z = self.encode(x)
        if isinstance(self.encoder, blocks.VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z = self.encoder.reparametrize(z)[0]
        y = self.decode(z, attr=attr)

        distance = self.audio_distance(x, y)
        full_distance = sum(distance.values())

        if self.trainer is not None:
            from ..train_logging import log_val

            log_val(self, "loss", full_distance)

        return torch.cat([x, y], -1), mean
