"""Unit tests for CycleGAN cycle_domain / latent D / warp init."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from rave.canonicalizer.cycle_trainer import CycleGANTrainer
from rave.canonicalizer.gan_utils import audio_gan_d, audio_gan_g, feature_matching_loss
from rave.canonicalizer.in_domain_discriminator import InDomainLatentDiscriminator
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer
from rave.core import hinge_gan


class _DummyEncoder(nn.Module):
    def reparametrize(self, z_raw):
        return z_raw, z_raw, torch.zeros(1, device=z_raw.device)


class _FakeBackbone(nn.Module):
    def __init__(self, latent_size=8, n_channels=1):
        super().__init__()
        self.latent_size = latent_size
        self.n_channels = n_channels
        self.sr = 44100
        self.output_mode = "raw"
        self.encoder = _DummyEncoder()
        self.decoder = nn.Conv1d(latent_size, n_channels, 1)
        self._dummy = nn.Parameter(torch.zeros(1))

    def encode(self, x, return_mb=True):
        t_lat = max(1, x.shape[-1] // 8)
        z = torch.randn(x.shape[0], self.latent_size, t_lat, device=x.device)
        return z, x


def _tiny_trainer(**kwargs):
    bb_x = _FakeBackbone()
    bb_y = _FakeBackbone()
    defaults = dict(
        backbone_x=bb_x,
        backbone_y=bb_y,
        warp_xy=LatentCanonicalizer(latent_size=8, init_mode="random"),
        warp_yx=LatentCanonicalizer(latent_size=8, init_mode="random"),
        canonicalizer_type="latent",
        disc_x=InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2),
        disc_y=InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2),
        calibrate_loss_scales=False,
        lambda_identity=0.0,
        lambda_latent_spread=0.0,
        cycle_warmup_duration=10,
        gan_ramp_duration=0,
    )
    defaults.update(kwargs)
    return CycleGANTrainer(**defaults)


def test_latent_discriminator_shapes_and_gan_utils():
    disc = InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2)
    z = torch.randn(3, 8, 12)
    feats = disc(z)
    assert len(feats) == 1
    assert feats[0][-1].shape[0] == 3
    fake = disc(torch.randn(3, 8, 12))
    loss_d = audio_gan_d(feats, fake, hinge_gan)
    loss_g = audio_gan_g(fake, hinge_gan)
    loss_fm = feature_matching_loss(feats, fake, num_skipped_features=0)
    assert loss_d.ndim == 0 and loss_g.ndim == 0 and loss_fm.ndim == 0


def test_cycle_domain_waveform_keeps_warmup_and_audio_gan():
    t = _tiny_trainer(
        cycle_domain="waveform",
        disc_x=SimpleNamespace(),
        disc_y=SimpleNamespace(),
    )
    assert t.cycle_domain == "waveform"
    assert t.gan_domain == "audio"
    assert t.use_waveform_cycle is True
    assert t.use_ae_aware_cycle is False
    assert t.cycle_warmup_duration == 10
    assert t._needs_train_decode() is True


def test_cycle_domain_latent_ae_aware():
    t = _tiny_trainer(cycle_domain="latent", latent_cycle_mode="ae_aware")
    assert t.gan_domain == "latent"
    assert t.use_ae_aware_cycle is True
    assert t.use_latent_cycle is False
    assert t.use_waveform_cycle is False
    assert t.cycle_warmup_duration == 10
    assert t._needs_train_decode() is True


def test_cycle_domain_latent_direct_no_warmup_no_decode():
    t = _tiny_trainer(cycle_domain="latent", latent_cycle_mode="direct")
    assert t.gan_domain == "latent"
    assert t.use_latent_cycle is True
    assert t.use_ae_aware_cycle is False
    assert t.cycle_warmup_duration == 0
    assert t._needs_train_decode() is False


def test_direct_forward_skips_decoder():
    t = _tiny_trainer(cycle_domain="latent", latent_cycle_mode="direct")
    t.eval()
    x = torch.randn(4, 1, 64)
    domain = ["ood", "ood", "in", "in"]
    x_mask = torch.tensor([True, True, False, False])
    y_mask = ~x_mask
    with torch.no_grad():
        fwd = t._forward_batch(x, x_mask, y_mask, decode=False)
    assert fwd["z_x"].shape[0] == 2
    assert fwd["z_xy"].shape[0] == 2
    assert fwd["z_x_cycle"].shape == fwd["z_x"].shape
    assert fwd["y_fake"].shape[0] == 0
    assert fwd["x_fake"].shape[0] == 0
    assert fwd["z_y"].shape[0] == 2
    assert fwd["z_yx"].shape[0] == 2


def test_ae_aware_cycle_nonzero_at_random_init():
    t = _tiny_trainer(cycle_domain="latent", latent_cycle_mode="ae_aware")
    t.eval()
    x = torch.randn(4, 1, 64)
    x_mask = torch.tensor([True, True, False, False])
    y_mask = ~x_mask
    with torch.no_grad():
        fwd = t._forward_batch(x, x_mask, y_mask, decode=True)
        loss = t._latent_cycle_loss(fwd["z_x"], fwd["z_x_cycle"])
    assert fwd["y_fake"].shape[0] == 2
    assert loss.item() > 1e-4


def test_gan_pairs_use_latents_in_latent_domain():
    t = _tiny_trainer(cycle_domain="latent", latent_cycle_mode="direct")
    t.eval()
    x = torch.randn(4, 1, 64)
    x_mask = torch.tensor([True, True, False, False])
    y_mask = ~x_mask
    with torch.no_grad():
        fwd = t._forward_batch(x, x_mask, y_mask, decode=False)
        feat_real_y, feat_fake_y, feat_real_x, feat_fake_x = t._gan_pairs(
            fwd, detach=True)
    assert feat_real_y[0][-1].shape[0] == 2
    assert feat_fake_y[0][-1].shape[0] == 2
    assert feat_real_x[0][-1].shape[0] == 2
    assert feat_fake_x[0][-1].shape[0] == 2
