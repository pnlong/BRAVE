"""Unit tests for CycleGAN cycle_domain / latent D / warp init."""

from types import SimpleNamespace

import gin
import torch
import torch.nn as nn

from rave.canonicalizer.cycle_trainer import CycleGANTrainer
from rave.canonicalizer.gan_utils import audio_gan_d, audio_gan_g, feature_matching_loss
from rave.canonicalizer.in_domain_discriminator import (
    InDomainLatentDiscriminator,
    build_latent_discriminator,
)
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


def test_build_latent_discriminator_honors_gin_architecture():
    gin.clear_config()
    gin.parse_config("""
        InDomainLatentDiscriminator.hidden_size = 64
        InDomainLatentDiscriminator.n_layers = 1
        InDomainLatentDiscriminator.kernel_size = 1
    """)
    try:
        disc = build_latent_discriminator(8)
        assert len(disc.blocks) == 1
        conv = disc.blocks[0][0]
        assert conv.in_channels == 8
        assert conv.out_channels == 64
        assert conv.kernel_size == (1,)
    finally:
        gin.clear_config()


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


def test_hybrid_latent_gan_requires_discs():
    try:
        _tiny_trainer(
            cycle_domain="waveform",
            disc_x=SimpleNamespace(),
            disc_y=SimpleNamespace(),
            lambda_latent_gan=1.0,
            disc_latent_x=None,
            disc_latent_y=None,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "disc_latent" in str(exc)


def test_hybrid_latent_gan_active_on_waveform():
    lat_x = InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2)
    lat_y = InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2)
    t = _tiny_trainer(
        cycle_domain="waveform",
        disc_x=SimpleNamespace(),
        disc_y=SimpleNamespace(),
        disc_latent_x=lat_x,
        disc_latent_y=lat_y,
        lambda_latent_gan=1.0,
    )
    assert t.hybrid_latent_gan is True
    assert t.gan_domain == "audio"
    assert t._waveform_cycle_weight() == t.lambda_cycle


def test_audio_polish_weight_ramps_on_latent():
    t = _tiny_trainer(
        cycle_domain="latent",
        latent_cycle_mode="ae_aware",
        audio_polish_start_step=100,
        audio_polish_ramp_duration=10,
        lambda_audio_polish=10.0,
    )
    assert t.audio_polish_active is True
    assert t.use_waveform_cycle is False
    t.audio_polish_factor = 0.0
    assert t._waveform_cycle_weight() == 0.0
    t.audio_polish_factor = 0.5
    assert abs(t._waveform_cycle_weight() - 5.0) < 1e-6
    assert t._needs_train_decode() is True
    assert t._needs_waveform_roundtrip() is True


def test_latent_gan_pairs_hybrid():
    lat_x = InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2)
    lat_y = InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2)
    t = _tiny_trainer(
        cycle_domain="waveform",
        disc_x=InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2),
        disc_y=InDomainLatentDiscriminator(latent_size=8, hidden_size=16, n_layers=2),
        disc_latent_x=lat_x,
        disc_latent_y=lat_y,
        lambda_latent_gan=1.0,
    )
    t.eval()
    x = torch.randn(4, 1, 64)
    x_mask = torch.tensor([True, True, False, False])
    y_mask = ~x_mask
    with torch.no_grad():
        fwd = t._forward_batch(x, x_mask, y_mask, decode=True)
        feat_real_y, feat_fake_y, feat_real_x, feat_fake_x = t._latent_gan_pairs(
            fwd, detach=True)
    assert feat_real_y[0][-1].shape[0] == 2
    assert feat_fake_y[0][-1].shape[0] == 2
    assert feat_real_x[0][-1].shape[0] == 2
    assert feat_fake_x[0][-1].shape[0] == 2


def test_cyclegan_checkpoint_strips_backbones_and_restores_scales():
    t = _tiny_trainer(cycle_domain="waveform")
    t.stft_loss_scale = 12.5
    t.loss_scales_calibrated = True
    t.latent_var_ref_x = torch.ones(8)
    checkpoint = {
        "state_dict": {
            "backbone_x._dummy": torch.tensor([1.0]),
            "warp_xy.conv.weight": torch.zeros(8, 8, 1),
        }
    }
    t.on_save_checkpoint(checkpoint)
    assert "backbone_x._dummy" not in checkpoint["state_dict"]
    assert "warp_xy.conv.weight" in checkpoint["state_dict"]
    assert checkpoint["cyclegan_extra"]["stft_loss_scale"] == 12.5
    assert checkpoint["cyclegan_extra"]["loss_scales_calibrated"] is True

    t2 = _tiny_trainer(cycle_domain="waveform")
    t2.on_load_checkpoint(checkpoint)
    assert t2.stft_loss_scale == 12.5
    assert t2.loss_scales_calibrated is True
    assert t2.calibrate_loss_scales is False
    assert torch.equal(t2.latent_var_ref_x.cpu(), torch.ones(8))
    assert any(k.startswith("backbone_x.") for k in checkpoint["state_dict"])


def test_shared_backbone_alias_and_identity_init():
    bb = _FakeBackbone()
    warp_xy = LatentCanonicalizer(latent_size=8, init_mode="identity")
    warp_yx = LatentCanonicalizer(latent_size=8, init_mode="identity")
    assert warp_xy.init_mode == "identity"
    assert warp_yx.init_mode == "identity"
    t = _tiny_trainer(
        backbone_x=bb,
        backbone_y=bb,
        warp_xy=warp_xy,
        warp_yx=warp_yx,
        shared_backbone=True,
        cycle_domain="waveform",
        disc_x=SimpleNamespace(),
        disc_y=SimpleNamespace(),
        lambda_identity=0.5,
    )
    assert t.shared_backbone is True
    assert t.backbone_x is t.backbone_y
    assert list(t._iter_backbones()) == [bb]
    # Identity residual at init: warp ≈ identity
    z = torch.randn(2, 8, 4)
    assert torch.allclose(warp_xy(z), z, atol=1e-5)


def test_joint_export_manifest_flags_shared_load(monkeypatch):
    """load_cyclegan_xy_for_export loads one backbone when geometry=joint."""
    from rave.canonicalizer.config import CycleGANManifest
    from rave.canonicalizer.export import cyclegan_nn as export_mod

    calls = []

    def fake_load_backbone(config_path, ckpt_path, n_channels):
        calls.append((config_path, ckpt_path, n_channels))
        return _FakeBackbone()

    monkeypatch.setattr(export_mod, "_load_backbone", fake_load_backbone)
    monkeypatch.setattr(
        export_mod,
        "load_cyclegan_checkpoint",
        lambda path: (
            {},
            {},
            CycleGANManifest(
                canonicalizer_type="latent",
                backbone_x_config="/j.gin",
                backbone_x_ckpt="/j.ckpt",
                backbone_y_config="/j.gin",
                backbone_y_ckpt="/j.ckpt",
                db_path_x="/x",
                db_path_y="/y",
                init_mode="identity",
                geometry="joint",
                shared_backbone=True,
            ),
        ),
    )

    def fake_warps(manifest, backbone_x, backbone_y, warp_xy_state, warp_yx_state):
        w = LatentCanonicalizer(latent_size=8, init_mode="identity")
        return w, w

    monkeypatch.setattr(export_mod, "build_cyclegan_warps", fake_warps)
    monkeypatch.setattr(export_mod.cc, "use_cached_conv", lambda *_a, **_k: None)
    monkeypatch.setattr(export_mod, "_remove_weight_norm", lambda *_a, **_k: None)

    bx, by, warp, manifest = export_mod.load_cyclegan_xy_for_export("/fake.ckpt")
    assert len(calls) == 1
    assert bx is by
    assert manifest.geometry == "joint"
    assert warp.init_mode == "identity"
