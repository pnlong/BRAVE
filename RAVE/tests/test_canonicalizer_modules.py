import torch

from rave.dsp import BiquadBank, CausalReverb
from rave.canonicalizer.waveform_canonicalizer import WaveformCanonicalizer
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer


def test_biquad_bank_identity_at_init():
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    x = torch.randn(2, 1, 4096)
    y = eq(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)


def test_biquad_bank_grad_flow():
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    eq.filters[0].gain_db.data.fill_(1.0)
    x = torch.randn(1, 1, 2048, requires_grad=True)
    y = eq(x).sum()
    y.backward()
    assert eq.filters[0].gain_db.grad is not None


def test_biquad_bank_grad_at_identity_init():
    """Identity init must still connect gain_db to the graph (no early-return bypass)."""
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    x = torch.randn(1, 1, 2048)
    y = eq(x).sum()
    assert y.requires_grad
    y.backward()
    assert eq.filters[0].gain_db.grad is not None


def test_causal_reverb_grad_at_dry_init():
    rv = CausalReverb(sample_rate=44100)
    x = torch.randn(1, 1, 8192)
    y = rv(x).sum()
    assert y.requires_grad
    y.backward()
    assert rv.wet_logit.grad is not None


def test_waveform_canonicalizer_grad_at_identity_init():
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    rv = CausalReverb(sample_rate=44100)
    wc = WaveformCanonicalizer(eq=eq, reverb=rv, use_reverb=True)
    x = torch.randn(1, 1, 4096)
    y = wc(x).sum()
    assert y.requires_grad
    y.backward()
    assert eq.filters[0].gain_db.grad is not None
    assert rv.wet_logit.grad is not None


def test_causal_reverb_dry_identity():
    rv = CausalReverb(sample_rate=44100)
    x = torch.randn(1, 1, 8192)
    y = rv(x)
    assert torch.allclose(y, x, atol=1e-4)


def test_waveform_canonicalizer_eq_only():
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    wc = WaveformCanonicalizer(eq=eq, reverb=None, use_reverb=False)
    x = torch.randn(1, 1, 4096)
    assert torch.allclose(wc(x), x, atol=1e-4)


def test_waveform_canonicalizer_with_reverb():
    eq = BiquadBank(sample_rate=44100, n_bands=4)
    rv = CausalReverb(sample_rate=44100)
    wc = WaveformCanonicalizer(eq=eq, reverb=rv, use_reverb=True)
    x = torch.randn(1, 1, 4096)
    assert torch.allclose(wc(x), x, atol=1e-4)


def test_domain_y_gan_mismatched_batch_sizes():
    """Mixed batches have unequal in-domain vs OOD counts."""
    import torch.nn as nn
    from types import SimpleNamespace

    from rave.canonicalizer.trainer import CanonicalizerTrainer
    from rave.core import hinge_gan

    class TinyDisc(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv1d(1, 1, kernel_size=1)

        def forward(self, x):
            return [[self.proj(x).mean(dim=-1, keepdim=True)]]

    trainer = SimpleNamespace(
        in_domain_disc=TinyDisc(),
        gan_loss_fn=hinge_gan,
    )
    y = torch.randn(64, 1, 4096, requires_grad=True)
    in_mask = torch.cat([
        torch.ones(53, dtype=torch.bool),
        torch.zeros(11, dtype=torch.bool),
    ])
    ood_mask = ~in_mask
    feat_real, feat_fake = CanonicalizerTrainer._disc_features(
        trainer, y[in_mask], y[ood_mask], detach=True)
    loss_d = CanonicalizerTrainer._audio_gan_d(trainer, feat_real, feat_fake)
    assert loss_d.ndim == 0
    loss_d.backward()

    y2 = torch.randn(11, 1, 4096, requires_grad=True)
    feat_fake_g = trainer.in_domain_disc(y2)
    loss_g = CanonicalizerTrainer._audio_gan_g(trainer, feat_fake_g)
    assert loss_g.ndim == 0
    loss_g.backward()


def test_latent_canonicalizer_identity():
    lc = LatentCanonicalizer(latent_size=128)
    z = torch.randn(2, 128, 64)
    z2 = lc(z)
    assert z2.shape == z.shape
    assert torch.allclose(z2, z, atol=1e-5)


def test_frame_rms_curve_shape_and_grad():
    from rave.canonicalizer.losses import frame_rms_curve, rms_recon_l1

    x = torch.randn(2, 1, 8192, requires_grad=True)
    y = x * 0.9 + 0.05
    curve = frame_rms_curve(x, n_frames=32)
    assert curve.shape == (2, 32)
    loss = rms_recon_l1(y, x, n_frames=32)
    loss.backward()
    assert x.grad is not None
