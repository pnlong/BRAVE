"""Tests for conditional canonicalizer (Fader attr_cls D + OOD attr policy)."""

import torch
import torch.nn as nn

from rave.canonicalizer.attribute_conditioning import AttributeConditioningEmbed
from rave.canonicalizer.backbone import assign_ood_target_attrs
from rave.canonicalizer.in_domain_discriminator import InDomainAudioDiscriminator
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer


class _TinyMSD(nn.Module):
    def __init__(self, n_channels: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_channels, 4, 15, stride=4), nn.Conv1d(4, 1, 1)),
        ])

    def forward(self, x):
        out = []
        for layer in self.layers:
            feats = []
            h = x
            for mod in layer:
                h = mod(h)
                if isinstance(mod, nn.Conv1d):
                    feats.append(h)
            out.append(feats)
        return out


class _FakeFaderBackbone(nn.Module):
    n_channels = 1
    num_attributes = 2
    attribute_names = ["rms", "texture_class"]
    attribute_kinds = {"rms": "continuous", "texture_class": "discrete"}
    discrete_num_classes = {"texture_class": 4}

    def _prepare_attributes(self, attr_raw):
        b, d, t = attr_raw.shape
        attr_norm = torch.zeros(b, d, t, device=attr_raw.device)
        attr_cls = torch.zeros(b, d, t, device=attr_raw.device, dtype=torch.long)
        attr_cls[:, 1, :] = attr_raw[:, 1, :].long().clamp(0, 3)
        return attr_norm, attr_cls


def test_attribute_conditioning_embed_shape():
    emb = AttributeConditioningEmbed(
        num_attributes=2,
        num_classes_per_attribute=[16, 4],
        embed_dim=8,
    )
    attr_cls = torch.zeros(3, 2, 16, dtype=torch.long)
    attr_cls[:, 1, :] = 2
    out = emb(attr_cls=attr_cls)
    assert out.shape == (3, 16)


def test_in_domain_disc_unconditional_matches_no_attr():
    disc = InDomainAudioDiscriminator(
        discriminator=lambda **_: _TinyMSD(),
        n_channels=1,
        num_attributes=0,
    )
    x = torch.randn(2, 1, 4096)
    feats = disc(x)
    assert len(feats) == 1
    assert feats[0][-1].shape[0] == 2


def test_in_domain_disc_conditional_forward():
    disc = InDomainAudioDiscriminator(
        discriminator=lambda **_: _TinyMSD(),
        n_channels=1,
        num_attributes=2,
        num_classes_per_attribute=[16, 4],
        embed_dim=8,
    )
    x = torch.randn(2, 1, 4096)
    attr_cls = torch.zeros(2, 2, 8, dtype=torch.long)
    attr_cls[:, 1, :] = 1
    feats = disc(x, attr_cls=attr_cls)
    assert feats[0][-1].shape == (2, 1, feats[0][-1].shape[-1])


def test_assign_ood_target_attrs_samples_discrete_only():
    model = _FakeFaderBackbone()
    t = 8
    attr = torch.zeros(4, 2, t)
    ood_mask = torch.tensor([False, True, False, True])
    torch.manual_seed(0)
    out = assign_ood_target_attrs(attr, model, ood_mask)
    assert torch.all(out[0, 1, :] == 0)
    assert torch.all(out[2, 1, :] == 0)
    assert torch.all((out[1, 1, :] >= 0) & (out[1, 1, :] < 4))
    assert torch.all((out[3, 1, :] >= 0) & (out[3, 1, :] < 4))
    assert torch.all(out[1, 0, :] == 0)
    assert torch.all(out[3, 0, :] == 0)


def test_latent_canonicalizer_conditional_identity():
    lc = LatentCanonicalizer(
        latent_size=16,
        num_attributes=2,
        num_classes_per_attribute=[16, 4],
        embed_dim=8,
    )
    z = torch.randn(2, 16, 32)
    attr_cls = torch.zeros(2, 2, 32, dtype=torch.long)
    z2 = lc(z, attr_cls=attr_cls)
    assert z2.shape == z.shape
    assert torch.allclose(z2, z, atol=1e-5)
