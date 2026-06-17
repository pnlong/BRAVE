import torch

from rave.fader.latent_domain_discriminator import LatentDomainDiscriminator


def test_latent_domain_discriminator_forward_shape():
    disc = LatentDomainDiscriminator(latent_size=128, base_channels=64, num_layers=2)
    z = torch.randn(3, 128, 32)
    out = disc(z)
    assert out.shape == (3, 1, 32)
