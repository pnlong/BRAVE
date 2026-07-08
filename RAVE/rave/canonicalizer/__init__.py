"""Stage-1 input canonicalizer: waveform / latent warps on frozen RAVE backbones."""

from .latent_canonicalizer import LatentCanonicalizer
from .waveform_canonicalizer import WaveformCanonicalizer
from .in_domain_discriminator import InDomainAudioDiscriminator
from .trainer import CanonicalizerTrainer

__all__ = [
    "LatentCanonicalizer",
    "WaveformCanonicalizer",
    "InDomainAudioDiscriminator",
    "CanonicalizerTrainer",
]

# Export helpers: ``rave.canonicalizer.export``
