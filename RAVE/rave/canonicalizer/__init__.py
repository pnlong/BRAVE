"""Stage-1 input canonicalizer: waveform / latent warps on frozen RAVE backbones."""

from .latent_canonicalizer import LatentCanonicalizer, infer_latent_warp_hparams
from .waveform_canonicalizer import (
    WaveformCanonicalizer,
    WaveformKnobEncoder,
    WaveformKnobLayout,
    build_waveform_canonicalizer,
)
from .in_domain_discriminator import InDomainAudioDiscriminator
from .trainer import CanonicalizerTrainer
from .cycle_trainer import CycleGANTrainer

__all__ = [
    "LatentCanonicalizer",
    "infer_latent_warp_hparams",
    "WaveformCanonicalizer",
    "WaveformKnobEncoder",
    "WaveformKnobLayout",
    "build_waveform_canonicalizer",
    "InDomainAudioDiscriminator",
    "CanonicalizerTrainer",
    "CycleGANTrainer",
]

# Export helpers: ``rave.canonicalizer.export``
