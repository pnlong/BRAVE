"""Fader Networks conditioning for BRAVE RAVE."""

from .attributes import (
    compute_bins,
    compute_descriptor_matrix,
    latent_length_from_config,
    load_attribute_stats,
    ordered_attributes,
    resolve_stats_path,
    save_attribute_stats,
)
from .callbacks import LambdaWarmupCallback
from .dataset import FaderAttributeDataset, wrap_fader_dataset
from .latent_discriminator import LatentDiscriminator
from .model import FaderRAVE
from .providers import (
    AttributeLoader,
    AudioDescriptorProvider,
    build_attribute_loader,
)

__all__ = [
    "FaderRAVE",
    "LatentDiscriminator",
    "LambdaWarmupCallback",
    "FaderAttributeDataset",
    "wrap_fader_dataset",
    "AttributeLoader",
    "AudioDescriptorProvider",
    "build_attribute_loader",
    "compute_descriptor_matrix",
    "compute_bins",
    "latent_length_from_config",
    "load_attribute_stats",
    "save_attribute_stats",
    "ordered_attributes",
    "resolve_stats_path",
]
