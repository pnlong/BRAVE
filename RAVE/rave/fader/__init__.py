"""Fader Networks conditioning for BRAVE RAVE.

Import submodules directly (``rave.fader.dataset``, ``rave.fader.model``, …) or rely on
gin config files (``configs/brave_fader.gin``) to register configurables before the
config lock. Avoid ``from rave.fader import …`` after ``gin.parse_config*`` unless
those modules were already imported during config parsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "compute_bins": (".attributes", "compute_bins"),
    "compute_descriptor_matrix": (".attributes", "compute_descriptor_matrix"),
    "latent_length_from_config": (".attributes", "latent_length_from_config"),
    "load_attribute_stats": (".attributes", "load_attribute_stats"),
    "ordered_attributes": (".attributes", "ordered_attributes"),
    "resolve_stats_path": (".attributes", "resolve_stats_path"),
    "save_attribute_stats": (".attributes", "save_attribute_stats"),
    "LambdaWarmupCallback": (".callbacks", "LambdaWarmupCallback"),
    "FaderAttributeDataset": (".dataset", "FaderAttributeDataset"),
    "wrap_fader_dataset": (".dataset", "wrap_fader_dataset"),
    "LatentDiscriminator": (".latent_discriminator", "LatentDiscriminator"),
    "FaderRAVE": (".model", "FaderRAVE"),
    "AttributeLoader": (".providers", "AttributeLoader"),
    "AudioDescriptorProvider": (".providers", "AudioDescriptorProvider"),
    "build_attribute_loader": (".providers", "build_attribute_loader"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
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
    from .providers import AttributeLoader, AudioDescriptorProvider, build_attribute_loader
