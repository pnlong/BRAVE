"""
Pluggable attribute providers for Fader training.

Swap attribute sources via gin + sidecar YAML, not dataset.py.
"""

from .audio import AudioDescriptorProvider, CachingAudioDescriptorProvider
from .base import ContinuousAttributeProvider, DiscreteAttributeProvider
from .learned import LearnedFeatureProvider
from .loader import AttributeLoader, build_attribute_loader
from .midi_cc import MidiCCSidecarProvider
from .null import NullDiscreteProvider
from .sidecar import SidecarAttributeProvider

__all__ = [
    "AttributeLoader",
    "AudioDescriptorProvider",
    "CachingAudioDescriptorProvider",
    "ContinuousAttributeProvider",
    "DiscreteAttributeProvider",
    "LearnedFeatureProvider",
    "MidiCCSidecarProvider",
    "NullDiscreteProvider",
    "SidecarAttributeProvider",
    "build_attribute_loader",
]
