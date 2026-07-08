"""TorchScript and nn~ export helpers for FaderRAVE."""

from .bundle import finalize_nn_bundle, print_max_copy_instructions
from .host_controls import write_host_controls_json
from .load_for_export import (
    is_fader_config,
    is_fader_model,
    load_fader_for_export,
    strip_weight_norm,
)
from .max_patch import write_fader_play_patch, write_vanilla_play_patch
from .nn_module import ScriptedFaderRAVE
from .trace_model import FaderTraceModel, build_trace_model

__all__ = [
    "FaderTraceModel",
    "ScriptedFaderRAVE",
    "build_trace_model",
    "write_host_controls_json",
    "finalize_nn_bundle",
    "print_max_copy_instructions",
    "load_fader_for_export",
    "strip_weight_norm",
    "is_fader_model",
    "write_fader_play_patch",
    "write_vanilla_play_patch",
]
