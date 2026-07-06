"""Offline reconstruction using the same block-wise signal flow as Max/nn~.

Loads an exported ``model.ts`` and processes audio in fixed-size blocks via
``model.forward()``, matching ``nn~ model.ts forward N`` in a Max patch.

Use this to reproduce (or debug) real-time nn~ behavior offline. Offline
checkpoint scripts such as ``latent_exploration/mask_reconstruct.py`` run the
full clip in one forward pass and will *not* match nn~ at small block sizes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

_RAVE_ROOT = Path(__file__).resolve().parents[2] / "RAVE"
if str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

from absl import app, flags, logging
import cached_conv as cc
import torch
import torchaudio

import rave

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "Path to exported model.ts (required).")
flags.DEFINE_multi_string("input", None, "Input wav file(s) or directory(ies).")
flags.DEFINE_string("out_dir", "nn_block_recon", "Output directory.")
flags.DEFINE_integer("block_size", 512, "Block size (samples), like nn~ ``forward N``.")
flags.DEFINE_integer("warmup_blocks", 0, "Number of zero blocks before processing.")
flags.DEFINE_bool(
    "streaming",
    True,
    "Use cached conv streaming mode (matches default nn~ export; --nostreaming for legacy .ts)",
)
flags.DEFINE_integer("attr_mode", 2,
                     "Fader only: attr_mode passed to set_attr_mode (2 = extract only).")
flags.DEFINE_integer("gpu", -1, "GPU index (-1 for CPU).")
flags.DEFINE_bool("save_input", True, "Also write resampled mono input wav.")


def _collect_audio_files(paths: Iterable[str]) -> List[Path]:
    valid_exts = rave.core.get_valid_extensions()
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for root, _, names in os.walk(p):
                for name in names:
                    if Path(name).suffix.lower() in valid_exts:
                        files.append(Path(root) / name)
        elif p.is_file():
            files.append(p)
        else:
            logging.warning("Skipping missing path: %s", raw)
    return sorted(set(files))


def _load_model(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    if hasattr(model, "set_attr_mode"):
        model.set_attr_mode(int(FLAGS.attr_mode))
        logging.info("Fader attr_mode=%d", int(FLAGS.attr_mode))
    return model


def _prepare_audio(
    path: Path,
    target_sr: int,
    n_channels: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    x, sr = torchaudio.load(str(path))
    if sr != target_sr:
        x = torchaudio.functional.resample(x, sr, target_sr)
    if x.shape[0] > n_channels:
        x = x[:n_channels]
    elif x.shape[0] < n_channels:
        x = x.repeat(n_channels, 1)
    return x.unsqueeze(0).to(device), target_sr


@torch.no_grad()
def block_reconstruct(
    model: torch.jit.ScriptModule,
    x: torch.Tensor,
    block_size: int,
    warmup_blocks: int = 0,
) -> torch.Tensor:
    """Process ``x`` (B, C, T) in ``block_size`` chunks, mirroring nn~ forward."""
    if x.dim() != 3:
        raise ValueError(f"Expected (batch, channels, time), got shape {tuple(x.shape)}")

    n_channels = x.shape[1]
    zeros = torch.zeros(
        x.shape[0], n_channels, block_size, device=x.device, dtype=x.dtype,
    )
    for _ in range(warmup_blocks):
        model(zeros)

    outputs: List[torch.Tensor] = []
    n_samples = x.shape[-1]
    for start in range(0, n_samples, block_size):
        chunk = x[..., start:start + block_size]
        valid = chunk.shape[-1]
        if valid < block_size:
            chunk = torch.nn.functional.pad(chunk, (0, block_size - valid))
        out = model(chunk)
        outputs.append(out[..., :valid])
    return torch.cat(outputs, dim=-1)


@torch.no_grad()
def main(argv):
    del argv
    if not FLAGS.model:
        raise app.UsageError("--model is required")
    if not FLAGS.input:
        raise app.UsageError("--input is required")

    torch.set_float32_matmul_precision("high")
    model_path = Path(FLAGS.model)
    if not model_path.is_file():
        raise app.UsageError(f"model not found: {model_path}")

    cc.use_cached_conv(FLAGS.streaming)
    logging.info(
        "cached_conv streaming=%s block_size=%d warmup_blocks=%d",
        FLAGS.streaming, FLAGS.block_size, FLAGS.warmup_blocks,
    )

    device = torch.device(
        f"cuda:{FLAGS.gpu}" if FLAGS.gpu >= 0 and torch.cuda.is_available() else "cpu",
    )
    probe = _load_model(model_path, device)
    target_sr = int(probe.sr)
    n_channels = int(probe.n_channels)
    del probe

    out_dir = Path(FLAGS.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = _collect_audio_files(FLAGS.input)
    if not audio_files:
        logging.error("No audio files found under %s", FLAGS.input)
        return

    for path in audio_files:
        logging.info("Processing %s", path)
        model = _load_model(model_path, device)
        x, _ = _prepare_audio(path, target_sr, n_channels, device)
        y = block_reconstruct(model, x, FLAGS.block_size, FLAGS.warmup_blocks)

        stem = path.stem
        tag = f"block{FLAGS.block_size}"
        recon_path = out_dir / f"{stem}_reconstructed_{tag}.wav"
        torchaudio.save(str(recon_path), y.squeeze(0).cpu(), target_sr)
        logging.info("Wrote %s (%d samples)", recon_path, y.shape[-1])

        if FLAGS.save_input:
            in_path = out_dir / f"{stem}_input_{tag}.wav"
            torchaudio.save(str(in_path), x.squeeze(0).cpu(), target_sr)


if __name__ == "__main__":
    app.run(main)
