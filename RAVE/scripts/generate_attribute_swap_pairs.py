"""
Generate WAV pairs for subjective Fader attribute-swap listening tests.

Usage (BRAVE root):
  python RAVE/scripts/generate_attribute_swap_pairs.py \\
    --model runs/brave_fader_run --db_path /path/to/lmdb \\
    --output_dir listening/swap_pairs --n_clips 8
"""

from __future__ import annotations

import json
import os
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import gin
import numpy as np
import soundfile as sf
import torch
from absl import app, flags
from torch.utils.data import DataLoader
from tqdm import tqdm

import rave
import rave.dataset
from rave.fader.attributes import resolve_stats_path
from rave.fader.dataset import wrap_fader_dataset
from rave.fader.model import FaderRAVE

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "FaderRAVE run dir", required=True)
flags.DEFINE_string("db_path", None, "LMDB path", required=True)
flags.DEFINE_string("output_dir", None, "Output directory", required=True)
flags.DEFINE_integer("n_clips", 8, "Number of val clips to export")
flags.DEFINE_integer("n_signal", 131072, "Chunk length")
flags.DEFINE_integer("batch", 4, "Loader batch size")


def load_model(model_path: str) -> FaderRAVE:
    config_path = rave.core.search_for_config(model_path)
    gin.parse_config_file(config_path)
    run = rave.core.search_for_run(model_path)
    model = FaderRAVE()
    model = model.load_from_checkpoint(run)
    model.eval()
    return model


@torch.no_grad()
def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(FLAGS.model).to(device)

    stats_path = resolve_stats_path(FLAGS.db_path)
    model.load_attribute_stats_from_file(stats_path)

    dataset = rave.dataset.get_dataset(
        FLAGS.db_path, model.sr, FLAGS.n_signal, n_channels=model.n_channels)
    _, val = rave.dataset.split_dataset(dataset, 98)
    val = wrap_fader_dataset(
        val,
        sampling_rate=model.sr,
        n_signal=FLAGS.n_signal,
        db_path=FLAGS.db_path,
    )
    loader = DataLoader(val, FLAGS.batch, shuffle=False, num_workers=0)

    manifest = []
    exported = 0
    sr = model.sr

    for batch_idx, (x_raw, attr_raw) in enumerate(tqdm(loader, desc="export pairs")):
        if exported >= FLAGS.n_clips:
            break
        x_raw = x_raw.to(device)
        attr_raw = attr_raw.to(device)
        attr_norm, _ = model._prepare_attributes(attr_raw)

        z = model.encode(x_raw, return_mb=False)
        z, _ = model.encoder.reparametrize(z)[:2]

        b = x_raw.shape[0]
        perm = torch.roll(torch.arange(b, device=device), 1)

        y_orig = model.decode(z, attr=attr_norm)
        y_swap = model.decode(z, attr=attr_norm[perm])
        identity = torch.arange(b, device=device)
        y_self = model.decode(z, attr=attr_norm[identity])

        for i in range(b):
            if exported >= FLAGS.n_clips:
                break
            tag = f"clip_{exported:04d}"
            mono = lambda t: t[i].detach().cpu().numpy().squeeze()

            sf.write(os.path.join(FLAGS.output_dir, f"{tag}_original.wav"),
                     mono(y_orig), sr)
            sf.write(os.path.join(FLAGS.output_dir, f"{tag}_swapped.wav"),
                     mono(y_swap), sr)
            sf.write(os.path.join(FLAGS.output_dir, f"{tag}_self_swap.wav"),
                     mono(y_self), sr)

            manifest.append({
                "tag": tag,
                "batch_index": batch_idx,
                "in_batch": i,
                "swap_partner_in_batch": int(perm[i].item()),
                "attribute_names": list(model.attribute_names),
            })
            exported += 1

    manifest_path = os.path.join(FLAGS.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"clips": manifest, "n_exported": exported}, f, indent=2)
    print(f"Wrote {exported} clip triplets to {FLAGS.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    app.run(main)
