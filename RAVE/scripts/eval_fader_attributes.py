"""
Evaluate Fader attribute disentanglement via attribute swap correlation.

Port of neurorave helpers/eval.py get_corr_attr for BRAVE FaderRAVE.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/eval_fader_attributes.py \\
    --model runs/brave_fader_run \\
    --db_path /path/to/lmdb \\
    --batch 4 --max_batches 10
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

flags.DEFINE_string("model", None, "FaderRAVE run directory or ckpt", required=True)
flags.DEFINE_string("db_path", None, "LMDB path", required=True)
flags.DEFINE_integer("batch", 4, "Batch size")
flags.DEFINE_integer("max_batches", 10, "Max eval batches (0 = all val)")
flags.DEFINE_integer("n_signal", 131072, "Chunk length")
flags.DEFINE_string("output", None, "JSON output path (default: model dir)")


def load_fader_model(model_path: str) -> FaderRAVE:
    """Load FaderRAVE checkpoint from run directory."""
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        raise FileNotFoundError(f"config.gin not found near {model_path}")
    gin.parse_config_file(config_path)
    run = rave.core.search_for_run(model_path)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found near {model_path}")
    model = FaderRAVE()
    model = model.load_from_checkpoint(run)
    model.eval()
    return model


@torch.no_grad()
def eval_attribute_swap(
    model: FaderRAVE,
    loader: DataLoader,
    max_batches: int,
    device: torch.device,
) -> dict:
    """
    Swap normalized attributes across batch pairs; measure re-extracted correlation.

    Metric intuition: if z is disentangled from attributes, swapping attr while
    keeping z should change synthesized timbre; re-extracted attrs should match
    the swapped controls (high cosine, low L1 vs swapped attr_norm).
    """
    continuous = model.continuous_attributes
    if not continuous:
        return {"note": "no continuous attributes configured"}

    l1_sums = {n: 0.0 for n in continuous}
    cos_sums = {n: 0.0 for n in continuous}
    n_pairs = 0

    for bi, batch in enumerate(tqdm(loader, desc="eval swap")):
        if max_batches and bi >= max_batches:
            break
        x_raw, attr_raw = batch
        x_raw = x_raw.to(device)
        attr_raw = attr_raw.to(device)
        attr_norm, _ = model._prepare_attributes(attr_raw)

        b = x_raw.shape[0]
        if b < 2:
            continue

        # --- Encode content latents ---
        z = model.encode(x_raw, return_mb=False)
        z, _ = model.encoder.reparametrize(z)[:2]

        # --- Swap attributes: keep z[i], use attr from j=(i+1)%b ---
        perm = torch.roll(torch.arange(b, device=device), 1)
        attr_swapped = attr_norm[perm]

        y = model.decode(z, attr=attr_swapped)

        # --- Re-extract continuous raw attributes from synthesized audio ---
        from rave.fader.providers import AudioDescriptorProvider
        from rave.fader.attributes import latent_length_from_config

        t_lat = attr_raw.shape[-1]
        provider = AudioDescriptorProvider(
            continuous_attributes=continuous,
            sampling_rate=model.sr,
        )
        reextract = []
        for i in range(b):
            wav = y[i].detach().cpu().numpy()
            if wav.ndim == 2:
                mono = wav.mean(axis=0)
            else:
                mono = wav.reshape(-1)
            feat = provider.load(0, mono, model.sr, t_lat)
            reextract.append(feat)
        reextract = np.stack(reextract, axis=0)
        reextract_t = torch.from_numpy(reextract).float().to(device)
        re_norm, _ = model._prepare_attributes(reextract_t)

        for ci, name in enumerate(continuous):
            idx = model.attribute_names.index(name)
            orig = attr_swapped[:, idx, :]
            new = re_norm[:, idx, :]
            # --- Compare swapped control vs re-extracted from synthesis ---
            l1 = (orig - new).abs().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                orig.reshape(b, -1), new.reshape(b, -1), dim=1).mean().item()
            l1_sums[name] += l1
            cos_sums[name] += cos
        n_pairs += b

    if n_pairs == 0:
        return {"error": "no pairs evaluated"}

    return {
        "n_samples": n_pairs,
        "l1": {k: v / max(n_pairs, 1) for k, v in l1_sums.items()},
        "cosine": {k: v / max(n_pairs, 1) for k, v in cos_sums.items()},
    }


def main(argv):
    del argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fader_model(FLAGS.model)
    model = model.to(device)

    stats_path = resolve_stats_path(FLAGS.db_path)
    if stats_path is None:
        raise FileNotFoundError(f"No attribute_stats.yaml in {FLAGS.db_path}")
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

    results = eval_attribute_swap(model, loader, FLAGS.max_batches, device)
    out = FLAGS.output or os.path.join(
        FLAGS.model if os.path.isdir(FLAGS.model) else os.path.dirname(FLAGS.model),
        "eval_attribute_swap.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    app.run(main)
