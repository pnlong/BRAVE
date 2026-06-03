"""
Precompute attribute statistics (min/max, quantile bins) for Fader training.

Uses the same AttributeLoader as training. Writes attribute_stats.yaml.

Usage (from BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/precompute_descriptors.py \\
    --db_path /path/to/lmdb \\
    --n_signal 131072 \\
    --n_bands 16 \\
    --ratios 2,2,2,1
"""

from __future__ import annotations

import os
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import yaml
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

import rave.dataset
from rave.fader.attributes import (
    compute_bins,
    latent_length_from_config,
    min_max_from_features,
    ordered_attributes,
    save_attribute_stats,
)
from rave.fader.providers import build_attribute_loader

FLAGS = flags.FLAGS

flags.DEFINE_string("db_path", None, "LMDB dataset path", required=True)
flags.DEFINE_integer("n_signal", 131072, "Chunk length in samples")
flags.DEFINE_integer("n_bands", 16, "PQMF bands")
flags.DEFINE_list("ratios", ["2", "2", "2", "1"], "Encoder ratios (comma-sep)")
flags.DEFINE_multi_string(
    "continuous_attributes",
    ["centroid", "rms", "bandwidth", "sharpness", "booming"],
    "Continuous attribute names",
)
flags.DEFINE_multi_string(
    "discrete_attributes",
    [],
    "Discrete attribute names",
)
flags.DEFINE_integer("nb_bins", 16, "Quantile bins for continuous latent CE")
flags.DEFINE_integer("max_chunks", 0, "Cap chunks scanned (0 = all)")
flags.DEFINE_integer("sr", 0, "Sample rate override (0 = metadata.yaml)")
flags.DEFINE_bool(
    "train_only",
    True,
    "Compute stats on train split only (mirrors split_dataset seed/percent)",
)
flags.DEFINE_integer("split_percent", 98, "Train split percent")
flags.DEFINE_integer("split_seed", 42, "Random split seed")
flags.DEFINE_string(
    "bin_method",
    "quantile",
    "Binning method: quantile or rms_gate",
)
flags.DEFINE_float("rms_gate_db", -40.0, "RMS gate threshold for rms_gate bin_method")


def _stats_path(db_path: str) -> str:
    return os.path.join(db_path, "attribute_stats.yaml")


def _train_indices(n: int, percent: int, seed: int) -> np.ndarray:
    """Mirror rave.dataset.split_dataset index assignment."""
    split1 = max((percent * n) // 100, 1)
    split2 = n - split1
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[:split1].numpy()


def main(argv):
    del argv
    db_path = FLAGS.db_path
    ratios = [int(r) for r in FLAGS.ratios]
    continuous = list(FLAGS.continuous_attributes)
    discrete = list(FLAGS.discrete_attributes)
    attribute_names = ordered_attributes(continuous, discrete)
    t_lat = latent_length_from_config(FLAGS.n_signal, FLAGS.n_bands, ratios)

    # --- Dataset metadata (sr, channels, lazy vs eager LMDB) ---
    with open(os.path.join(db_path, "metadata.yaml"), "r") as f:
        meta = yaml.safe_load(f)
    sr = FLAGS.sr or meta.get("sr", 44100)

    import rave.transforms as transforms

    base_transforms = transforms.Compose([
        lambda x: x.astype(np.float32),
        transforms.Dequantize(16),
    ])
    lazy = meta.get("lazy", False)
    if lazy:
        dataset = rave.dataset.LazyAudioDataset(
            db_path,
            FLAGS.n_signal,
            sr,
            base_transforms,
            meta.get("channels", 1),
        )
    else:
        dataset = rave.dataset.AudioDataset(
            db_path,
            transforms=base_transforms,
            n_channels=meta.get("channels", 1),
        )

    n = len(dataset)
    if FLAGS.max_chunks > 0:
        n = min(n, FLAGS.max_chunks)

    if FLAGS.train_only:
        # --- Match train.py split so stats reflect training distribution only ---
        indices = _train_indices(n, FLAGS.split_percent, FLAGS.split_seed)
        print(
            f"Train-only stats: {len(indices)}/{n} chunks "
            f"(split={FLAGS.split_percent}%, seed={FLAGS.split_seed})"
        )
    else:
        indices = np.arange(n)
        print(f"Full-dataset stats: {n} chunks")

    loader = build_attribute_loader(
        continuous_attributes=continuous,
        discrete_attributes=discrete,
        sampling_rate=sr,
        latent_length=t_lat,
        db_path=db_path,
    )

    print(f"Computing attributes for {len(indices)} chunks, T_lat={t_lat}, D={len(attribute_names)}")

    # --- Scan chunks: same AttributeLoader path as training dataloader ---
    allfeatures = []
    for i in tqdm(indices):
        audio = dataset[int(i)]
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio.reshape(-1)
        if len(mono) > FLAGS.n_signal:
            mono = mono[:FLAGS.n_signal]
        elif len(mono) < FLAGS.n_signal:
            mono = np.pad(mono, (0, FLAGS.n_signal - len(mono)))
        try:
            feat = loader.load(int(i), mono if mono.ndim == 1 else mono, sr=sr)
        except Exception as exc:
            print(f"chunk {i} failed: {exc}")
            feat = np.zeros((len(attribute_names), t_lat), dtype=np.float32)
        allfeatures.append(feat)

    allfeatures = np.stack(allfeatures, axis=0)

    # --- Min/max + bins for continuous attributes only ---
    cont_indices = [attribute_names.index(c) for c in continuous]
    if continuous:
        cont_feats = allfeatures[:, cont_indices, :]
        min_max = min_max_from_features(cont_feats, continuous)
        bin_values_full = np.zeros((len(attribute_names), FLAGS.nb_bins), dtype=np.float32)
        cont_bins = compute_bins(
            cont_feats,
            continuous,
            FLAGS.nb_bins,
            bin_method=FLAGS.bin_method,
            rms_gate_db=FLAGS.rms_gate_db,
        )
        for j, ci in enumerate(cont_indices):
            bin_values_full[ci] = cont_bins[j]
    else:
        min_max = {}
        bin_values_full = np.zeros((len(attribute_names), FLAGS.nb_bins), dtype=np.float32)

    kinds = {n: "continuous" for n in continuous}
    kinds.update({n: "discrete" for n in discrete})
    disc_classes = {}
    for name in discrete:
        # --- Infer num_classes from max observed index (+1 for zero-based) ---
        idx = attribute_names.index(name)
        vals = allfeatures[:, idx, :].flatten()
        disc_classes[name] = int(max(vals.max(), 1)) + 1

    out_path = _stats_path(db_path)
    save_attribute_stats(
        out_path,
        attribute_names=attribute_names,
        min_max_features=min_max,
        bin_values=bin_values_full,
        nb_bins=FLAGS.nb_bins,
        latent_length=t_lat,
        sr=sr,
        continuous_attributes=continuous,
        discrete_attributes=discrete,
        attribute_kinds=kinds,
        discrete_num_classes=disc_classes,
        version=1,
        split={
            "train_only": FLAGS.train_only,
            "percent": FLAGS.split_percent,
            "seed": FLAGS.split_seed,
        },
        precompute={
            "n_signal": FLAGS.n_signal,
            "n_bands": FLAGS.n_bands,
            "ratios": ratios,
            "continuous_attributes": continuous,
            "discrete_attributes": discrete,
            "nb_bins": FLAGS.nb_bins,
            "bin_method": FLAGS.bin_method,
        },
        providers={
            "continuous": "AudioDescriptorProvider",
            "discrete": "SidecarAttributeProvider",
        },
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    app.run(main)
