"""
Evaluate FaderRAVE conditioning: decode with modified attribute trajectories.

For each input clip, ablates one continuous attribute (--attr) while keeping
content latent z and all other attributes from the input. Writes:
  - PNG of the ablated attribute curve (raw + normalized)
  - WAV reconstruction for several variants

Usage (from BRAVE root):
  micromamba activate brave
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

  python RAVE/scripts/eval_fader_attribute.py \\
    --model "${BRAVE_STORAGE}/tabla_ismir21/runs/tabla_fader_run_XXXXX" \\
    --db_path "${BRAVE_STORAGE}/tabla_ismir21/preprocessed" \\
    --attr=rms \\
    --output artifacts/eval_fader_rms \\
    --max_samples 4

  python RAVE/scripts/eval_fader_attribute.py \\
    --model runs/my_fader_run \\
    --db_path /path/to/lmdb \\
    --attr=centroid \\
    --input path/to/a.wav --input path/to/b.wav \\
    --output artifacts/eval_fader_centroid
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import gin
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from absl import app, flags

import rave
import rave.core
from rave.fader.attributes import resolve_stats_path
from rave.fader.providers import AudioDescriptorProvider

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "FaderRAVE run directory or .ckpt", required=True)
flags.DEFINE_string("attr", None, "Continuous attribute to ablate (e.g. rms, centroid)", required=True)
flags.DEFINE_string("db_path", None, "LMDB path (for attribute_stats.yaml + val samples)")
flags.DEFINE_multi_string("input", [], "Optional WAV paths (instead of LMDB val samples)")
flags.DEFINE_string("output", None, "Output directory (default: eval_fader_<attr>)")
flags.DEFINE_integer("max_samples", 4, "Max clips to process (LMDB val mode)")
flags.DEFINE_integer("n_signal", 131072, "Crop length in samples")
flags.DEFINE_bool("gpu", False, "Use CUDA")


@dataclass
class AttrVariant:
    name: str
    description: str
    modify: Callable[[np.ndarray, Dict[str, Tuple[float, float]]], np.ndarray]


def _load_fader(model_path: str, db_path: Optional[str], device: torch.device):
    from rave.fader.model import FaderRAVE

    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        raise FileNotFoundError(f"config.gin not found near {model_path}")
    gin.parse_config_file(config_path)
    run = rave.core.search_for_run(model_path)
    if run is None:
        raise FileNotFoundError(f"checkpoint not found near {model_path}")

    model = FaderRAVE().load_from_checkpoint(run)
    stats_path = resolve_stats_path(db_path)
    if stats_path is None:
        raise FileNotFoundError(
            "attribute_stats.yaml not found; pass --db_path with precomputed stats"
        )
    model.load_attribute_stats_from_file(stats_path)
    model.eval()
    model.to(device)
    return model


def _latent_time_axis(model, t_lat: int) -> np.ndarray:
    """Seconds per latent frame."""
    cr = rave.core.get_minimum_size(model)
    hop = cr / float(model.sr)
    return np.arange(t_lat) * hop


def _extract_raw_attributes(
    model,
    mono: np.ndarray,
    t_lat: int,
) -> np.ndarray:
    """(D, T_lat) raw attribute matrix (continuous + discrete rows)."""
    provider = AudioDescriptorProvider(
        continuous_attributes=model.continuous_attributes,
        sampling_rate=model.sr,
    )
    raw_cont = provider.load(0, mono, model.sr, t_lat)
    parts = [raw_cont]
    if model.discrete_attributes:
        disc = np.zeros((len(model.discrete_attributes), t_lat), dtype=np.float32)
        parts.append(disc)
    return np.concatenate(parts, axis=0)


def _raw_to_attr_norm(model, raw: np.ndarray, device: torch.device) -> torch.Tensor:
    raw_t = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0).to(device)
    attr_norm, _ = model._prepare_attributes(raw_t)
    return attr_norm


def _attr_row_index(model, attr: str) -> int:
    if attr not in model.attribute_names:
        raise ValueError(
            f"Unknown attribute {attr!r}; model has {model.attribute_names}"
        )
    return model.attribute_names.index(attr)


def _validate_attr(model, attr: str) -> None:
    if attr not in model.attribute_names:
        raise ValueError(
            f"Unknown attribute {attr!r}; model has {model.attribute_names}"
        )
    if model.attribute_kinds.get(attr) != "continuous":
        raise ValueError(
            f"Attribute {attr!r} is not continuous; "
            f"kind={model.attribute_kinds.get(attr)}"
        )


def _default_variants(attr: str) -> List[AttrVariant]:
    label = attr

    def extracted(row: np.ndarray, stats: Dict) -> np.ndarray:
        return row.copy()

    def scale(f: float):
        def fn(row: np.ndarray, stats: Dict) -> np.ndarray:
            out = row.copy()
            out *= f
            return out

        return fn

    def constant_quantile(q: float):
        def fn(row: np.ndarray, stats: Dict) -> np.ndarray:
            lo, hi = stats[attr]
            val = lo + q * (hi - lo)
            return np.full_like(row, val)

        return fn

    def smooth(row: np.ndarray, stats: Dict) -> np.ndarray:
        win = max(3, row.shape[-1] // 32) | 1
        kernel = np.ones(win, dtype=np.float32) / win
        return np.convolve(row, kernel, mode="same").astype(np.float32)

    def ramp_up(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        return np.linspace(lo, hi, row.shape[-1], dtype=np.float32)

    def ramp_down(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        return np.linspace(hi, lo, row.shape[-1], dtype=np.float32)

    return [
        AttrVariant("01_extracted", f"{label} from input (baseline)", extracted),
        AttrVariant(f"02_{attr}_x0.5", f"{label} × 0.5", scale(0.5)),
        AttrVariant(f"03_{attr}_x2.0", f"{label} × 2.0", scale(2.0)),
        AttrVariant(
            f"04_{attr}_flat_median",
            f"constant {label} = median of input",
            lambda r, s: np.full_like(r, np.median(r)),
        ),
        AttrVariant(
            f"05_{attr}_flat_low",
            f"constant {label} = train min (via stats)",
            constant_quantile(0.0),
        ),
        AttrVariant(
            f"06_{attr}_flat_high",
            f"constant {label} = train max (via stats)",
            constant_quantile(1.0),
        ),
        AttrVariant(f"07_{attr}_smooth", f"smoothed {label} envelope", smooth),
        AttrVariant(
            f"08_{attr}_ramp_up",
            f"linear {label} ramp: train min → train max (stats)",
            ramp_up,
        ),
        AttrVariant(
            f"09_{attr}_ramp_down",
            f"linear {label} ramp: train max → train min (stats)",
            ramp_down,
        ),
    ]


def _plot_attr_curve(
    path: Path,
    times: np.ndarray,
    raw_row: np.ndarray,
    attr_norm_row: np.ndarray,
    attr: str,
    title: str,
    description: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(times, raw_row, color="steelblue", linewidth=1.2)
    axes[0].set_ylabel(f"raw {attr}")
    axes[0].set_title(f"{title}\n{description}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, attr_norm_row, color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("attr_norm (decoder)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_wav_mono(path: Path, sr: int, n_signal: int, device: torch.device) -> torch.Tensor:
    import soundfile as sf

    data, file_sr = sf.read(str(path), always_2d=True)
    x = torch.from_numpy(data.T).float()
    if file_sr != sr:
        x = torchaudio.functional.resample(x, file_sr, sr)
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    if x.shape[-1] < n_signal:
        reps = int(np.ceil(n_signal / x.shape[-1]))
        x = x.repeat(1, reps)[:, :n_signal]
    else:
        start = (x.shape[-1] - n_signal) // 2
        x = x[:, start : start + n_signal]
    return x.to(device)


def _iter_samples(
    model,
    device: torch.device,
) -> List[Tuple[str, torch.Tensor]]:
    """Return list of (stem, [1,C,T] audio on device)."""
    if FLAGS.input:
        out = []
        for p in FLAGS.input:
            path = Path(p)
            x = _load_wav_mono(path, model.sr, FLAGS.n_signal, device)
            out.append((path.stem, x.unsqueeze(0)))
        return out

    if not FLAGS.db_path:
        raise ValueError("Pass --input WAV(s) or --db_path for LMDB val samples")

    from rave.fader.dataset import wrap_fader_dataset
    from torch.utils.data import DataLoader

    dataset = rave.dataset.get_dataset(
        FLAGS.db_path,
        model.sr,
        FLAGS.n_signal,
        n_channels=model.n_channels,
    )
    _, val = rave.dataset.split_dataset(dataset, 98)
    val = wrap_fader_dataset(
        val,
        sampling_rate=model.sr,
        n_signal=FLAGS.n_signal,
        db_path=FLAGS.db_path,
    )
    loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    samples = []
    for i, (x_raw, _) in enumerate(loader):
        if i >= FLAGS.max_samples:
            break
        samples.append((f"val_{i:04d}", x_raw.to(device)))
    return samples


@torch.no_grad()
def _process_sample(
    model,
    attr: str,
    attr_idx: int,
    stem: str,
    x_raw: torch.Tensor,
    out_root: Path,
    variants: List[AttrVariant],
    device: torch.device,
) -> List[dict]:
    """x_raw: [1, C, T]"""
    import soundfile as sf

    z_mb = model.encode(x_raw, return_mb=False)
    z, _ = model.encoder.reparametrize(z_mb)[:2]

    t_lat = z.shape[-1]
    times = _latent_time_axis(model, t_lat)

    mono = x_raw[0].detach().cpu().numpy()
    if mono.ndim == 2:
        mono = mono.mean(axis=0)

    raw_full = _extract_raw_attributes(model, mono, t_lat)
    attr_base = raw_full[attr_idx].copy()

    min_max = {k: tuple(v) for k, v in model.min_max_features.items()}
    records = []

    sample_dir = out_root / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    for var in variants:
        raw_var = raw_full.copy()
        raw_var[attr_idx] = var.modify(attr_base, min_max)
        attr_norm = _raw_to_attr_norm(model, raw_var, device)

        y = model.decode(z, attr=attr_norm)
        y_np = y[0].detach().cpu().numpy()

        wav_path = sample_dir / f"{var.name}.wav"
        sf.write(str(wav_path), y_np.T, model.sr)

        raw_row = raw_var[attr_idx]
        norm_row = attr_norm[0, attr_idx].detach().cpu().numpy()
        png_path = sample_dir / f"{var.name}_{attr}.png"
        _plot_attr_curve(
            png_path,
            times,
            raw_row,
            norm_row,
            attr=attr,
            title=var.name,
            description=var.description,
        )

        records.append({
            "sample": stem,
            "attr": attr,
            "variant": var.name,
            "description": var.description,
            "wav": str(wav_path),
            "attr_plot": str(png_path),
            "raw_mean": float(raw_row.mean()),
            "raw_max": float(raw_row.max()),
        })

    return records


def main(argv):
    del argv
    device = torch.device("cuda" if FLAGS.gpu and torch.cuda.is_available() else "cpu")
    model = _load_fader(FLAGS.model, FLAGS.db_path, device)

    attr = FLAGS.attr
    _validate_attr(model, attr)
    attr_idx = _attr_row_index(model, attr)

    out_root = Path(FLAGS.output or f"eval_fader_{attr}")
    out_root.mkdir(parents=True, exist_ok=True)

    variants = _default_variants(attr)
    all_records = []
    for stem, x_raw in _iter_samples(model, device):
        print(f"Processing {stem} (ablating {attr})...")
        recs = _process_sample(
            model, attr, attr_idx, stem, x_raw, out_root, variants, device)
        all_records.extend(recs)

    manifest = {
        "model": FLAGS.model,
        "attr": attr,
        "db_path": FLAGS.db_path,
        "n_signal": FLAGS.n_signal,
        "continuous_attributes": list(model.continuous_attributes),
        "attribute_names": list(model.attribute_names),
        "variants": [v.name for v in variants],
        "results": all_records,
    }
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(all_records)} reconstructions under {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    app.run(main)
