"""
Evaluate FaderRAVE conditioning: decode with modified attribute trajectories.

Supports continuous ablation (--attr=rms, …) and discrete class sweeps
(--attr=texture_class, …). For each input clip, keeps content latent z and
all non-target attributes from the input (LMDB sidecar + descriptors when
--db_path is set). Writes WAV reconstructions, attribute curve PNGs, and
manifest.json.

Usage (from BRAVE root):
  micromamba activate brave
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

  # Continuous ablation (percussive / pitched / texture)
  python RAVE/scripts/eval_fader_attribute.py \\
    --model "${BRAVE_STORAGE}/tabla_ismir21/runs/tabla_fader_run_XXXXX" \\
    --db_path "${BRAVE_STORAGE}/tabla_ismir21/preprocessed" \\
    --attr=rms \\
    --output artifacts/eval_fader_rms \\
    --max_samples 4

  # Discrete texture_class sweep (FSD50K texture model)
  python RAVE/scripts/eval_fader_attribute.py \\
    --model runs/texture_fader \\
    --db_path /path/to/fsd50k/preprocessed \\
    --attr=texture_class \\
    --output artifacts/eval_fader_texture_class \\
    --max_samples 4

  # Arbitrary WAVs (continuous only; pass --discrete_baseline for texture)
  python RAVE/scripts/eval_fader_attribute.py \\
    --model runs/my_fader_run \\
    --attr=centroid \\
    --input path/to/a.wav --input path/to/b.wav \\
    --discrete_baseline 3 \\
    --output artifacts/eval_fader_centroid
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = Path(_RAVE_ROOT).parent
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
flags.DEFINE_string(
    "attr",
    None,
    "Attribute to ablate / sweep (continuous or discrete, e.g. rms, texture_class)",
    required=True,
)
flags.DEFINE_string("db_path", None, "LMDB path (for attribute_stats.yaml + val samples)")
flags.DEFINE_multi_string("input", [], "Optional WAV paths (instead of LMDB val samples)")
flags.DEFINE_string("output", None, "Output directory (default: eval_fader_<attr>)")
flags.DEFINE_integer("max_samples", 4, "Max clips to process (LMDB val mode)")
flags.DEFINE_integer("n_signal", 131072, "Crop length in samples")
flags.DEFINE_bool("gpu", False, "Use CUDA")
flags.DEFINE_integer(
    "discrete_baseline",
    -1,
    "Class index for discrete row when using --input WAVs (-1 = 0 with warning)",
)
flags.DEFINE_string(
    "class_names_yaml",
    None,
    "YAML with class_names map (default: FSD50K fader_texture_class_tags.yaml)",
)


@dataclass
class AttrVariant:
    name: str
    description: str
    modify: Callable[[np.ndarray, Dict], np.ndarray]


@dataclass
class EvalSample:
    stem: str
    x_raw: torch.Tensor  # [1, C, T]
    attr_raw: Optional[np.ndarray] = None  # (D, T_lat) from loader / sidecar


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


def _sanitize_stem(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "class"


def _default_class_names_yaml() -> Path:
    return (
        _BRAVE_ROOT
        / "dataset_exploration"
        / "fsd50k"
        / "configs"
        / "fader_texture_class_tags.yaml"
    )


def _load_class_names(
    attr: str,
    num_classes: int,
    yaml_path: Optional[str],
) -> Dict[int, str]:
    path = Path(yaml_path) if yaml_path else _default_class_names_yaml()
    names: Dict[int, str] = {i: str(i) for i in range(num_classes)}
    if not path.is_file():
        return names
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("class_names") or {}
    for key, label in raw.items():
        idx = int(key)
        if 0 <= idx < num_classes:
            names[idx] = str(label)
    return names


def _extract_continuous_attributes(
    model,
    mono: np.ndarray,
    t_lat: int,
) -> np.ndarray:
    """(D_cont, T_lat) raw continuous rows from audio descriptors."""
    provider = AudioDescriptorProvider(
        continuous_attributes=model.continuous_attributes,
        sampling_rate=model.sr,
    )
    return provider.load(0, mono, model.sr, t_lat)


def _build_raw_full(
    model,
    mono: np.ndarray,
    t_lat: int,
    attr_raw: Optional[np.ndarray],
) -> np.ndarray:
    """(D, T_lat) raw attribute matrix (continuous + discrete rows)."""
    if attr_raw is not None:
        out = np.asarray(attr_raw, dtype=np.float32)
        if out.shape[-1] != t_lat:
            raise ValueError(
                f"attr_raw T_lat={out.shape[-1]} != latent T_lat={t_lat}"
            )
        return out.copy()

    parts = []
    if model.continuous_attributes:
        parts.append(_extract_continuous_attributes(model, mono, t_lat))
    if model.discrete_attributes:
        baseline = FLAGS.discrete_baseline
        if baseline < 0:
            baseline = 0
            if model.discrete_attributes:
                print(
                    "Warning: --input WAV mode without sidecar; "
                    f"discrete rows default to class {baseline}. "
                    "Pass --discrete_baseline or --db_path for sidecar labels."
                )
        disc = np.full(
            (len(model.discrete_attributes), t_lat),
            float(baseline),
            dtype=np.float32,
        )
        parts.append(disc)
    if not parts:
        return np.zeros((0, t_lat), dtype=np.float32)
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


def _validate_attr(model, attr: str) -> str:
    if attr not in model.attribute_names:
        raise ValueError(
            f"Unknown attribute {attr!r}; model has {model.attribute_names}"
        )
    kind = model.attribute_kinds.get(attr)
    if kind not in ("continuous", "discrete"):
        raise ValueError(f"Attribute {attr!r} has unknown kind {kind!r}")
    return kind


def _num_discrete_classes(model, attr: str) -> int:
    return int(model.discrete_num_classes.get(attr, 2))


def _default_continuous_variants(attr: str) -> List[AttrVariant]:
    label = attr

    def extracted(row: np.ndarray, _stats: Dict) -> np.ndarray:
        return row.copy()

    def scale(f: float):
        def fn(row: np.ndarray, _stats: Dict) -> np.ndarray:
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

    def smooth(row: np.ndarray, _stats: Dict) -> np.ndarray:
        win = max(3, row.shape[-1] // 32) | 1
        kernel = np.ones(win, dtype=np.float32) / win
        return np.convolve(row, kernel, mode="same").astype(np.float32)

    def ramp_up(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        return np.linspace(lo, hi, row.shape[-1], dtype=np.float32)

    def ramp_down(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        return np.linspace(hi, lo, row.shape[-1], dtype=np.float32)

    def bell(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        x = np.linspace(-3.0, 3.0, row.shape[-1], dtype=np.float32)
        g = np.exp(-0.5 * x * x)
        g /= g.max()
        return (lo + g * (hi - lo)).astype(np.float32)

    def sine(row: np.ndarray, stats: Dict) -> np.ndarray:
        lo, hi = stats[attr]
        phase = np.linspace(0.0, 2.0 * np.pi, row.shape[-1], dtype=np.float32)
        return (lo + (np.sin(phase) + 1.0) * 0.5 * (hi - lo)).astype(np.float32)

    return [
        AttrVariant("01_extracted", f"{label} from input (baseline)", extracted),
        AttrVariant(f"02_{attr}_x0.5", f"{label} × 0.5", scale(0.5)),
        AttrVariant(f"03_{attr}_x2.0", f"{label} × 2.0", scale(2.0)),
        AttrVariant(
            f"04_{attr}_flat_median",
            f"constant {label} = median of input",
            lambda r, _s: np.full_like(r, np.median(r)),
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
        AttrVariant(
            f"10_{attr}_bell",
            f"Gaussian {label} bump: train min at edges, train max at center",
            bell,
        ),
        AttrVariant(
            f"11_{attr}_sine",
            f"one sine cycle of {label}: train min → max → min (stats range)",
            sine,
        ),
    ]


def _default_discrete_variants(
    attr: str,
    num_classes: int,
    class_names: Dict[int, str],
) -> List[AttrVariant]:
    variants: List[AttrVariant] = [
        AttrVariant(
            "01_extracted",
            f"{attr} from sidecar / baseline (ground truth label)",
            lambda row, _stats: row.copy(),
        ),
    ]

    for cls_idx in range(num_classes):
        label = class_names.get(cls_idx, str(cls_idx))
        safe = _sanitize_stem(label)
        name = f"{cls_idx + 2:02d}_class_{cls_idx:02d}_{safe}"

        def make_fixed(k: int):
            def fn(_row: np.ndarray, _stats: Dict) -> np.ndarray:
                return np.full_like(_row, float(k), dtype=np.float32)

            return fn

        variants.append(
            AttrVariant(
                name,
                f"fixed {attr}={cls_idx} ({label}); z + continuous attrs unchanged",
                make_fixed(cls_idx),
            )
        )

    def wrong_class(row: np.ndarray, stats: Dict) -> np.ndarray:
        n_cls = int(stats["num_classes"])
        extracted = int(round(float(row.flat[0])))
        extracted = max(0, min(extracted, n_cls - 1))
        wrong = (extracted + max(1, n_cls // 3)) % n_cls
        return np.full_like(row, float(wrong), dtype=np.float32)

    variants.append(
        AttrVariant(
            f"{num_classes + 2:02d}_wrong_class",
            f"{attr} offset from extracted label (sanity check)",
            wrong_class,
        )
    )
    return variants


def _plot_continuous_curve(
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


def _plot_discrete_curve(
    path: Path,
    times: np.ndarray,
    class_idx: int,
    class_name: str,
    attr_norm_value: float,
    attr: str,
    title: str,
    description: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].step(
        times,
        np.full_like(times, class_idx, dtype=np.float32),
        where="mid",
        color="steelblue",
        linewidth=1.5,
    )
    axes[0].set_ylabel(f"{attr} index")
    axes[0].set_title(f"{title}\n{description}\nclass {class_idx}: {class_name}")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(
        attr_norm_value,
        color="darkorange",
        linewidth=1.5,
        label=f"attr_norm = {attr_norm_value:.3f}",
    )
    axes[1].set_ylabel("attr_norm (decoder)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(loc="upper right")
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


def _iter_samples(model, device: torch.device) -> List[EvalSample]:
    if FLAGS.input:
        out = []
        for p in FLAGS.input:
            path = Path(p)
            x = _load_wav_mono(path, model.sr, FLAGS.n_signal, device)
            out.append(EvalSample(stem=path.stem, x_raw=x.unsqueeze(0)))
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
    for i, (x_raw, attr_raw) in enumerate(loader):
        if i >= FLAGS.max_samples:
            break
        attr_np = attr_raw[0].detach().cpu().numpy()
        samples.append(
            EvalSample(
                stem=f"val_{i:04d}",
                x_raw=x_raw.to(device),
                attr_raw=attr_np,
            )
        )
    return samples


@torch.no_grad()
def _process_continuous_sample(
    model,
    attr: str,
    attr_idx: int,
    sample: EvalSample,
    out_root: Path,
    variants: List[AttrVariant],
    device: torch.device,
) -> List[dict]:
    import soundfile as sf

    x_raw = sample.x_raw
    z_mb = model.encode(x_raw, return_mb=False)
    z, _ = model.encoder.reparametrize(z_mb)[:2]

    t_lat = z.shape[-1]
    times = _latent_time_axis(model, t_lat)

    mono = x_raw[0].detach().cpu().numpy()
    if mono.ndim == 2:
        mono = mono.mean(axis=0)

    raw_full = _build_raw_full(model, mono, t_lat, sample.attr_raw)
    attr_base = raw_full[attr_idx].copy()

    min_max = {k: tuple(v) for k, v in model.min_max_features.items()}
    records = []

    sample_dir = out_root / sample.stem
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
        _plot_continuous_curve(
            png_path,
            times,
            raw_row,
            norm_row,
            attr=attr,
            title=var.name,
            description=var.description,
        )

        records.append({
            "sample": sample.stem,
            "mode": "continuous",
            "attr": attr,
            "variant": var.name,
            "description": var.description,
            "wav": str(wav_path),
            "attr_plot": str(png_path),
            "raw_mean": float(raw_row.mean()),
            "raw_max": float(raw_row.max()),
        })

    return records


@torch.no_grad()
def _process_discrete_sample(
    model,
    attr: str,
    attr_idx: int,
    sample: EvalSample,
    out_root: Path,
    variants: List[AttrVariant],
    class_names: Dict[int, str],
    num_classes: int,
    device: torch.device,
) -> List[dict]:
    import soundfile as sf

    x_raw = sample.x_raw
    z_mb = model.encode(x_raw, return_mb=False)
    z, _ = model.encoder.reparametrize(z_mb)[:2]

    t_lat = z.shape[-1]
    times = _latent_time_axis(model, t_lat)

    mono = x_raw[0].detach().cpu().numpy()
    if mono.ndim == 2:
        mono = mono.mean(axis=0)

    raw_full = _build_raw_full(model, mono, t_lat, sample.attr_raw)
    attr_base = raw_full[attr_idx].copy()
    extracted_class = int(round(float(attr_base.flat[0])))

    stats = {"num_classes": num_classes}
    records = []

    sample_dir = out_root / sample.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    for var in variants:
        raw_var = raw_full.copy()
        raw_var[attr_idx] = var.modify(attr_base, stats)
        attr_norm = _raw_to_attr_norm(model, raw_var, device)

        y = model.decode(z, attr=attr_norm)
        y_np = y[0].detach().cpu().numpy()

        wav_path = sample_dir / f"{var.name}.wav"
        sf.write(str(wav_path), y_np.T, model.sr)

        class_idx = int(round(float(raw_var[attr_idx].flat[0])))
        class_idx = max(0, min(class_idx, num_classes - 1))
        norm_val = float(attr_norm[0, attr_idx].mean().item())
        class_name = class_names.get(class_idx, str(class_idx))
        png_path = sample_dir / f"{var.name}_{attr}.png"
        _plot_discrete_curve(
            png_path,
            times,
            class_idx,
            class_name,
            norm_val,
            attr=attr,
            title=var.name,
            description=var.description,
        )

        records.append({
            "sample": sample.stem,
            "mode": "discrete",
            "attr": attr,
            "variant": var.name,
            "description": var.description,
            "class_index": class_idx,
            "class_name": class_name,
            "extracted_class_index": extracted_class,
            "extracted_class_name": class_names.get(extracted_class, str(extracted_class)),
            "attr_norm": norm_val,
            "wav": str(wav_path),
            "attr_plot": str(png_path),
        })

    return records


def main(argv):
    del argv
    device = torch.device("cuda" if FLAGS.gpu and torch.cuda.is_available() else "cpu")
    model = _load_fader(FLAGS.model, FLAGS.db_path, device)

    attr = FLAGS.attr
    kind = _validate_attr(model, attr)
    attr_idx = _attr_row_index(model, attr)

    out_root = Path(FLAGS.output or f"eval_fader_{attr}")
    out_root.mkdir(parents=True, exist_ok=True)

    if kind == "continuous":
        variants = _default_continuous_variants(attr)
        class_names: Dict[int, str] = {}
        num_classes = 0
    else:
        num_classes = _num_discrete_classes(model, attr)
        class_names = _load_class_names(attr, num_classes, FLAGS.class_names_yaml)
        variants = _default_discrete_variants(attr, num_classes, class_names)

    all_records = []
    for sample in _iter_samples(model, device):
        print(f"Processing {sample.stem} ({kind} eval on {attr})...")
        if kind == "continuous":
            recs = _process_continuous_sample(
                model, attr, attr_idx, sample, out_root, variants, device)
        else:
            recs = _process_discrete_sample(
                model,
                attr,
                attr_idx,
                sample,
                out_root,
                variants,
                class_names,
                num_classes,
                device,
            )
        all_records.extend(recs)

    manifest = {
        "model": FLAGS.model,
        "attr": attr,
        "mode": kind,
        "db_path": FLAGS.db_path,
        "n_signal": FLAGS.n_signal,
        "continuous_attributes": list(model.continuous_attributes),
        "discrete_attributes": list(model.discrete_attributes),
        "attribute_names": list(model.attribute_names),
        "variants": [v.name for v in variants],
        "class_names": class_names if kind == "discrete" else None,
        "num_classes": num_classes if kind == "discrete" else None,
        "results": all_records,
    }
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(all_records)} reconstructions under {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    app.run(main)
