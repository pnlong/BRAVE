"""
Precompute attribute statistics (min/max, quantile bins) for Fader training.

Uses the same AttributeLoader as training. Writes attribute_stats.yaml.

Usage (from BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

  # Match a Fader training gin config (recommended):
  python RAVE/scripts/precompute_descriptors.py \\
    --db_path /path/to/lmdb \\
    --config configs/brave_fader_texture.gin

  # Or specify attributes manually:
  python RAVE/scripts/precompute_descriptors.py \\
    --db_path /path/to/lmdb \\
    --continuous_attributes=rms --continuous_attributes=flatness \\
    --discrete_attributes=texture_class
"""

from __future__ import annotations

import ast
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRAVE_ROOT = os.path.dirname(_RAVE_ROOT)
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import yaml
import numpy as np
import torch
import gin
from gin.config import ConfigurableReference
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
from rave.fader.discrete_class_labels import resolve_discrete_class_labels
from rave.fader.providers import build_attribute_loader

FLAGS = flags.FLAGS

DEFAULT_CONTINUOUS = ["centroid", "rms", "bandwidth", "sharpness", "booming"]

flags.DEFINE_string("db_path", None, "LMDB dataset path", required=True)
flags.DEFINE_multi_string(
    "config",
    None,
    "Fader training gin config(s); sets continuous/discrete attrs, N_BAND, RATIOS",
)
flags.DEFINE_multi_string(
    "override",
    [],
    "Gin binding overrides (same as train.py)",
)
flags.DEFINE_integer("n_signal", 131072, "Chunk length in samples")
flags.DEFINE_integer("n_bands", 0, "PQMF bands (0 = from gin or 16)")
flags.DEFINE_list("ratios", None, "Encoder ratios (default: gin or 2,2,2,1)")
flags.DEFINE_multi_string(
    "continuous_attributes",
    [],
    "Continuous attribute names (override --config when set)",
)
flags.DEFINE_multi_string(
    "discrete_attributes",
    [],
    "Discrete attribute names (override --config when set)",
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
flags.DEFINE_integer(
    "workers",
    0,
    "Chunk worker processes (0=all logical CPU cores; 1=serial)",
)
flags.DEFINE_bool("no_progress", False, "Disable progress bars")

_WORKER: Dict[str, Any] = {}


def _stats_path(db_path: str) -> str:
    return os.path.join(db_path, "attribute_stats.yaml")


def _add_gin_extension(name: str) -> str:
    return name if name.endswith(".gin") else f"{name}.gin"


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), _BRAVE_ROOT):
        candidate = os.path.join(base, path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(path)


def _ordered_config_files(config_paths: Sequence[str]) -> List[str]:
    """Resolve paths; prepend ``brave.gin`` for ``brave_fader*.gin`` includes."""
    ordered: List[str] = []
    seen: set[str] = set()
    for p in config_paths:
        rp = _resolve_config_path(_add_gin_extension(p))
        parent = os.path.dirname(rp)
        brave = os.path.join(parent, "brave.gin")
        candidates = [brave, rp] if (
            os.path.basename(rp).startswith("brave_fader") and os.path.isfile(brave)
        ) else [rp]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
    return ordered


def _parse_gin_macros(config_paths: Sequence[str]) -> Dict[str, Any]:
    """Read UPPER_CASE macro assignments (``N_BAND``, ``CONTINUOUS_ATTRIBUTES``, …)."""
    macros: Dict[str, Any] = {}
    for path in config_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line or line.endswith(":"):
                continue
            key, _, rhs = line.partition("=")
            key = key.strip()
            rhs = rhs.strip()
            if not key.isupper():
                continue
            if rhs.startswith("%"):
                continue
            try:
                macros[key] = ast.literal_eval(rhs)
            except (ValueError, SyntaxError):
                continue
    return macros


def _resolve_gin_value(value: Any, macros: Dict[str, Any]) -> Any:
    if isinstance(value, ConfigurableReference):
        ref = str(value).lstrip("%")
        if ref in macros:
            return macros[ref]
        raise ValueError(f"Unresolved gin macro reference: {value}")
    if isinstance(value, dict):
        return {k: _resolve_gin_value(v, macros) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolve_gin_value(v, macros) for v in value]
    return value


@dataclass(frozen=True)
class _GinTrainingSettings:
    continuous: List[str]
    discrete: List[str]
    n_bands: Optional[int]
    ratios: Optional[List[int]]
    discrete_num_classes: Dict[str, int]
    config_paths: List[str]


def _ensure_gin_imports() -> None:
    import cached_conv as cc  # noqa: F401
    import rave.blocks  # noqa: F401
    import rave.core  # noqa: F401
    import rave.discriminator  # noqa: F401
    import rave.fader.callbacks  # noqa: F401
    import rave.fader.dataset  # noqa: F401
    import rave.fader.latent_discriminator  # noqa: F401
    import rave.fader.model  # noqa: F401
    import rave.fader.providers  # noqa: F401
    import rave.pqmf  # noqa: F401
    import rave.training  # noqa: F401


def _load_gin_training_settings(
    config_paths: Sequence[str],
    overrides: Sequence[str],
) -> _GinTrainingSettings:
    _ensure_gin_imports()
    from rave.fader.providers.loader import build_attribute_loader

    gin.clear_config()
    resolved = _ordered_config_files(config_paths)
    for path in resolved:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Gin config not found: {path}")
    gin.parse_config_files_and_bindings(resolved, list(overrides))
    macros = _parse_gin_macros(resolved)

    # Gin resolves %CONTINUOUS_ATTRIBUTES at factory call time, not via query_parameter.
    probe_loader = build_attribute_loader(
        sampling_rate=int(macros.get("SAMPLING_RATE", 44100)),
        latent_length=64,
        db_path=os.path.devnull,
    )
    continuous = list(probe_loader.continuous_attributes)
    discrete = list(probe_loader.discrete_attributes)

    n_bands = macros.get("N_BAND")
    ratios = macros.get("RATIOS")

    disc_classes: Dict[str, int] = {}
    try:
        raw_dnc = gin.query_parameter(
            "rave.fader.model.FaderRAVE.discrete_num_classes") or {}
        for key, val in raw_dnc.items():
            disc_classes[str(key)] = int(_resolve_gin_value(val, macros))
    except ValueError:
        pass

    return _GinTrainingSettings(
        continuous=continuous,
        discrete=discrete,
        n_bands=int(n_bands) if n_bands is not None else None,
        ratios=[int(r) for r in ratios] if ratios is not None else None,
        discrete_num_classes=disc_classes,
        config_paths=resolved,
    )


def _resolve_precompute_settings() -> tuple[
    List[str], List[str], int, List[int], Dict[str, int], Optional[List[str]]
]:
    gin_settings = None
    if FLAGS.config:
        gin_settings = _load_gin_training_settings(FLAGS.config, FLAGS.override)

    continuous = (
        list(FLAGS.continuous_attributes)
        if FLAGS.continuous_attributes
        else (gin_settings.continuous if gin_settings else DEFAULT_CONTINUOUS)
    )
    discrete = (
        list(FLAGS.discrete_attributes)
        if FLAGS.discrete_attributes
        else (gin_settings.discrete if gin_settings else [])
    )

    if FLAGS.n_bands > 0:
        n_bands = FLAGS.n_bands
    elif gin_settings and gin_settings.n_bands is not None:
        n_bands = gin_settings.n_bands
    else:
        n_bands = 16

    if FLAGS.ratios is not None:
        ratios = [int(r) for r in FLAGS.ratios]
    elif gin_settings and gin_settings.ratios is not None:
        ratios = gin_settings.ratios
    else:
        ratios = [2, 2, 2, 1]

    gin_disc_classes = gin_settings.discrete_num_classes if gin_settings else {}
    config_paths = gin_settings.config_paths if gin_settings else None
    return continuous, discrete, n_bands, ratios, gin_disc_classes, config_paths


def _train_indices(n: int, percent: int, seed: int) -> np.ndarray:
    """Mirror rave.dataset.split_dataset index assignment."""
    split1 = max((percent * n) // 100, 1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return perm[:split1].numpy()


def _worker_count(workers: int) -> int:
    if workers == 1:
        return 1
    if workers > 0:
        return workers
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _lazy_index_payload(dataset) -> Optional[Dict[str, Any]]:
    items = getattr(dataset, "items", None)
    keys = getattr(dataset, "_keys", None)
    if items is None or keys is None:
        return None
    return {"keys": list(keys), "items": np.asarray(items).tolist()}


def _close_dataset_lmdb(dataset) -> None:
    """LMDB cannot be opened twice in one process (breaks forked/spawned workers)."""
    env = getattr(dataset, "_env", None)
    if env is not None:
        env.close()
        dataset._env = None


def _build_dataset(cfg: Dict[str, Any]):
    import rave.transforms as transforms

    base_transforms = transforms.Compose([
        lambda x: x.astype(np.float32),
        transforms.Dequantize(16),
    ])
    show_progress = cfg.get("show_progress", True)
    if cfg["lazy"]:
        kwargs = {"show_progress": show_progress}
        if cfg.get("lazy_index"):
            kwargs["lazy_index"] = cfg["lazy_index"]
        return rave.dataset.LazyAudioDataset(
            cfg["db_path"],
            cfg["n_signal"],
            cfg["sr"],
            base_transforms,
            cfg["channels"],
            **kwargs,
        )
    return rave.dataset.AudioDataset(
        cfg["db_path"],
        transforms=base_transforms,
        n_channels=cfg["channels"],
        show_progress=show_progress,
    )


def _build_attribute_loader(cfg: Dict[str, Any]):
    return build_attribute_loader(
        continuous_attributes=cfg["continuous"],
        discrete_attributes=cfg["discrete"],
        sampling_rate=cfg["sr"],
        latent_length=cfg["t_lat"],
        db_path=cfg["db_path"],
    )


def _bind_worker_state(cfg: Dict[str, Any], dataset, loader) -> None:
    global _WORKER
    _WORKER = {
        "cfg": cfg,
        "dataset": dataset,
        "loader": loader,
        "n_attrs": len(cfg["attribute_names"]),
        "t_lat": cfg["t_lat"],
    }


def _init_precompute_worker(cfg: Dict[str, Any]) -> None:
    _bind_worker_state(cfg, _build_dataset(cfg), _build_attribute_loader(cfg))


def _extract_chunk_features(index: int) -> np.ndarray:
    w = _WORKER
    cfg = w["cfg"]
    i = int(index)
    try:
        audio = w["dataset"][i]
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio.reshape(-1)
        n_signal = cfg["n_signal"]
        if len(mono) > n_signal:
            mono = mono[:n_signal]
        elif len(mono) < n_signal:
            mono = np.pad(mono, (0, n_signal - len(mono)))
        return w["loader"].load(i, mono if mono.ndim == 1 else mono, sr=cfg["sr"])
    except Exception as exc:
        print(f"chunk {i} failed: {exc}")
        return np.zeros((w["n_attrs"], w["t_lat"]), dtype=np.float32)


def _scan_chunks(
    indices: np.ndarray,
    worker_cfg: Dict[str, Any],
    *,
    dataset=None,
    loader=None,
    show_progress: bool = True,
) -> List[np.ndarray]:
    n_workers = _worker_count(FLAGS.workers)
    indices_list = [int(i) for i in indices]

    if n_workers == 1:
        if dataset is None or loader is None:
            _init_precompute_worker(worker_cfg)
        else:
            _bind_worker_state(worker_cfg, dataset, loader)
        row_iter = indices_list
        if show_progress:
            row_iter = tqdm(indices_list, desc="precompute", unit="chunk")
        return [_extract_chunk_features(i) for i in row_iter]

    chunksize = max(1, len(indices_list) // (n_workers * 8))
    print(
        f"Spawning {n_workers} workers (lazy decode + timbral attrs are slow per chunk)...",
        flush=True,
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_precompute_worker,
        initargs=(worker_cfg,),
    ) as pool:
        row_iter = pool.imap(_extract_chunk_features, indices_list, chunksize=chunksize)
        if show_progress:
            row_iter = tqdm(
                row_iter,
                total=len(indices_list),
                desc="precompute",
                unit="chunk",
            )
        return list(row_iter)


def main(argv):
    del argv
    db_path = FLAGS.db_path
    continuous, discrete, n_bands, ratios, gin_disc_classes, config_paths = (
        _resolve_precompute_settings())
    attribute_names = ordered_attributes(continuous, discrete)
    t_lat = latent_length_from_config(FLAGS.n_signal, n_bands, ratios)

    if config_paths:
        print(
            f"Config: {', '.join(config_paths)}\n"
            f"  continuous={continuous}\n"
            f"  discrete={discrete}\n"
            f"  n_bands={n_bands} ratios={ratios}"
        )
    else:
        print(
            f"Attributes: continuous={continuous} discrete={discrete} "
            f"n_bands={n_bands} ratios={ratios}"
        )

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
    show_progress = not FLAGS.no_progress
    print("Probing LMDB clip metadata...", flush=True)
    if lazy:
        dataset = rave.dataset.LazyAudioDataset(
            db_path,
            FLAGS.n_signal,
            sr,
            base_transforms,
            meta.get("channels", 1),
            show_progress=show_progress,
        )
    else:
        dataset = rave.dataset.AudioDataset(
            db_path,
            transforms=base_transforms,
            n_channels=meta.get("channels", 1),
            show_progress=show_progress,
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

    print(f"Computing attributes for {len(indices)} chunks, T_lat={t_lat}, D={len(attribute_names)}")

    worker_cfg = {
        "db_path": db_path,
        "lazy": lazy,
        "sr": sr,
        "channels": meta.get("channels", 1),
        "n_signal": FLAGS.n_signal,
        "continuous": continuous,
        "discrete": discrete,
        "t_lat": t_lat,
        "attribute_names": attribute_names,
        "show_progress": show_progress,
    }
    if lazy:
        lazy_index = _lazy_index_payload(dataset)
        if lazy_index is not None:
            worker_cfg["lazy_index"] = lazy_index

    n_workers = _worker_count(FLAGS.workers)
    serial_dataset = dataset
    serial_loader = None
    if n_workers == 1:
        serial_loader = _build_attribute_loader(worker_cfg)
    else:
        _close_dataset_lmdb(dataset)
        del dataset

    allfeatures = _scan_chunks(
        indices,
        worker_cfg,
        dataset=serial_dataset if n_workers == 1 else None,
        loader=serial_loader,
        show_progress=not FLAGS.no_progress,
    )

    if n_workers == 1:
        _close_dataset_lmdb(serial_dataset)

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
        idx = attribute_names.index(name)
        vals = allfeatures[:, idx, :].flatten()
        max_idx = int(vals.max()) if len(vals) else 0
        inferred = max(max_idx + 1, 1)
        gin_n = gin_disc_classes.get(name)
        if gin_n is not None:
            if max_idx >= int(gin_n):
                print(
                    f"WARNING: '{name}' sidecar max class {max_idx} >= gin "
                    f"NUM_TEXTURE_CLASSES={gin_n}; training will fail unless "
                    f"sidecar is rebuilt with --texture_only or gin is updated."
                )
            disc_classes[name] = int(gin_n)
        else:
            disc_classes[name] = inferred

    out_path = _stats_path(db_path)
    discrete_class_labels = resolve_discrete_class_labels(
        db_path,
        discrete,
        disc_classes,
    )
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
        discrete_class_labels=discrete_class_labels,
        version=1,
        split={
            "train_only": FLAGS.train_only,
            "percent": FLAGS.split_percent,
            "seed": FLAGS.split_seed,
        },
        precompute={
            "n_signal": FLAGS.n_signal,
            "n_bands": n_bands,
            "ratios": ratios,
            "continuous_attributes": continuous,
            "discrete_attributes": discrete,
            "nb_bins": FLAGS.nb_bins,
            "bin_method": FLAGS.bin_method,
            "config": config_paths,
            "override": list(FLAGS.override) or None,
        },
        providers={
            "continuous": "AudioDescriptorProvider",
            "discrete": "SidecarAttributeProvider",
        },
    )
    print(f"Wrote {out_path}")
    for name, labels in discrete_class_labels.items():
        print(f"  discrete_class_labels[{name}]: {labels}")


if __name__ == "__main__":
    app.run(main)
