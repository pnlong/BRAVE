"""
Attribute preprocessing for Fader RAVE training.

Pipeline (offline → online)
---------------------------
1. precompute_descriptors.py scans train split via AttributeLoader
2. Writes attribute_stats.yaml: min/max (decoder), quantile bins (latent CE)
3. FaderRAVE.load_attribute_stats_from_file() loads stats into buffers
4. _prepare_attributes() on each batch:
     continuous → normalize to [-1,1] (decoder) + quantify bins (latent CE)
     discrete   → index→float (decoder) + native index (latent CE)

Mirrors neurorave attr_dataset.py and faderave.py quantify().
See neurorave: raving_fader/datasets/attr_dataset.py, models/fader/faderave.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import warnings

from .audio_descriptors.features import compute_all


# --- Attribute naming / ordering ---


def ordered_attributes(
    continuous: Sequence[str],
    discrete: Sequence[str],
) -> List[str]:
    """Canonical concat order: continuous first, then discrete."""
    return list(continuous) + list(discrete)


def resolve_attribute_config(
    continuous_attributes: Sequence[str] = (),
    discrete_attributes: Sequence[str] = (),
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """Resolve attribute lists from gin continuous/discrete bindings."""
    cont = list(continuous_attributes)
    disc = list(discrete_attributes)
    names = ordered_attributes(cont, disc)
    kinds = {n: "continuous" for n in cont}
    kinds.update({n: "discrete" for n in disc})
    return cont, disc, names, kinds


# --- Discrete ↔ decoder float conversion ---


def discrete_index_to_decoder_float(
    indices: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Map class index to [-1, 1] for decoder concat (uniform spacing over classes)."""
    if num_classes <= 1:
        return torch.zeros_like(indices, dtype=torch.float32)
    idx = indices.float().clamp(0, num_classes - 1)
    return 2.0 * (idx / (num_classes - 1)) - 1.0


def decoder_float_to_discrete_index(
    values: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Inverse of discrete_index_to_decoder_float."""
    if num_classes <= 1:
        return torch.zeros_like(values, dtype=torch.long)
    norm = (values + 1.0) / 2.0
    idx = (norm * (num_classes - 1)).round().long()
    return idx.clamp(0, num_classes - 1)


def denormalize_continuous(
    attr_norm: torch.Tensor,
    attribute_name: str,
    min_max_features: Dict[str, Tuple[float, float]],
) -> torch.Tensor:
    """Invert min/max normalization for one continuous attribute channel."""
    lo, hi = min_max_features[attribute_name]
    return (attr_norm + 1.0) / 2.0 * (hi - lo + 1e-8) + lo


def stats_file_candidates(db_path: Union[str, Path]) -> List[Path]:
    """Search order for attribute stats next to LMDB (single canonical filename)."""
    root = Path(db_path)
    return [root / "attribute_stats.yaml"]


def resolve_stats_path(
    db_path: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[Path]:
    """Resolve stats YAML: explicit flag wins, else search db_path candidates."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    if db_path:
        for candidate in stats_file_candidates(db_path):
            if candidate.is_file():
                return candidate
    return None


def latent_length_from_config(
    n_signal: int,
    n_bands: int,
    ratios: Sequence[int],
) -> int:
    """Temporal latent frames T_lat = n_signal / (pqmf_bands * prod(ratios))."""
    compression = n_bands * int(np.prod(ratios))
    if n_signal % compression != 0:
        raise ValueError(
            f"n_signal={n_signal} must be divisible by {compression} "
            f"(n_bands={n_bands}, ratios={list(ratios)})"
        )
    return n_signal // compression


def compute_descriptor_matrix(
    audio_1d: np.ndarray,
    sr: int,
    descriptors: Sequence[str],
    latent_length: int,
) -> np.ndarray:
    """
    Extract D descriptor trajectories resampled to latent_length.

    Args:
        audio_1d: mono waveform (samples,)
        sr: sample rate
        descriptors: names e.g. centroid, rms, sharpness
        latent_length: T_lat target length

    Returns:
        (D, T_lat) float32 raw values (not normalized)
    """
    # --- Run librosa + timbral extractors; resample each series to T_lat ---
    features = compute_all(
        audio_1d.astype(np.float64),
        sr=sr,
        descriptors=list(descriptors),
        mean=False,
        resample=latent_length,
    )
    missing = [n for n in descriptors if n not in features]
    if missing:
        warnings.warn(
            f"Audio descriptors missing or failed extraction: {missing}. "
            "Using zero trajectories for those rows.",
            stacklevel=2,
        )

    rows = []
    for name in descriptors:
        if name not in features:
            rows.append(np.zeros(latent_length, dtype=np.float32))
            continue
        row = features[name]
        if row.ndim > 1:
            row = row[0]
        rows.append(row.astype(np.float32))
    return np.stack(rows, axis=0)


def normalize_descriptor(
    array: np.ndarray,
    min_max: Tuple[float, float],
) -> np.ndarray:
    """Map values to [-1, 1] using per-descriptor dataset min/max. See attr_dataset.normalize."""
    lo, hi = min_max
    if hi <= lo:
        return np.zeros_like(array)
    return 2.0 * ((array - lo) / (hi - lo) - 0.5)


def normalize_attributes(
    attr: torch.Tensor,
    descriptors: Sequence[str],
    min_max_features: Dict[str, Tuple[float, float]],
) -> torch.Tensor:
    """
    Normalize batch of raw attributes for decoder concat.

    Used by AttributeStream and inference helpers. Training uses
    FaderRAVE._prepare_attributes (handles discrete rows too).

    Args:
        attr: (B, D, T_lat) raw continuous
        min_max_features: per-descriptor (min, max)

    Returns:
        (B, D, T_lat) in [-1, 1]
    """
    out = attr.clone()
    for i, descr in enumerate(descriptors):
        lo, hi = min_max_features[descr]
        out[:, i] = 2.0 * ((out[:, i] - lo) / (hi - lo + 1e-8) - 0.5)
    return out


# --- Offline quantile binning (precompute_descriptors.py) ---


def compute_bins(
    allfeatures: np.ndarray,
    descriptors: Sequence[str],
    nb_bins: int,
    bin_method: str = "quantile",
    rms_gate_db: float = -40.0,
) -> np.ndarray:
    """
    Equal-density quantile bin edges per descriptor (offline, train split).

    bin_method: 'quantile' (default) or 'rms_gate' (skip silent frames on rms row).
    """
    all_values = []
    for i, descr in enumerate(descriptors):
        # --- Flatten all train frames for this descriptor ---
        data = allfeatures[:, i, :].flatten().copy()
        if bin_method == "rms_gate" and descr == "rms":
            # --- Paper-style: drop near-silent frames before binning ---
            gate = 10 ** (rms_gate_db / 20.0)
            data = data[data >= gate]
        if len(data) == 0:
            data = np.zeros(1, dtype=np.float32)
        data[data < 0] = 0
        data.sort()
        # --- Pick nb_bins evenly spaced quantile points as bin edges ---
        index = np.linspace(0, len(data) - 2, nb_bins).astype(int)
        values = [data[j] for j in index]
        all_values.append(values)
    return np.array(all_values, dtype=np.float32)


# --- Online bucketize (FaderRAVE._prepare_attributes, continuous rows) ---


def quantify(
    attr: torch.Tensor,
    bin_values: torch.Tensor,
) -> torch.Tensor:
    """
    Bucketize raw continuous attributes into class indices for latent CE.

    See neurorave faderave.py quantify(). Uses bin edges bins[i, 1:].

    Args:
        attr: (B, D, T_lat) raw (pre-normalize)
        bin_values: (D, nb_bins)

    Returns:
        (B, D, T_lat) long class indices
    """
    nz = attr.shape[-1]
    allarr_cls = torch.zeros_like(attr, dtype=torch.long)
    for i in range(attr.shape[1]):
        data = attr[:, i, :].reshape(-1)
        # --- bucketize against edges after the first quantile point ---
        edges = bin_values[i, 1:]
        data_cls = torch.bucketize(data, edges, right=False)
        allarr_cls[:, i, :] = data_cls.reshape(-1, nz)
    return allarr_cls


# --- Latent discriminator losses + W&B metrics ---


def attribute_classification_loss(
    attr_cls_pred: Union[torch.Tensor, List[torch.Tensor]],
    attr_cls: torch.Tensor,
    num_classes_per_attribute: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Cross-entropy for latent discriminator (+CE in lat_dis_step, -CE on gen).

    Args:
        attr_cls_pred: (B, D, C, T_lat) stacked OR list of (B, C_i, T_lat) per head
        attr_cls: (B, D, T_lat) integer targets
        num_classes_per_attribute: optional per-head class counts

    Returns:
        scalar CE loss (mean over attributes)
    """
    if isinstance(attr_cls_pred, list):
        # --- Per-head CE when heads have different num_classes ---
        losses = []
        for i, pred in enumerate(attr_cls_pred):
            target = attr_cls[:, i, :]
            losses.append(F.cross_entropy(pred, target))
        return sum(losses) / max(len(losses), 1)

    b, d, c, t = attr_cls_pred.shape
    # --- Stacked logits path (legacy equal-class heads) ---
    pred = attr_cls_pred.reshape(b * d, c, t)
    target = attr_cls.reshape(b * d, t)
    return F.cross_entropy(pred, target)


def per_attribute_ce_losses(
    attr_cls_pred: Union[torch.Tensor, List[torch.Tensor]],
    attr_cls: torch.Tensor,
    attribute_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Per-attribute CE for W&B logging."""
    out: Dict[str, torch.Tensor] = {}
    if isinstance(attr_cls_pred, list):
        for i, name in enumerate(attribute_names):
            if i >= len(attr_cls_pred):
                break
            out[name] = F.cross_entropy(
                attr_cls_pred[i], attr_cls[:, i, :], reduction="mean")
    else:
        b, d, c, t = attr_cls_pred.shape
        for i, name in enumerate(attribute_names):
            if i >= d:
                break
            pred = attr_cls_pred[:, i, :, :]
            target = attr_cls[:, i, :]
            out[name] = F.cross_entropy(pred, target, reduction="mean")
    return out


def per_attribute_accuracies(
    attr_cls_pred: Union[torch.Tensor, List[torch.Tensor]],
    attr_cls: torch.Tensor,
    attribute_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Per-attribute argmax accuracy for W&B logging."""
    out: Dict[str, torch.Tensor] = {}
    if isinstance(attr_cls_pred, list):
        for i, name in enumerate(attribute_names):
            if i >= len(attr_cls_pred):
                break
            pred_cls = attr_cls_pred[i].detach().argmax(dim=1)
            acc = (pred_cls == attr_cls[:, i, :]).float().mean()
            out[name] = acc
    else:
        for i, name in enumerate(attribute_names):
            if i >= attr_cls_pred.shape[1]:
                break
            pred_cls = attr_cls_pred[:, i, :, :].detach().argmax(dim=1)
            acc = (pred_cls == attr_cls[:, i, :]).float().mean()
            out[name] = acc
    return out


# --- attribute_stats.yaml I/O ---


def min_max_from_features(
    allfeatures: np.ndarray,
    descriptors: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    """Per-descriptor global min/max over (N, T_lat)."""
    stats = {}
    for i, descr in enumerate(descriptors):
        stats[descr] = (
            float(np.min(allfeatures[:, i, :])),
            float(np.max(allfeatures[:, i, :])),
        )
    return stats


def save_attribute_stats(
    path: Union[str, Path],
    attribute_names: Sequence[str],
    min_max_features: Dict[str, Tuple[float, float]],
    bin_values: np.ndarray,
    nb_bins: int,
    latent_length: int,
    sr: int,
    continuous_attributes: Optional[Sequence[str]] = None,
    discrete_attributes: Optional[Sequence[str]] = None,
    attribute_kinds: Optional[Dict[str, str]] = None,
    discrete_num_classes: Optional[Dict[str, int]] = None,
    discrete_class_labels: Optional[Dict[str, Sequence[str]]] = None,
    version: int = 1,
    split: Optional[Dict] = None,
    precompute: Optional[Dict] = None,
    providers: Optional[Dict] = None,
) -> None:
    """
    Write attribute_stats.yaml next to LMDB.

    Required for training: min_max_features (decoder), bin_values (latent CE).
    Metadata (split, precompute, providers) aids reproducibility checks.
    """
    path = Path(path)
    cont = list(continuous_attributes or [])
    disc = list(discrete_attributes or [])
    if not cont and not disc:
        cont = [n for n in attribute_names
                if (attribute_kinds or {}).get(n, "continuous") == "continuous"]
        disc = [n for n in attribute_names
                if (attribute_kinds or {}).get(n) == "discrete"]
    payload = {
        "version": version,
        "continuous_attributes": cont,
        "discrete_attributes": disc,
        "attribute_names": list(attribute_names),
        "nb_bins": nb_bins,
        "latent_length": latent_length,
        "sr": sr,
        "min_max_features": {k: list(v) for k, v in min_max_features.items()},
        "bin_values": bin_values.tolist(),
    }
    if attribute_kinds:
        payload["attribute_kinds"] = dict(attribute_kinds)
    if discrete_num_classes:
        payload["discrete_num_classes"] = dict(discrete_num_classes)
    if discrete_class_labels:
        payload["discrete_class_labels"] = {
            k: list(v) for k, v in discrete_class_labels.items()
        }
    if split:
        payload["split"] = split
    if precompute:
        payload["precompute"] = precompute
    if providers:
        payload["providers"] = providers
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False)


def load_attribute_stats(path: Union[str, Path]) -> Dict:
    """Load attribute_stats.yaml; fill missing keys for older partial files."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if "continuous_attributes" not in data:
        data["continuous_attributes"] = []
    if "discrete_attributes" not in data:
        data["discrete_attributes"] = []
    if "attribute_names" not in data:
        data["attribute_names"] = ordered_attributes(
            data["continuous_attributes"], data["discrete_attributes"])
    data["min_max_features"] = {
        k: tuple(v) for k, v in data["min_max_features"].items()
    }
    data["bin_values"] = np.array(data["bin_values"], dtype=np.float32)
    if "attribute_kinds" not in data:
        data["attribute_kinds"] = {
            n: "continuous" for n in data["continuous_attributes"]
        }
        data["attribute_kinds"].update(
            {n: "discrete" for n in data["discrete_attributes"]})
    data["discrete_num_classes"] = data.get("discrete_num_classes") or {}
    data["discrete_class_labels"] = data.get("discrete_class_labels") or {}
    return data


def validate_stats_against_config(
    stats: Dict,
    continuous_attributes: Sequence[str],
    discrete_attributes: Sequence[str],
    n_signal: Optional[int] = None,
) -> None:
    """Warn on version / attribute list mismatch vs current gin."""
    expected = ordered_attributes(continuous_attributes, discrete_attributes)
    loaded = stats.get("attribute_names", [])
    if list(loaded) != list(expected):
        warnings.warn(
            f"Stats attribute list {loaded} != gin config {expected}",
            stacklevel=2,
        )
    if n_signal and stats.get("precompute", {}).get("n_signal") not in (None, n_signal):
        warnings.warn(
            f"Stats n_signal {stats['precompute'].get('n_signal')} != {n_signal}",
            stacklevel=2,
        )


def validate_discrete_sidecar(
    db_path: str,
    attribute_names: Sequence[str],
    discrete_attributes: Sequence[str],
    num_classes_per_attribute: Sequence[int],
) -> None:
    """Fail fast when sidecar class indices exceed latent-disc head sizes."""
    sidecar_path = Path(db_path) / "attribute_sidecar.yaml"
    if not sidecar_path.is_file():
        return
    with open(sidecar_path, "r") as f:
        data = yaml.safe_load(f) or {}
    schema = data.get("attributes", {})
    for name in discrete_attributes:
        if name not in schema or name not in attribute_names:
            continue
        row = attribute_names.index(name)
        n_cls = int(num_classes_per_attribute[row])
        values = schema[name].get("values", {})
        if not values:
            continue
        max_val = max(int(v) for v in values.values())
        if max_val >= n_cls:
            raise ValueError(
                f"{sidecar_path}: '{name}' contains class {max_val}, but the "
                f"latent discriminator only has {n_cls} classes (valid: 0..{n_cls - 1}). "
                f"Rebuild the sidecar with texture_only (default) or raise "
                f"NUM_TEXTURE_CLASSES in brave_fader_texture.gin."
            )
