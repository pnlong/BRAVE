"""Manifest and training profile helpers for canonicalizer Stage-1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..fader.attributes import load_attribute_stats, resolve_stats_path

# Continuous attrs excluded from legacy descriptor loss (pitch / discrete).
TAP_SAFE_DESCRIPTOR_ATTRS = frozenset({
    "rms",
    "centroid",
    "warmth",
    "bandwidth",
    "sharpness",
    "booming",
    "flatness",
    "roughness",
    "brightness",
    "reverb",
    "hardness",
    "depth",
})


def descriptor_loss_attributes(continuous_attributes: Sequence[str]) -> List[str]:
    return [a for a in continuous_attributes if a in TAP_SAFE_DESCRIPTOR_ATTRS]


@dataclass
class TrainingProfile:
    """Lightweight metadata for canonicalizer runs (stats optional)."""

    config_path: Path
    db_path: Path
    stats_path: Optional[Path] = None
    ood_db_path: Optional[Path] = None
    is_fader: bool = False
    continuous_attributes: List[str] = field(default_factory=list)
    discrete_attributes: List[str] = field(default_factory=list)
    attribute_names: List[str] = field(default_factory=list)
    descriptor_loss_attrs: List[str] = field(default_factory=list)
    latent_stats_path: Optional[Path] = None
    descriptor_means: Dict[str, float] = field(default_factory=dict)

    @property
    def stats_hash(self) -> str:
        if self.stats_path is None or not self.stats_path.is_file():
            return ""
        return hashlib.sha256(self.stats_path.read_bytes()).hexdigest()[:16]

    def descriptor_mean_vector(self):
        import torch
        return torch.tensor(
            [self.descriptor_means[a] for a in self.descriptor_loss_attrs],
            dtype=torch.float32,
        )


# Backward-compatible alias
DomainProfile = TrainingProfile


def _is_fader_config(config_path: Path) -> bool:
    text = config_path.read_text()
    return "FaderRAVE" in text or "fader.model" in text


def build_training_profile(
    config_path: Union[str, Path],
    db_path: Union[str, Path],
    *,
    stats_path: Optional[Union[str, Path]] = None,
    ood_db_path: Optional[Union[str, Path]] = None,
    require_stats: bool = False,
) -> TrainingProfile:
    config_path = Path(config_path)
    db_path = Path(db_path)
    is_fader = _is_fader_config(config_path)

    stats_file = resolve_stats_path(str(db_path), str(stats_path) if stats_path else None)
    if require_stats and stats_file is None:
        raise FileNotFoundError(f"attribute_stats.yaml not found for db_path={db_path}")
    if is_fader and stats_file is None:
        raise FileNotFoundError(
            f"Fader backbone requires attribute_stats.yaml for db_path={db_path}")

    cont: List[str] = []
    disc: List[str] = []
    names: List[str] = []
    desc_attrs: List[str] = []
    descriptor_means: Dict[str, float] = {}
    if stats_file is not None:
        stats = load_attribute_stats(stats_file)
        cont = list(stats.get("continuous_attributes", []))
        disc = list(stats.get("discrete_attributes", []))
        names = list(stats.get("attribute_names", cont + disc))
        desc_attrs = descriptor_loss_attributes(cont)
        mm = stats.get("min_max_features", {})
        descriptor_means = {
            a: 0.5 * (mm[a][0] + mm[a][1])
            for a in desc_attrs
            if a in mm
        }

    latent_path = latent_stats_cache_path(db_path)
    if not latent_path.is_file():
        latent_path = None

    return TrainingProfile(
        config_path=config_path,
        db_path=db_path,
        stats_path=Path(stats_file) if stats_file else None,
        ood_db_path=Path(ood_db_path) if ood_db_path else None,
        is_fader=is_fader,
        continuous_attributes=cont,
        discrete_attributes=disc,
        attribute_names=names,
        descriptor_loss_attrs=desc_attrs,
        latent_stats_path=latent_path,
        descriptor_means=descriptor_means,
    )


def build_domain_profile(
    config_path: Union[str, Path],
    db_path: Union[str, Path],
    stats_path: Optional[Union[str, Path]] = None,
) -> TrainingProfile:
    """Backward-compatible alias."""
    return build_training_profile(config_path, db_path, stats_path=stats_path)


def latent_stats_cache_path(db_path: Union[str, Path]) -> Path:
    return Path(db_path) / "latent_stats_canonicalizer.npz"


def load_latent_stats(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    data = np.load(str(path))
    return {"latent_mean": data["latent_mean"], "latent_std": data.get("latent_std")}


@dataclass
class CanonicalizerManifest:
    canonicalizer_type: str
    backbone_config: str
    backbone_ckpt: str
    db_path: str
    ood_db_path: str = ""
    use_reverb: bool = True
    stats_hash: str = ""
    backbone_kind: str = "RAVE"

    def to_dict(self) -> dict:
        out = {
            "canonicalizer_type": self.canonicalizer_type,
            "backbone_config": self.backbone_config,
            "backbone_ckpt": self.backbone_ckpt,
            "db_path": self.db_path,
            "use_reverb": self.use_reverb,
            "stats_hash": self.stats_hash,
            "backbone_kind": self.backbone_kind,
        }
        if self.ood_db_path:
            out["ood_db_path"] = self.ood_db_path
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalizerManifest":
        return cls(
            canonicalizer_type=data["canonicalizer_type"],
            backbone_config=data["backbone_config"],
            backbone_ckpt=data["backbone_ckpt"],
            db_path=data["db_path"],
            ood_db_path=data.get("ood_db_path", ""),
            use_reverb=bool(data.get("use_reverb", True)),
            stats_hash=data.get("stats_hash", ""),
            backbone_kind=data.get("backbone_kind", "RAVE"),
        )


def save_canonicalizer_checkpoint(
    path: Union[str, Path],
    state_dict: dict,
    manifest: CanonicalizerManifest,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest.to_dict(), indent=2))
    torch_save = __import__("torch")
    torch_save.save({"state_dict": state_dict, "manifest": manifest.to_dict()}, path)


def load_canonicalizer_checkpoint(
    path: Union[str, Path],
) -> tuple[dict, CanonicalizerManifest]:
    import torch

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "manifest" in payload:
        manifest = CanonicalizerManifest.from_dict(payload["manifest"])
        return payload["state_dict"], manifest
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.is_file():
        manifest = CanonicalizerManifest.from_dict(json.loads(sidecar.read_text()))
        if isinstance(payload, dict) and "state_dict" in payload:
            return payload["state_dict"], manifest
        return payload, manifest
    raise ValueError(f"No manifest found for canonicalizer checkpoint: {path}")


def attach_canonicalizer_to_model(
    model,
    state_dict: dict,
    canonicalizer_type: str,
) -> None:
    from .backbone import attach_canonicalizer_modules

    attach_canonicalizer_modules(model, state_dict, canonicalizer_type)


def load_canonicalizer_onto_model(
    model,
    ckpt_path: Union[str, Path],
) -> CanonicalizerManifest:
    state, manifest = load_canonicalizer_checkpoint(ckpt_path)
    attach_canonicalizer_to_model(model, state, manifest.canonicalizer_type)
    return manifest


def validate_manifest(
    manifest: CanonicalizerManifest,
    *,
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    db_path: Union[str, Path],
    strict: bool = True,
) -> None:
    errors = []
    if Path(config_path).resolve() != Path(manifest.backbone_config).resolve():
        if str(config_path) != manifest.backbone_config:
            errors.append(
                f"config mismatch: {config_path} vs {manifest.backbone_config}")
    if Path(ckpt_path).resolve() != Path(manifest.backbone_ckpt).resolve():
        if str(ckpt_path) != manifest.backbone_ckpt:
            errors.append(
                f"ckpt mismatch: {ckpt_path} vs {manifest.backbone_ckpt}")
    if Path(db_path).resolve() != Path(manifest.db_path).resolve():
        if str(db_path) != manifest.db_path:
            errors.append(f"db_path mismatch: {db_path} vs {manifest.db_path}")
    if errors:
        msg = "Canonicalizer manifest mismatch:\n" + "\n".join(errors)
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)
