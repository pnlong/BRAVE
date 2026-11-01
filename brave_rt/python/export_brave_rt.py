#!/usr/bin/env python3
"""Export BRAVE / CycleGAN checkpoints to a BraveRT bundle.

Writes ``brave_rt.json``, ``weights.h5``, ``npy_weights/``, ``host_controls.json``
(STAGE_07 schema), and for CycleGAN ``cyclegan_latent.manifest.json``.
FaderRAVE runs also copy ``attribute_stats.yaml`` and fill ``host_controls.attributes``.

Mean-encode / no-PCA host. C++ owns streaming caches — dump with cached_conv
off and strip pad/cache buffers.

  cd BRAVE
  export PYTHONPATH="${PWD}/RAVE:${PWD}/brave_rt/python:${PYTHONPATH}"
  python brave_rt/python/export_brave_rt.py \\
    --model /path/to/run_or.ckpt \\
    --output_dir brave_rt/testdata/bundles/tap_vanilla

  # FaderRAVE (Stage 7 host_controls):
  python brave_rt/python/export_brave_rt.py \\
    --model /path/to/fader_run \\
    --stats_path /path/to/lmdb/attribute_stats.yaml \\
    --output_dir brave_rt/testdata/bundles/water_fader --domain_y_name water

  # CycleGAN X→Y:
  python brave_rt/python/export_brave_rt.py \\
    --model /path/to/cyclegan_latent.ckpt \\
    --output_dir brave_rt/testdata/bundles/tap_brook_cyc \\
    --direction xy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_BRAVE = Path(__file__).resolve().parents[2]
_RAVE = _BRAVE / "RAVE"
if str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

import cached_conv as cc
import gin
import h5py
import numpy as np
import torch

import rave
from rave.fader.attributes import load_attribute_stats
from rave.canonicalizer.export import resolve_cyclegan_ckpt
from rave.canonicalizer.export.cyclegan_nn import load_cyclegan_pair_for_export


STRIP_PREFIXES = ("discriminator", "audio", "multiband", "latent_discriminator")
STRIP_SUFFIXES = ("pad", "cache")


def _filter_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if any(k.startswith(p) for p in STRIP_PREFIXES):
            continue
        if any(k.endswith(s) for s in STRIP_SUFFIXES):
            continue
        if k == "warmed_up" or k.endswith(".warmed_up"):
            continue
        if not torch.is_tensor(v):
            continue
        if v.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            continue
        out[k] = v.detach().cpu().float()
    return out


def _write_h5(path: Path, state: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in state.items():
            f.create_dataset(k, data=v.numpy())


def _dump_npy_weights(h5_path: Path, npy_dir: Path) -> int:
    npy_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            safe = k.replace(".", "__")
            arr = np.array(f[k])
            np.save(npy_dir / f"{safe}.npy", arr)
            written.append(k)
    (npy_dir / "keys.json").write_text(json.dumps(written, indent=2) + "\n")
    return len(written)


def _manifest_vanilla() -> Dict[str, Any]:
    return {
        "format_version": 1,
        "model_type": "brave",
        "sample_rate": 44100,
        "n_channels": 1,
        "n_band": 16,
        "latent_size": 128,
        "capacity": 32,
        "ratios": [2, 2, 2, 1],
        "compression_ratio": 128,
        "pqmf_attenuation_db": 40,
        "deterministic_encode": True,
        "use_noise": False,
        "dilations_list": [[3, 1], [9, 1], [27, 1], [36, 1]],
        "residual_kernel_sizes": [3],
        "host_controls": {
            "attr_mode_default": 1,
            "domains": {},
            "attributes": [],
        },
    }


def _write_pqmf_root_copies(npy_dir: Path, output_dir: Path,
                            prefix: str = "pqmf") -> None:
    """Vanilla convenience copies at bundle root (Stage 0/1 layout)."""
    fwd = npy_dir / f"{prefix}__forward_conv__weight.npy"
    inv = npy_dir / f"{prefix}__inverse_conv__weight.npy"
    if fwd.is_file():
        shutil.copy2(fwd, output_dir / "pqmf_forward_conv_weight.npy")
    if inv.is_file():
        shutil.copy2(inv, output_dir / "pqmf_inverse_conv_weight.npy")


STATS_FILENAMES = ("attribute_stats.yaml", "attribute_stats_merged.yaml")


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [t.strip() for t in value.split(",") if t.strip()]


def _is_fader_gin(config_path: str) -> bool:
    from rave.fader.export.load_for_export import is_fader_config

    return is_fader_config(config_path)


def _remove_weight_norm(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)


def find_attribute_stats(
    model_path: Optional[Path] = None,
    *,
    stats_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve attribute stats YAML: explicit > db_path > beside run/checkpoint."""
    if stats_path is not None:
        p = Path(stats_path)
        return p if p.is_file() else None
    roots: List[Path] = []
    if db_path is not None:
        roots.append(Path(db_path))
    if model_path is not None:
        mp = Path(model_path).resolve()
        run_dir = mp if mp.is_dir() else mp.parent
        roots += [run_dir, run_dir.parent]
        # …/<domain>/runs/<run> → …/<domain>/preprocessed
        if run_dir.parent.name == "runs":
            roots.append(run_dir.parent.parent / "preprocessed")
    for root in roots:
        for name in STATS_FILENAMES:
            cand = root / name
            if cand.is_file():
                return cand
    return None


def load_brave_or_fader(model_path: Path) -> Tuple[torch.nn.Module, bool]:
    """Load vanilla RAVE or FaderRAVE (weight-norm removed, eval). Returns (model, is_fader)."""
    cc.use_cached_conv(False)
    if model_path.suffix == ".ts":
        raise SystemExit(
            "TorchScript not supported for BraveRT weight export; pass a .ckpt or run dir"
        )
    config_path = rave.core.search_for_config(str(model_path))
    if config_path is None:
        raise SystemExit(f"gin config not found for {model_path}")
    gin.parse_config_file(config_path)
    is_fader = _is_fader_gin(config_path)
    run = rave.core.search_for_run(str(model_path))
    if run is None:
        raise SystemExit(f"checkpoint not found under {model_path}")
    if is_fader:
        from rave.fader.model import FaderRAVE

        model = FaderRAVE()
    else:
        model = rave.RAVE()
    model = model.load_from_checkpoint(run)
    model.eval()
    _remove_weight_norm(model)
    return model, is_fader


def load_rave_from_path(model_path: Path) -> torch.nn.Module:
    """Vanilla-only loader (kept for callers); rejects FaderRAVE."""
    model, is_fader = load_brave_or_fader(model_path)
    if is_fader:
        raise SystemExit("FaderRAVE detected; use export_fader() / load_brave_or_fader()")
    return model


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def build_host_controls(
    stats: Dict[str, Any],
    *,
    x_only: Sequence[str] = (),
    y_only: Sequence[str] = (),
    domain_x_name: Optional[str] = None,
    domain_y_name: Optional[str] = None,
    attr_mode_default: int = 1,
) -> Dict[str, Any]:
    """STAGE_07 ``host_controls`` block from attribute_stats."""
    names = list(stats.get("attribute_names", []))
    kinds = stats.get("attribute_kinds", {})
    min_max = stats.get("min_max_features", {})
    n_classes = stats.get("discrete_num_classes", {}) or {}
    class_labels = stats.get("discrete_class_labels", {}) or {}
    x_set, y_set = set(x_only), set(y_only)
    unknown = (x_set | y_set) - set(names)
    if unknown:
        raise SystemExit(f"unknown attrs in --x_only_attrs/--y_only_attrs: {sorted(unknown)}")
    both = x_set & y_set
    if both:
        raise SystemExit(f"attrs in both --x_only_attrs and --y_only_attrs: {sorted(both)}")

    rows: List[Dict[str, Any]] = []
    for name in names:
        kind = kinds.get(name, "continuous")
        if name in x_set:
            domain, applies = "x", False
        elif name in y_set:
            domain, applies = "y", True
        else:
            domain, applies = "both", True
        row: Dict[str, Any] = {
            "name": name,
            "label": _humanize(name),
            "kind": kind,
            "control": "knob" if kind == "continuous" else "menu",
            "applies_to_output": applies,
            "domain": domain,
        }
        if kind == "continuous":
            lo, hi = (float(v) for v in min_max.get(name, (0.0, 1.0)))
            row.update({"min": lo, "max": hi, "default": (lo + hi) / 2.0})
        else:
            nc = int(n_classes.get(name, len(class_labels.get(name, [])) or 1))
            labels = list(class_labels.get(name) or [str(i) for i in range(nc)])
            row.update({"num_classes": nc, "class_labels": labels, "default": 0})
        rows.append(row)

    domains: Dict[str, Any] = {}
    if domain_x_name:
        domains["x"] = {"id": "x", "display_name": domain_x_name}
    if domain_y_name:
        domains["y"] = {"id": "y", "display_name": domain_y_name}
    return {
        "attr_mode_default": attr_mode_default,
        "domains": domains,
        "attributes": rows,
    }


def _copy_stats(stats_path: Path, output_dir: Path) -> Path:
    dst = output_dir / "attribute_stats.yaml"
    if stats_path.resolve() != dst.resolve():
        shutil.copy2(stats_path, dst)
    return dst


def export_vanilla(
    model_path: Path,
    output_dir: Path,
    *,
    stats_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    x_only_attrs: Sequence[str] = (),
    y_only_attrs: Sequence[str] = (),
    domain_x_name: Optional[str] = None,
    domain_y_name: Optional[str] = None,
) -> Path:
    """Export vanilla BRAVE, or FaderRAVE when the run config says so."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, is_fader = load_brave_or_fader(model_path)
    stats_file: Optional[Path] = None
    if is_fader:
        stats_file = find_attribute_stats(
            model_path, stats_path=stats_path, db_path=db_path
        )
        if stats_file is None:
            raise SystemExit(
                "FaderRAVE export needs attribute stats: pass --stats_path "
                "/path/attribute_stats.yaml or --db_path <lmdb dir> "
                "(looked for attribute_stats.yaml / attribute_stats_merged.yaml "
                "beside --db_path, the run dir, and ../preprocessed)"
            )
    state = _filter_state_dict(model.state_dict())
    h5_path = output_dir / "weights.h5"
    _write_h5(h5_path, state)
    n = _dump_npy_weights(h5_path, output_dir / "npy_weights")
    _write_pqmf_root_copies(output_dir / "npy_weights", output_dir)
    manifest = _manifest_vanilla()
    manifest["source_checkpoint"] = str(model_path.resolve())

    if is_fader:
        assert stats_file is not None
        stats = load_attribute_stats(stats_file)
        hc = build_host_controls(
            stats,
            x_only=x_only_attrs,
            y_only=y_only_attrs,
            domain_x_name=domain_x_name,
            domain_y_name=domain_y_name,
        )
        d = len(hc["attributes"])
        latent = int(getattr(model, "latent_size", 128))
        manifest.update(
            {
                "model_type": "fader",
                "latent_size": latent,
                "content_latent_size": latent,
                "num_attributes": d,
                "decoder_latent_size": latent + d,
                "attribute_stats": "attribute_stats.yaml",
            }
        )
        manifest["host_controls"] = hc
        _copy_stats(stats_file, output_dir)
        sidecar: Dict[str, Any] = dict(hc)
        sidecar.update(
            {
                "model_type": "fader",
                "content_latent_size": latent,
                "num_attributes": d,
                "decoder_latent_size": latent + d,
            }
        )
    else:
        hc = manifest["host_controls"]
        if domain_x_name:
            hc["domains"]["x"] = {"id": "x", "display_name": domain_x_name}
        if domain_y_name:
            hc["domains"]["y"] = {"id": "y", "display_name": domain_y_name}
        sidecar = dict(hc)
        sidecar["model_type"] = "brave"

    (output_dir / "brave_rt.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "host_controls.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote {h5_path} ({len(state)} tensors, {n} npy)")
    print(f"Wrote {output_dir / 'brave_rt.json'}")
    if is_fader:
        print(f"Fader: {len(hc['attributes'])} attributes, stats {stats_file}")
    return output_dir


def _prefix_module_state(
    module: torch.nn.Module, prefix: str, strip_prefixes: Tuple[str, ...] = ()
) -> Dict[str, torch.Tensor]:
    state = _filter_state_dict(module.state_dict())
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if any(k.startswith(p) for p in strip_prefixes):
            continue
        out[f"{prefix}.{k}"] = v
    return out


def _guess_domain_names(manifest) -> Dict[str, str]:
    def _name(ckpt: str, fallback: str) -> str:
        p = Path(ckpt)
        # …/<domain>/runs/<run>/epoch_….ckpt → <domain>
        if p.parent.parent.name == "runs" and p.parent.parent.parent.name:
            return p.parent.parent.parent.name
        if p.parent.name == "runs" and p.parent.parent.name:
            return p.parent.parent.name
        return p.parent.name or fallback

    return {
        "x": _name(manifest.backbone_x_ckpt, "domain_x"),
        "y": _name(manifest.backbone_y_ckpt, "domain_y"),
    }


def export_cyclegan(
    model_path: Path,
    output_dir: Path,
    *,
    direction: str = "xy",
    display_name: Optional[str] = None,
    stats_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    x_only_attrs: Sequence[str] = (),
    y_only_attrs: Sequence[str] = (),
    domain_x_name: Optional[str] = None,
    domain_y_name: Optional[str] = None,
) -> Path:
    """Export CycleGAN brave_rt bundle (X→Y or Y→X)."""
    if direction not in ("xy", "yx"):
        raise SystemExit("--direction must be xy or yx")
    output_dir.mkdir(parents=True, exist_ok=True)
    # C++ owns caches — dump with non-streaming weights.
    try:
        backbone_x, backbone_y, warp_xy, warp_yx, manifest = load_cyclegan_pair_for_export(
            model_path, n_channels=1, streaming=False
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "CycleGAN brave_rt export failed to load backbones "
            f"({type(exc).__name__}: {exc}). Conditional CycleGAN with Fader "
            "backbones is not supported by load_cyclegan_pair_for_export yet; "
            "export the Fader run with this script instead."
        ) from exc

    shared = bool(getattr(manifest, "shared_backbone", False)) or (
        getattr(manifest, "geometry", "separate") == "joint"
    )
    geometry = getattr(manifest, "geometry", "joint" if shared else "separate")
    init_mode = str(getattr(manifest, "init_mode", "random"))
    n_layers = int(getattr(manifest, "latent_n_layers", 2))
    hidden = getattr(manifest, "latent_hidden_size", 128)
    if hidden is None:
        hidden = 128

    if direction == "xy":
        model_type = "cyclegan_xy"
        enc_bb, dec_bb = backbone_x, backbone_y
        warp = warp_xy
        prefixes = {
            "pqmf_x": backbone_x.pqmf,
            "encoder_x": backbone_x.encoder,
            "warp_xy": warp,
            "decoder_y": backbone_y.decoder,
            "pqmf_y": backbone_y.pqmf,
        }
        # Also keep unused inverse warp out of the hot path; omit warp_yx.
    else:
        model_type = "cyclegan_yx"
        enc_bb, dec_bb = backbone_y, backbone_x
        warp = warp_yx
        prefixes = {
            "pqmf_y": backbone_y.pqmf,
            "encoder_y": backbone_y.encoder,
            "warp_yx": warp,
            "decoder_x": backbone_x.decoder,
            "pqmf_x": backbone_x.pqmf,
        }

    state: Dict[str, torch.Tensor] = {}
    for pref, mod in prefixes.items():
        if mod is None:
            raise SystemExit(f"missing module for prefix {pref}")
        # Encoder/decoder live under .encoder / are the module itself
        state.update(_prefix_module_state(mod, pref))

    # Ensure Enc/Dec include nested encoder.encoder.* paths as exported
    # VariationalEncoder state_dict keys start with "encoder." already when
    # dumping backbone.encoder — _prefix_module_state yields encoder_x.encoder.*

    h5_path = output_dir / "weights.h5"
    _write_h5(h5_path, state)
    n = _dump_npy_weights(h5_path, output_dir / "npy_weights")

    domains = _guess_domain_names(manifest)
    if domain_x_name:
        domains["x"] = domain_x_name
    if domain_y_name:
        domains["y"] = domain_y_name
    if display_name is None:
        # ASCII "->" — many DAW typefaces lack U+2192; C++ also decodes \uXXXX.
        if direction == "xy":
            display_name = f"{domains['x']} -> {domains['y']}"
        else:
            display_name = f"{domains['y']} -> {domains['x']}"

    br_manifest: Dict[str, Any] = {
        "format_version": 1,
        "model_type": model_type,
        "sample_rate": 44100,
        "n_channels": 1,
        "n_band": 16,
        "latent_size": 128,
        "capacity": 32,
        "ratios": [2, 2, 2, 1],
        "compression_ratio": 128,
        "pqmf_attenuation_db": 40,
        "deterministic_encode": True,
        "use_noise": False,
        "dilations_list": [[3, 1], [9, 1], [27, 1], [36, 1]],
        "residual_kernel_sizes": [3],
        "geometry": geometry,
        "shared_backbone": bool(shared),
        "init_mode": init_mode,
        "latent_n_layers": n_layers,
        "latent_hidden_size": int(hidden),
        "encode_hop": 128,
        "display_name": display_name,
        "sidechain": {
            "present": True,
            "default_enabled": True,
            "knee_db": -40.0,
            "smooth_frames": 8,
        },
        "weight_prefixes": {
            "pqmf_in": "pqmf_x." if direction == "xy" else "pqmf_y.",
            "encoder": "encoder_x." if direction == "xy" else "encoder_y.",
            "warp": "warp_xy." if direction == "xy" else "warp_yx.",
            "decoder": "decoder_y." if direction == "xy" else "decoder_x.",
            "pqmf_out": "pqmf_y." if direction == "xy" else "pqmf_x.",
        },
        "source_checkpoint": str(model_path.resolve()),
    }

    # Conditional decoder (Fader-style): project conv in_ch > latent_size.
    dec_prefix = "decoder_y." if direction == "xy" else "decoder_x."
    dec_in = 128
    for k, v in state.items():
        if k.startswith(dec_prefix) and k.endswith("net.0.weight") and v.dim() == 3:
            dec_in = int(v.shape[1])
            break
    host_controls: Dict[str, Any] = {
        "attr_mode_default": 1,
        "domains": {
            "x": {"id": "x", "display_name": domains["x"]},
            "y": {"id": "y", "display_name": domains["y"]},
        },
        "attributes": [],
    }
    if dec_in > 128:
        stats_file = find_attribute_stats(
            model_path, stats_path=stats_path, db_path=db_path
        )
        if stats_file is None:
            raise SystemExit(
                f"conditional CycleGAN decoder (in_ch={dec_in}) needs attribute "
                "stats: pass --stats_path or --db_path"
            )
        stats = load_attribute_stats(stats_file)
        cond = build_host_controls(
            stats,
            x_only=x_only_attrs,
            y_only=y_only_attrs,
            domain_x_name=domains["x"],
            domain_y_name=domains["y"],
        )
        if len(cond["attributes"]) != dec_in - 128:
            raise SystemExit(
                f"stats have {len(cond['attributes'])} attributes but decoder "
                f"expects {dec_in - 128}"
            )
        host_controls = cond
        _copy_stats(stats_file, output_dir)
        br_manifest.update(
            {
                "content_latent_size": 128,
                "num_attributes": dec_in - 128,
                "decoder_latent_size": dec_in,
                "attribute_stats": "attribute_stats.yaml",
            }
        )
    br_manifest["host_controls"] = host_controls
    (output_dir / "brave_rt.json").write_text(
        json.dumps(br_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    sidecar = Path(model_path).with_suffix(".manifest.json")
    if not sidecar.is_file():
        # run dir form: …/cyclegan_latent.ckpt → …/cyclegan_latent.manifest.json
        alt = Path(model_path).parent / "cyclegan_latent.manifest.json"
        sidecar = alt if alt.is_file() else sidecar
    if sidecar.is_file():
        shutil.copy2(sidecar, output_dir / "cyclegan_latent.manifest.json")
    else:
        (output_dir / "cyclegan_latent.manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )

    sidecar_hc: Dict[str, Any] = dict(host_controls)
    sidecar_hc.update(
        {
            "model_type": model_type,
            "domain_display": domains,
            "display_name": display_name,
        }
    )
    if not host_controls["attributes"]:
        sidecar_hc["note"] = (
            "Unconditional CycleGAN — no conditioning features in this bundle "
            "(Stage 7 UI would show an empty-attributes message)."
        )
    (output_dir / "host_controls.json").write_text(
        json.dumps(sidecar_hc, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"Wrote CycleGAN {direction} bundle {output_dir} "
        f"({len(state)} tensors, {n} npy) init_mode={init_mode} geometry={geometry}"
    )
    # Silence unused for lint clarity
    _ = (enc_bb, dec_bb)
    return output_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Run dir or .ckpt")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "brave", "cyclegan"],
        help="auto: detect CycleGAN via resolve_cyclegan_ckpt",
    )
    p.add_argument(
        "--direction",
        default="xy",
        choices=["xy", "yx"],
        help="CycleGAN export direction (xy = Enc_X→W_xy→Dec_Y)",
    )
    p.add_argument("--display_name", default=None, help="Optional UI title override")
    p.add_argument("--stats_path", type=Path, default=None,
                   help="attribute_stats.yaml (Fader / conditional)")
    p.add_argument("--db_path", type=Path, default=None,
                   help="LMDB dir; looks for attribute_stats[_merged].yaml")
    p.add_argument("--x_only_attrs", "--native_attrs_x", dest="x_only_attrs",
                   default="", help="comma list: attrs only for X (applies_to_output=false)")
    p.add_argument("--y_only_attrs", "--native_attrs_y", dest="y_only_attrs",
                   default="", help="comma list: attrs only for Y/output")
    p.add_argument("--domain_x_name", default=None, help="Domain X display name")
    p.add_argument("--domain_y_name", default=None, help="Domain Y display name")
    args = p.parse_args()
    hc_kwargs = dict(
        stats_path=args.stats_path,
        db_path=args.db_path,
        x_only_attrs=_parse_csv(args.x_only_attrs),
        y_only_attrs=_parse_csv(args.y_only_attrs),
        domain_x_name=args.domain_x_name,
        domain_y_name=args.domain_y_name,
    )

    cycle_ckpt = resolve_cyclegan_ckpt(str(args.model))
    want_cycle = args.model_type == "cyclegan" or (
        args.model_type == "auto" and cycle_ckpt is not None
    )
    if want_cycle:
        if cycle_ckpt is None:
            raise SystemExit(f"no cyclegan_latent.ckpt under {args.model}")
        export_cyclegan(
            Path(cycle_ckpt),
            args.output_dir,
            direction=args.direction,
            display_name=args.display_name,
            **hc_kwargs,
        )
        return
    if args.model_type == "cyclegan":
        raise SystemExit("cyclegan requested but checkpoint not found")
    export_vanilla(args.model, args.output_dir, **hc_kwargs)


if __name__ == "__main__":
    main()
