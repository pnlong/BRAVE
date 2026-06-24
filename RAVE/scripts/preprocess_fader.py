#!/usr/bin/env python3
"""
Full Fader RAVE dataset prep: LMDB preprocess → index manifest → optional FSD sidecar → attribute stats.

Run from the BRAVE repo root::

  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

**Continuous-only** (birdsong, field recordings) — optional **PCEN** suppresses steady
environmental noise and boosts transients (bird calls) before LMDB write::

  python RAVE/scripts/preprocess_fader.py \\
    --input_path .../yt_birdsong/audio \\
    --db_path .../yt_birdsong/preprocessed \\
    --config configs/brave_fader_birdsong.gin \\
    --pcen --normalize

**Discrete sidecar** (FSD50K ``texture_class`` or ``water_scene``) — use a gin config
that lists the discrete attribute and pass ``--sidecar_scheme`` matching it::

  python RAVE/scripts/preprocess_fader.py \\
    --input_path .../fsd50k/texture/audio_subset \\
    --db_path .../fsd50k/texture/preprocessed \\
    --config configs/brave_fader_texture.gin \\
    --sidecar_scheme texture_class \\
    --sidecar_partition dev_train

Continuous attributes are summarized in ``attribute_stats.yaml`` (``precompute_descriptors.py``).
Discrete clip labels live in ``attribute_sidecar.yaml`` (requires manifest + FSD50K CSVs).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

_RAVE_ROOT = Path(__file__).resolve().parents[1]
_BRAVE_ROOT = _RAVE_ROOT.parent
if str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

RAVE_ROOT = _RAVE_ROOT
BRAVE_ROOT = _BRAVE_ROOT
SCRIPTS = RAVE_ROOT / "scripts"

_GIN_LIST_RE = re.compile(
    r"^\s*(?:DISCRETE_ATTRIBUTES|discrete_attributes)\s*=\s*(\[.*\])\s*$",
    re.MULTILINE,
)


def _env() -> dict:
    env = os.environ.copy()
    root = str(RAVE_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    return env


def _run(cmd: list[str], *, cwd: Path = BRAVE_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd, env=_env())


def _resolve_config(config: str) -> str:
    path = Path(config)
    if path.is_file():
        return str(path.resolve())
    brave_path = BRAVE_ROOT / config
    if brave_path.is_file():
        return str(brave_path.resolve())
    return config


def _discrete_attributes_from_gin(config_path: str) -> list[str]:
    """Best-effort parse of DISCRETE_ATTRIBUTES / discrete_attributes from gin text."""
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except OSError:
        return []
    names: list[str] = []
    for match in _GIN_LIST_RE.finditer(text):
        try:
            value = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list):
            names.extend(str(v) for v in value)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _validate_sidecar_config(
    discrete_attrs: list[str],
    sidecar_scheme: str | None,
    skip_sidecar: bool,
) -> None:
    if skip_sidecar or not discrete_attrs:
        return
    if sidecar_scheme is None:
        raise SystemExit(
            "Gin config declares discrete attribute(s) "
            f"{discrete_attrs!r} but --sidecar_scheme was not set.\n"
            "Pass --sidecar_scheme texture_class or water_scene (FSD50K), or use a "
            "continuous-only gin config."
        )
    scheme_to_attr = {
        "texture_class": "texture_class",
        "water_scene": "water_scene",
    }
    expected = scheme_to_attr.get(sidecar_scheme)
    if expected and expected not in discrete_attrs:
        print(
            f"Warning: --sidecar_scheme={sidecar_scheme} builds '{expected}' "
            f"but gin discrete list is {discrete_attrs!r}.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fader preprocess: LMDB + manifest + sidecar (optional) + attribute stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Directory of source WAV/MP3/… (same as preprocess.py)",
    )
    parser.add_argument(
        "--db_path",
        required=True,
        help="Output LMDB directory (created if missing)",
    )
    parser.add_argument(
        "--config",
        default="configs/brave_fader_birdsong.gin",
        help="Fader gin config for precompute_descriptors (attribute lists)",
    )
    parser.add_argument("--num_signal", type=int, default=131072)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=44100)
    parser.add_argument("--workers", type=int, default=0, help="0 = all CPU cores")
    parser.add_argument(
        "--concat_seed",
        type=int,
        default=42,
        help="Must match between preprocess and manifest (preprocess.py default)",
    )
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--denoise_strength", type=float, default=0.75)
    parser.add_argument("--denoise_noise_sec", type=float, default=0.0)
    parser.add_argument("--pcen", action="store_true",
                        help="PCEN: suppress steady noise, boost transients (birdsong)")
    parser.add_argument("--pcen_n_mels", type=int, default=128)
    parser.add_argument("--pcen_gain", type=float, default=0.98)
    parser.add_argument("--pcen_bias", type=float, default=2.0)
    parser.add_argument("--pcen_power", type=float, default=0.5)
    parser.add_argument("--pcen_time_constant", type=float, default=0.4)
    parser.add_argument("--pcen_max_gain", type=float, default=10.0)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize each LMDB chunk (same rule as train.py --normalize)",
    )
    parser.add_argument(
        "--normalize_max_gain_db",
        type=float,
        default=30.0,
        help="Max gain in dB when --normalize is set",
    )
    parser.add_argument(
        "--sidecar_scheme",
        default=None,
        choices=["water_scene", "texture_class"],
        help="Build attribute_sidecar.yaml from FSD50K CSV labels",
    )
    parser.add_argument(
        "--sidecar_partition",
        action="append",
        default=["dev_train"],
        help="FSD50K partition(s) for sidecar (repeat flag for multiple)",
    )
    parser.add_argument(
        "--sidecar_dataset_root",
        default=None,
        help="FSD50K release root (default: $BRAVE_STORAGE/FSD50K)",
    )
    parser.add_argument(
        "--sidecar_tags_config",
        default=None,
        help="YAML tag taxonomy override (scheme-specific default under fsd50k/configs/)",
    )
    parser.add_argument(
        "--sidecar_priority",
        default="storm_first",
        choices=["storm_first", "coastal_first"],
        help="water_scene multi-label priority",
    )
    parser.add_argument(
        "--sidecar_texture_only",
        action="store_true",
        default=True,
        help="texture_class: omit LMDB rows for classes 10–11 (default: on)",
    )
    parser.add_argument(
        "--no_sidecar_texture_only",
        action="store_false",
        dest="sidecar_texture_only",
        help="texture_class: keep all rows including music/vocal classes",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="LMDB already exists; only manifest / sidecar / stats",
    )
    parser.add_argument(
        "--skip_manifest",
        action="store_true",
        help="Skip lmdb_index_manifest.yaml",
    )
    parser.add_argument(
        "--skip_sidecar",
        action="store_true",
        help="Skip attribute_sidecar.yaml even if --sidecar_scheme is set",
    )
    parser.add_argument(
        "--skip_precompute",
        action="store_true",
        help="Skip attribute_stats.yaml",
    )
    parser.add_argument(
        "--no_train_only",
        action="store_true",
        help="Precompute stats on full LMDB, not train split only",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    input_path = os.path.abspath(args.input_path)
    config = _resolve_config(args.config)
    discrete_attrs = _discrete_attributes_from_gin(config)
    _validate_sidecar_config(
        discrete_attrs, args.sidecar_scheme, args.skip_sidecar)

    worker_args = (
        ["--workers", str(args.workers)] if args.workers > 0 else ["--workers", "0"]
    )

    if not args.skip_preprocess:
        cmd = [
            sys.executable,
            str(SCRIPTS / "preprocess.py"),
            f"--input_path={input_path}",
            f"--output_path={db_path}",
            f"--num_signal={args.num_signal}",
            f"--channels={args.channels}",
            f"--sampling_rate={args.sampling_rate}",
            f"--concat_seed={args.concat_seed}",
            *worker_args,
        ]
        if args.denoise:
            cmd.append("--denoise")
            cmd.append(f"--denoise_strength={args.denoise_strength}")
            cmd.append(f"--denoise_noise_sec={args.denoise_noise_sec}")
        if args.pcen:
            cmd.append("--pcen")
            cmd.append(f"--pcen_n_mels={args.pcen_n_mels}")
            cmd.append(f"--pcen_gain={args.pcen_gain}")
            cmd.append(f"--pcen_bias={args.pcen_bias}")
            cmd.append(f"--pcen_power={args.pcen_power}")
            cmd.append(f"--pcen_time_constant={args.pcen_time_constant}")
            cmd.append(f"--pcen_max_gain={args.pcen_max_gain}")
        if args.normalize:
            cmd.append("--normalize")
            cmd.append(f"--normalize_max_gain_db={args.normalize_max_gain_db}")
        _run(cmd)

    if not args.skip_manifest:
        _run([
            sys.executable,
            str(SCRIPTS / "build_lmdb_index_manifest.py"),
            f"--input_path={input_path}",
            f"--db_path={db_path}",
            f"--num_signal={args.num_signal}",
            f"--sampling_rate={args.sampling_rate}",
            f"--concat_seed={args.concat_seed}",
            *worker_args,
        ])

    if args.sidecar_scheme and not args.skip_sidecar:
        sidecar_cmd = [
            sys.executable,
            str(SCRIPTS / "build_attribute_sidecar.py"),
            f"--db_path={db_path}",
            f"--scheme={args.sidecar_scheme}",
            f"--priority={args.sidecar_priority}",
            *worker_args,
        ]
        if args.sidecar_texture_only:
            sidecar_cmd.append("--texture_only")
        else:
            sidecar_cmd.append("--notexture_only")
        if args.sidecar_dataset_root:
            sidecar_cmd.append(f"--dataset_root={args.sidecar_dataset_root}")
        if args.sidecar_tags_config:
            sidecar_cmd.append(f"--tags_config={args.sidecar_tags_config}")
        for part in args.sidecar_partition:
            sidecar_cmd.append(f"--partition={part}")
        _run(sidecar_cmd)

    if not args.skip_precompute:
        pre_cmd = [
            sys.executable,
            str(SCRIPTS / "precompute_descriptors.py"),
            f"--db_path={db_path}",
            f"--config={config}",
            f"--n_signal={args.num_signal}",
        ]
        if not args.no_train_only:
            pre_cmd.append("--train_only")
        _run(pre_cmd)

    print()
    print("Fader preprocess complete.")
    print(f"  LMDB:            {db_path}")
    print(f"  manifest:        {os.path.join(db_path, 'lmdb_index_manifest.yaml')}")
    if args.sidecar_scheme and not args.skip_sidecar:
        print(f"  sidecar:         {os.path.join(db_path, 'attribute_sidecar.yaml')}")
    elif discrete_attrs:
        print(
            f"  sidecar:         (none — discrete attrs in gin: {discrete_attrs})",
        )
    print(f"  attribute stats: {os.path.join(db_path, 'attribute_stats.yaml')}")
    print()
    print("Train:")
    print(
        f"  python RAVE/scripts/train.py --config={args.config} "
        f"--name=YOUR_RUN --db_path={db_path}"
    )


if __name__ == "__main__":
    main()
