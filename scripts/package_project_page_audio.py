#!/usr/bin/env python3
"""Package presentation demo stems into MP3s for the GitHub Pages project page.

Reads WAV stems from presentation_phase_demos output::

    <src>/<domain>/<clip>/stems/
      00_tap.wav
      02_phase1_recon.wav / 02_phase1_recon_sc.wav
      03_phase2_recon.wav / 03_phase2_recon_sc.wav
      04_cyclegan.wav / 04_cyclegan_sc.wav

Writes::

    <out>/<domain>/<clip>/{tap,phase1,phase2,phase3}.mp3
    <out>/<domain>/<clip>/{phase1,phase2,phase3}_sc.mp3
    <out>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

STEM_MAP = {
    "tap": "00_tap.wav",
    "phase1": "02_phase1_recon.wav",
    "phase2": "03_phase2_recon.wav",
    "phase3": "04_cyclegan.wav",
}

# Optional loudness-sidechain variants (tap envelope ducks the reconstruction).
SC_STEM_MAP = {
    "phase1_sc": "02_phase1_recon_sc.wav",
    "phase2_sc": "03_phase2_recon_sc.wav",
    "phase3_sc": "04_cyclegan_sc.wav",
}

DOMAIN_LABELS = {
    "babbling_brook": "Babbling Brook",
    "fire_crackling": "Fire Crackling",
    "birds_chirping": "Birds Chirping",
    "rain_sounds": "Rain Sounds",
    "asmr": "ASMR",
    "mouth_sounds": "Mouth Sounds",
    "vctk": "VCTK (speech)",
    "tabla": "Tabla",
}

CLIP_LABELS = {
    "sm57-0": "SM57 (take 0)",
    "0": "SM57 (take 0)",
    "Contact_Mic_L-moanin": "Contact mic — Moanin'",
    "moanin-Contact_Mic_L": "Contact mic — Moanin'",
    "Shotgun-tapping_11": "Shotgun — tapping",
    "Shotgun-heavy_3": "Shotgun — heavy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Root of presentation_phase_demos output",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (e.g. docs/assets/audio)",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Domains to include (default: all subdirs of --src)",
    )
    p.add_argument(
        "--clips",
        nargs="+",
        default=None,
        help="Clip stem names to include (default: all under each domain)",
    )
    p.add_argument("--bitrate", default="192k", help="ffmpeg MP3 bitrate")
    p.add_argument(
        "--peak-db",
        type=float,
        default=-1.0,
        help="Peak-normalize each stem to this dBFS before encode (0 = skip)",
    )
    return p.parse_args()


def peak_normalize(audio: np.ndarray, peak_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio
    target = 10.0 ** (peak_db / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    *,
    bitrate: str,
    peak_db: float,
) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wav_path), always_2d=True)
    audio = audio.astype(np.float32)
    if peak_db != 0.0:
        audio = peak_normalize(audio, peak_db)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), audio, sr)
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(tmp_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def clip_label(stem: str) -> str:
    if stem in CLIP_LABELS:
        return CLIP_LABELS[stem]
    if "moanin" in stem.lower():
        return "Contact mic — Moanin'"
    lower = stem.lower()
    if lower.startswith("shotgun"):
        rest = stem.split("-", 1)[-1].replace("_", " ")
        return f"Shotgun — {rest}"
    if stem.startswith("sm57") or stem in {"0", "1"}:
        return f"SM57 ({stem})"
    return stem.replace("_", " ")


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


def list_clip_dirs(
    domain_dir: Path, clip_filter: Optional[Sequence[str]]
) -> List[Path]:
    clips = [
        p for p in domain_dir.iterdir() if p.is_dir() and (p / "stems").is_dir()
    ]
    if clip_filter:
        keys = [c.lower() for c in clip_filter]
        clips = [
            p
            for p in clips
            if any(k == p.name.lower() or k in p.name.lower() for k in keys)
        ]

    def sort_key(p: Path) -> tuple:
        name = p.name.lower()
        # SM57 → shotgun → contact on the page.
        if name.startswith("sm57") or name in {"0", "1"}:
            return (0, name)
        if name.startswith("shotgun"):
            return (1, name)
        return (2, name)

    return sorted(clips, key=sort_key)


def package_domain(
    domain: str,
    domain_dir: Path,
    out_root: Path,
    *,
    clip_filter: Optional[Sequence[str]],
    bitrate: str,
    peak_db: float,
) -> Dict:
    clip_entries: List[Dict] = []
    for clip_dir in list_clip_dirs(domain_dir, clip_filter):
        stems_dir = clip_dir / "stems"
        missing = [
            name
            for key, name in STEM_MAP.items()
            if not (stems_dir / name).is_file()
        ]
        if missing:
            print(f"  skip {domain}/{clip_dir.name}: missing {missing}")
            continue

        rel_files: Dict[str, str] = {}
        for key, wav_name in {**STEM_MAP, **SC_STEM_MAP}.items():
            src_wav = stems_dir / wav_name
            if key in SC_STEM_MAP and not src_wav.is_file():
                continue
            dest = out_root / domain / clip_dir.name / f"{key}.mp3"
            print(f"  encode {domain}/{clip_dir.name}/{key}.mp3")
            wav_to_mp3(
                src_wav,
                dest,
                bitrate=bitrate,
                peak_db=peak_db,
            )
            rel_files[key] = f"{domain}/{clip_dir.name}/{key}.mp3"

        has_sc = all(k in rel_files for k in SC_STEM_MAP)
        clip_entries.append(
            {
                "id": clip_dir.name,
                "label": clip_label(clip_dir.name),
                "files": rel_files,
                "has_sidechain": has_sc,
            }
        )

    return {
        "id": domain,
        "label": domain_label(domain),
        "clips": clip_entries,
    }


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise FileNotFoundError(f"--src not found: {args.src}")

    domains = args.domains
    if domains is None:
        domains = sorted(
            p.name
            for p in args.src.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_domains: List[Dict] = []
    for domain in domains:
        domain_dir = args.src / domain
        if not domain_dir.is_dir():
            print(f"skip missing domain dir: {domain_dir}")
            continue
        print(f"=== {domain} ===")
        entry = package_domain(
            domain,
            domain_dir,
            args.out,
            clip_filter=args.clips,
            bitrate=args.bitrate,
            peak_db=args.peak_db,
        )
        if entry["clips"]:
            manifest_domains.append(entry)

    any_sc = any(
        c.get("has_sidechain")
        for d in manifest_domains
        for c in d.get("clips", [])
    )
    manifest = {
        "version": 2,
        "sidechain_default": True,
        "sidechain_available": any_sc,
        "tracks": [
            {"id": "tap", "label": "Tap (input)", "sidechain": False},
            {"id": "phase1", "label": "Zero-shot", "sidechain": True},
            {"id": "phase2", "label": "EQ + reverb", "sidechain": True},
            {"id": "phase3", "label": "CycleGAN", "sidechain": True},
        ],
        "domains": manifest_domains,
    }
    out_manifest = args.out / "manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {out_manifest} ({len(manifest_domains)} domains)")


if __name__ == "__main__":
    main()
