#!/usr/bin/env python3
"""
Sample random Tabla ISMIR WAV clips into a local listen folder.

Stroke classes are the immediate subfolders under ``train/`` or ``test/`` (e.g. one
folder per 4-way category). Output defaults to ``artifacts/listen_samples/`` (gitignored).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from paths import LISTEN_SAMPLES_DIR, SPLIT_CHOICES, split_audio_dir


def collect_wavs(
    split: str,
    *,
    stroke_class: str | None,
    wav_root: Path | None,
) -> list[tuple[str, str, Path]]:
    """Return (clip_id, stroke_class, path) for each WAV."""
    root = Path(wav_root) if wav_root is not None else split_audio_dir(split)
    if not root.is_dir():
        return []
    if stroke_class:
        class_dirs = [root / stroke_class]
    else:
        class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not class_dirs:
            class_dirs = [root]
    out: list[tuple[str, str, Path]] = []
    for class_dir in class_dirs:
        if not class_dir.is_dir():
            continue
        label = stroke_class or class_dir.name
        for wav in sorted(class_dir.rglob("*.wav")):
            if wav.is_file():
                out.append((wav.stem, label, wav))
    return out


def default_output_dir(split: str, stroke_class: str | None, n: int, seed: int) -> Path:
    tag = stroke_class or "all"
    return LISTEN_SAMPLES_DIR / f"{tag}_{split}_n{n}_seed{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "--stroke-class",
        default=None,
        help="Limit to one stroke-class subfolder name (default: sample across all classes).",
    )
    p.add_argument("-n", "--num-samples", type=int, default=8, metavar="N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=list(SPLIT_CHOICES), default="train")
    p.add_argument("--wav-root", type=Path, default=None, help="Override split audio root.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--symlink",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ns = p.parse_args()
    if ns.num_samples < 1:
        p.error("--num-samples must be >= 1")
    return ns


def main() -> None:
    args = parse_args()
    pool = collect_wavs(
        args.split,
        stroke_class=args.stroke_class,
        wav_root=args.wav_root,
    )
    if not pool:
        print("No WAV files found.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    picks = rng.sample(pool, min(args.num_samples, len(pool)))

    out_dir = args.output_dir or default_output_dir(
        args.split, args.stroke_class, args.num_samples, args.seed)
    out_dir = out_dir.resolve()
    manifest_path = out_dir / "manifest.tsv"
    if out_dir.exists() and manifest_path.exists() and not args.overwrite:
        print(f"Output exists (use --overwrite): {out_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = ["index\tclip_id\tstroke_class\tsource_path\toutput_path"]
    for i, (cid, stroke, src) in enumerate(picks, 1):
        dst = out_dir / f"{i:03d}_{stroke}_{cid}.wav"
        if dst.exists() or dst.is_symlink():
            if args.overwrite:
                dst.unlink()
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        rows.append(f"{i}\t{cid}\t{stroke}\t{src.resolve()}\t{dst.resolve()}")

    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    root = args.wav_root or split_audio_dir(args.split)
    print(f"Split:         {args.split}")
    print(f"Stroke class:  {args.stroke_class or '(all)'}")
    print(f"WAV root:      {root.resolve()}")
    print(f"Pool size:     {len(pool)}")
    print(f"Sampled:       {len(picks)} (seed {args.seed})")
    print(f"Output:        {out_dir}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
