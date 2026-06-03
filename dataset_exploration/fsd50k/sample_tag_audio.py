#!/usr/bin/env python3
"""
Copy or symlink random FSD50K WAV clips that contain a given ontology tag into a local
folder for listening.

Output defaults to ``artifacts/listen_samples/`` (gitignored). Each run creates a new
subdirectory with numbered ``.wav`` files (symlinks by default) plus ``manifest.tsv``.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from fsd50k_manifest import iter_manifest_clips
from paths import (
    PARTITION_CHOICES,
    canonical_partition,
    default_listen_samples_dir,
    partitions_for,
)
from tag_utils import normalize_tag


def clips_with_tag(
    tag: str,
    partition: str,
    *,
    dataset_root: Path | None,
    wav_root: Path | None,
) -> list[tuple[str, list[str], Path]]:
    tag = normalize_tag(tag)
    part = partitions_for(dataset_root)[canonical_partition(partition)]
    audio_dir = Path(wav_root) if wav_root is not None else part.audio_dir
    out: list[tuple[str, list[str], Path]] = []
    for cid, labels in iter_manifest_clips(part):
        if tag not in labels:
            continue
        wav = audio_dir / f"{cid}.wav"
        if wav.is_file():
            out.append((cid, labels, wav))
    return out


def default_output_dir(tag: str, partition: str, n: int, seed: int) -> Path:
    part = canonical_partition(partition)
    tag = normalize_tag(tag)
    return default_listen_samples_dir() / f"{tag}_{part}_n{n}_seed{seed}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "--tag",
        required=True,
        help="Ontology class token (strip + lowercase, e.g. water, rain).",
    )
    p.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=8,
        metavar="N",
        help="Number of random clips to stage (default: 8).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42).",
    )
    p.add_argument(
        "--partition",
        choices=list(PARTITION_CHOICES),
        default="dev_train",
        help="Manifest split (default: dev_train).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="FSD50K release root (default: $BRAVE_STORAGE/FSD50K).",
    )
    p.add_argument(
        "--wav-root",
        type=Path,
        default=None,
        help="Flat folder of <clip_id>.wav (default: official partition audio dir).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for this run (default: "
            "artifacts/listen_samples/<tag>_<partition>_n<N>_seed<S>). "
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory contents if the run folder already exists.",
    )
    p.add_argument(
        "--symlink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink WAVs into the output folder (default). Use --no-symlink to copy.",
    )
    ns = p.parse_args()
    if ns.num_samples < 1:
        p.error("--num-samples must be >= 1")
    return ns


def main() -> None:
    args = parse_args()
    tag = normalize_tag(args.tag)
    part_name = canonical_partition(args.partition)

    pool = clips_with_tag(
        tag,
        args.partition,
        dataset_root=args.dataset_root,
        wav_root=args.wav_root,
    )
    if not pool:
        print(
            f"No clips with tag '{tag}' and existing WAV on partition '{part_name}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    picks = rng.sample(pool, min(args.num_samples, len(pool)))

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = default_output_dir(tag, part_name, args.num_samples, args.seed)
    out_dir = out_dir.resolve()
    manifest_path = out_dir / "manifest.tsv"
    if out_dir.exists() and manifest_path.exists() and not args.overwrite:
        print(
            f"Output folder already exists (use --overwrite): {out_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = ["index\tclip_id\tlabels\tsource_path\toutput_path"]
    for i, (cid, labels, src) in enumerate(picks, 1):
        dst = out_dir / f"{i:03d}_{cid}.wav"
        if dst.exists() or dst.is_symlink():
            if args.overwrite:
                dst.unlink()
            else:
                print(f"Skip existing {dst.name} (use --overwrite)", file=sys.stderr)
                continue
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        label_str = ",".join(labels)
        rows.append(
            f"{i}\t{cid}\t{label_str}\t{src.resolve()}\t{dst.resolve()}"
        )

    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    wav_root = args.wav_root or partitions_for(args.dataset_root)[part_name].audio_dir
    print(f"Tag:           {tag}")
    print(f"Partition:     {part_name}")
    print(f"WAV root:      {wav_root.resolve()}")
    print(f"Pool size:     {len(pool)} clips with tag + WAV")
    print(f"Copied:        {len(picks)} (requested {args.num_samples}, seed {args.seed})")
    print(f"Method:        {'symlink' if args.symlink else 'copy'}")
    print(f"Output folder: {out_dir}")
    print(f"Manifest:      {manifest_path}")
    print()
    print("Open the .wav files in any player (e.g. ffplay, VLC, OS file browser).")


if __name__ == "__main__":
    main()
