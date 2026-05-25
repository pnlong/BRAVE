#!/usr/bin/env python3
"""
Build a flat folder of WAV clips whose ground-truth labels match a whitelist.

By default clips are staged as symlinks pointing into the dataset tree. Use ``--method
copy`` on filesystems (e.g. some NAS deployments) where symlinks are disallowed—this
writes full WAV copies instead (much higher disk usage).

Staging can overlap filesystem I/O with ``--workers`` parallel processes.

Whitelist: UTF-8 text, one stripped-lowercase ontology class token per non-empty line.

Labels come from official ``*.csv`` ``labels`` cells (comma-separated). A clip is kept
when any normalized token intersects the whitelist (exact match).

Default partition is ``dev_train`` (training split inside the official development set).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from fsd50k_manifest import count_manifest_clips, iter_manifest_clips
from paths import PARTITION_CHOICES, canonical_partition, default_subset_audio_dir, partitions_for
from tag_utils import normalize_tag


# Payload: cid, absolute_src.as_posix(), dst.as_posix(), method, overwrite
StagePayload = tuple[str, str, str, str, bool]


def _stage_worker(payload: StagePayload) -> tuple[str, str, str]:
    """
    Runs in forked worker. Returns ``(kind, cid, detail)``.
    ``kind`` is ``staged`` | ``skipped_existing`` | ``error``.
    """
    cid, src_s, dst_s, method, overwrite = payload
    dst = Path(dst_s)
    src = Path(src_s)
    try:
        if dst.exists() or dst.is_symlink():
            if overwrite:
                dst.unlink()
            else:
                return ("skipped_existing", cid, "")
        if method == "symlink":
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
        return ("staged", cid, "")
    except OSError as e:
        return ("error", cid, str(e))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--whitelist",
        required=True,
        type=Path,
        help="Plain text file with one strip+lower class token per non-empty line.",
    )
    parser.add_argument(
        "--partition",
        choices=list(PARTITION_CHOICES),
        default="dev_train",
        help="Split manifest (dev_train/dev_val/eval or synonyms train/valid/test). Default: dev_train.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Directory containing official FSD50K.* folders (default: $BRAVE_STORAGE/FSD50K).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory for staged WAVs (default: {default_subset_audio_dir()!s}).",
    )
    parser.add_argument(
        "--method",
        choices=("symlink", "copy"),
        default="symlink",
        help="Staging mode: symlink (default, low disk) or copy (NAS-friendly; duplicates data).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Parallel staging worker processes (>1 wraps Stage I/O via ProcessPoolExecutor). "
            "Default 1 streams the manifest in the main process (gentle on NAS quotas)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files or symlinks in the output folder when names collide.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (stderr); only prints final counters.",
    )
    ns = parser.parse_args()
    if ns.workers < 1:
        parser.error("--workers must be >= 1")
    return ns


def load_whitelist(path: Path) -> set[str]:
    out: set[str] = set()
    data = path.read_text(encoding="utf-8").splitlines()
    for line in data:
        t = normalize_tag(line)
        if t:
            out.add(t)
    if not out:
        raise ValueError(f"Whitelist appears empty after parsing: {path}")
    return out


def manifest_iter_for_ui(part, name: str, method: str, no_progress: bool):
    clip_iter = iter_manifest_clips(part)
    if no_progress:
        return clip_iter
    return tqdm(
        clip_iter,
        total=count_manifest_clips(part),
        desc=f"Subset {name} ({method})",
        unit="clip",
        smoothing=0.05,
        file=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    roots = partitions_for(args.dataset_root)
    name = canonical_partition(args.partition)
    part = roots[name]

    out_dir = args.output_dir if args.output_dir is not None else default_subset_audio_dir()
    audio_dir = part.audio_dir

    if not audio_dir.is_dir():
        print(f"error: audio partition directory missing: {audio_dir}", file=sys.stderr)
        sys.exit(1)
    if not part.csv_path.is_file():
        print(f"error: CSV manifest missing: {part.csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    whitelist = load_whitelist(args.whitelist)

    staged = 0
    skipped_no_match = 0
    skipped_no_wav = 0
    skipped_existing = 0
    staging_errors = 0

    clip_iter_wrapped = manifest_iter_for_ui(
        part, name, args.method, args.no_progress
    )

    if args.workers == 1:
        # Stream manifest; stage in-process (minimal RAM).
        for cid, labels in clip_iter_wrapped:
            if not set(labels) & whitelist:
                skipped_no_match += 1
                continue
            wav_path = audio_dir / f"{cid}.wav"
            if not wav_path.is_file():
                skipped_no_wav += 1
                continue
            payload: StagePayload = (
                cid,
                wav_path.resolve().as_posix(),
                (out_dir / f"{cid}.wav").as_posix(),
                args.method,
                args.overwrite,
            )
            kind, _cid, detail = _stage_worker(payload)
            if kind == "staged":
                staged += 1
            elif kind == "skipped_existing":
                skipped_existing += 1
            else:
                staging_errors += 1
                print(f"# staging error {_cid}: {detail}", file=sys.stderr)
        print(
            f"# partition={name} method={args.method} workers=1 staged={staged} "
            f"skipped_existing={skipped_existing} no_match={skipped_no_match} "
            f"no_wav={skipped_no_wav} staging_errors={staging_errors}",
            file=sys.stderr,
        )
        print(f"# output dir: {out_dir.resolve()}", file=sys.stderr)
        return

    pending: list[StagePayload] = []

    for cid, labels in clip_iter_wrapped:
        if not set(labels) & whitelist:
            skipped_no_match += 1
            continue
        wav_path = audio_dir / f"{cid}.wav"
        if not wav_path.is_file():
            skipped_no_wav += 1
            continue
        pending.append(
            (
                cid,
                wav_path.resolve().as_posix(),
                (out_dir / f"{cid}.wav").as_posix(),
                args.method,
                args.overwrite,
            )
        )

    if not pending:
        print(
            f"# partition={name} method={args.method} workers={args.workers} staged=0 "
            f"skipped_existing=0 no_match={skipped_no_match} no_wav={skipped_no_wav} "
            f"staging_errors=0",
            file=sys.stderr,
        )
        print(f"# output dir: {out_dir.resolve()}", file=sys.stderr)
        return

    chunksize = max(1, len(pending) // (args.workers * 8))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        result_iter = executor.map(_stage_worker, pending, chunksize=chunksize)
        if not args.no_progress:
            result_iter = tqdm(
                result_iter,
                total=len(pending),
                desc=f"Stage {name} ({args.method})",
                unit="file",
                smoothing=0.05,
                file=sys.stderr,
            )
        for kind, cid, detail in result_iter:
            if kind == "staged":
                staged += 1
            elif kind == "skipped_existing":
                skipped_existing += 1
            else:
                staging_errors += 1
                print(f"# staging error {cid}: {detail}", file=sys.stderr)

    print(
        f"# partition={name} method={args.method} workers={args.workers} staged={staged} "
        f"skipped_existing={skipped_existing} no_match={skipped_no_match} "
        f"no_wav={skipped_no_wav} staging_errors={staging_errors}",
        file=sys.stderr,
    )
    print(f"# output dir: {out_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()