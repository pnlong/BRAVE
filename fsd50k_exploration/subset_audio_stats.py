#!/usr/bin/env python3
"""
Summarize durations for FSD50K clips selected by the same whitelist + partition rules as
``build_subset.py``.

Reports:

* Raw WAV length distribution (seconds / hours) on disk.
* How many clips are **too short** for one full non-lazy preprocess chunk (``2 * num_signal``
  samples per LMDB row).
* Estimated **LMDB rows** after non-lazy preprocess (non-overlapping chunks of
  ``2 * num_signal`` samples; trailing samples are **dropped**). With vendored
  ``--concat_short`` (default), short clips are packed before chunking—this script reports
  **stock** per-file math unless you re-run after packing.
* **Hours per ontology tag** (each row in ``--whitelist`` treated as one "prompt" / label):
  summed clip duration over all clips containing that token (multi-label clips contribute their
  full duration to **every** tag they hit—standard if each tag defines a conditioning class).

Training-time behavior documented here matches vendored RAVE defaults used with
``configs/brave.gin`` (44100 Hz) unless you override ``--sample-rate``, ``--n-signal``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from fsd50k_manifest import iter_manifest_clips
from paths import PARTITION_CHOICES, canonical_partition, partitions_for
from tag_utils import normalize_tag


def load_whitelist(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        t = normalize_tag(line)
        if t:
            out.add(t)
    if not out:
        raise ValueError(f"Whitelist appears empty after parsing: {path}")
    return out


def ffprobe_duration_seconds(path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def expected_preprocess_chunks(duration_sec: float, sample_rate: int, num_signal: int) -> int:
    """Non-lazy LMDB rows (``2 * num_signal`` samples per row)."""
    if duration_sec <= 0:
        return 0
    n_samples = int(math.floor(duration_sec * sample_rate))
    return n_samples // (2 * num_signal)


def _worker_duration(payload: tuple[str, str]) -> tuple[str, float | None]:
    cid, wav_s = payload
    d = ffprobe_duration_seconds(Path(wav_s))
    return cid, d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        "--whitelist",
        required=True,
        type=Path,
        help="Same whitelist text file as ``build_subset.py`` (one normalized tag per line).",
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
        help=(
            "Directory holding ``<clip_id>.wav`` files to measure (default: official partition "
            "audio folder under ``--dataset-root``). Set to ``fsd50k_brave/audio_subset`` after "
            "staging to stats only WAVs actually used."
        ),
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sampling rate assumed for chunk math (RAVE/BRAVE configs use 44100 unless changed).",
    )
    p.add_argument(
        "--n-signal",
        type=int,
        default=131072,
        help="``--num_signal`` / ``--n_signal`` chunk length in samples "
        "(RAVE preprocess + train defaults).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel ffprobe worker processes (default: 8; use 1 for gentle NAS use).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm.",
    )
    p.add_argument(
        "--tag-hours-tsv",
        type=Path,
        default=None,
        help="If set, write columns tag, clip_count, hours.",
    )
    ns = p.parse_args()
    if ns.workers < 1:
        p.error("--workers must be >= 1")
    return ns


def main() -> None:
    args = parse_args()
    name = canonical_partition(args.partition)
    part = partitions_for(args.dataset_root)[name]
    whitelist = load_whitelist(args.whitelist)

    wav_root = args.wav_root if args.wav_root is not None else part.audio_dir
    if not part.csv_path.is_file():
        print(f"error: CSV manifest missing: {part.csv_path}", file=sys.stderr)
        sys.exit(1)

    clips: list[tuple[str, list[str], Path]] = []
    for cid, labels in iter_manifest_clips(part):
        if not set(labels) & whitelist:
            continue
        w = wav_root / f"{cid}.wav"
        if not w.is_file():
            continue
        clips.append((cid, labels, w))

    n_signal = args.n_signal
    sr = args.sample_rate
    chunk_sec = (2 * n_signal) / sr
    train_window_sec = n_signal / sr

    if not clips:
        print("No matching clips with WAV files found (check whitelist / wav-root / partition).")
        sys.exit(1)

    # --- durations ---
    durations: list[float] = []
    cid_to_duration: dict[str, float] = {}
    probe_errors = 0
    payloads = [(cid, wav.resolve().as_posix()) for cid, _labels, wav in clips]

    if args.workers == 1:
        it = payloads
        if not args.no_progress:
            it = tqdm(payloads, desc="ffprobe", unit="file", file=sys.stderr)
        for cid, wav_s in it:
            d = ffprobe_duration_seconds(Path(wav_s))
            if d is None or d <= 0:
                probe_errors += 1
                continue
            durations.append(d)
            cid_to_duration[cid] = d
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker_duration, pl): pl[0] for pl in payloads}
            iter_f = as_completed(futures)
            if not args.no_progress:
                iter_f = tqdm(iter_f, total=len(futures), desc="ffprobe", unit="file", file=sys.stderr)
            for fut in iter_f:
                cid, d = fut.result()
                if d is None or d <= 0:
                    probe_errors += 1
                    continue
                durations.append(d)
                cid_to_duration[cid] = d

    if not durations:
        print("All ffprobe attempts failed.")
        sys.exit(1)

    sorted_d = sorted(durations)
    n = len(sorted_d)
    sum_sec = sum(sorted_d)

    short_clips = sum(1 for d in sorted_d if d < chunk_sec)
    chunks_total = sum(expected_preprocess_chunks(d, sr, n_signal) for d in sorted_d)

    intersect_labels = defaultdict(float)  # tag -> seconds (only whitelist tags present on clip)
    tag_clip_count = defaultdict(int)

    for cid, labels, _wav in clips:
        d = cid_to_duration.get(cid)
        if d is None:
            continue
        seen_tags = sorted(set(labels) & whitelist)
        if not seen_tags:
            continue
        for t in seen_tags:
            intersect_labels[t] += d
            tag_clip_count[t] += 1

    # --- printable report ---
    print("## FSD50K subset audio statistics")
    print()
    print(f"| Field | Value |")
    print(f"| --- | --- |")
    print(f"| Partition | `{name}` |")
    print(f"| Manifest | `{part.csv_path}` |")
    print(f"| WAV root | `{wav_root.resolve()}` |")
    print(f"| Whitelist tags (count) | {len(whitelist)} |")
    print(f"| Clips matched manifest ∩ whitelist ∩ wav exists | {len(clips)} |")
    print(f"| Clips measured (ffprobe ok) | {n} |")
    print(f"| ffprobe failures / missing duration | {probe_errors} |")
    print(f"| Total measured duration | {sum_sec:.2f} s (~{sum_sec / 3600:.4f} h) |")
    print()
    if args.sample_rate != 44100:
        print(
            f"*(Chunk math uses `--sample-rate {args.sample_rate}`; `configs/brave.gin` defaults to "
            "44100 Hz.)*"
        )
        print()

    print("### Raw WAV length (seconds)")
    print(f"- min: `{sorted_d[0]:.4f}`")
    print(f"- max: `{sorted_d[-1]:.4f}`")
    print(f"- mean: `{sum_sec / n:.4f}`")
    print(f"- stdev: `{statistics.pstdev(sorted_d) if n > 1 else 0.0:.4f}`")
    print(f"- median: `{statistics.median(sorted_d):.4f}`")
    print(f"- p10 / p90: `{percentile(sorted_d, 10):.4f}` / `{percentile(sorted_d, 90):.4f}`")
    print()

    print("### RAVE preprocessing + training window (non-lazy pipeline)")
    print()
    print(
        f"Preprocess **`--num_signal {n_signal}`** writes LMDB rows of **`{2 * n_signal}`** "
        f"samples (**`{chunk_sec:.6f}` s** at **{sr} Hz**); only full rows are kept "
        "(see `scripts/preprocess.py`). Short files can be packed with **`--concat_short`** "
        "(default on)."
    )
    print(
        f"Train **`--n_signal {n_signal}`** crops **`{train_window_sec:.6f}` s** from each "
        f"**`{2 * n_signal}`**-sample LMDB buffer via `RandomCrop` in `rave.dataset.get_dataset`."
    )
    print(
        f"- **`brave.gin`**: `SAMPLING_RATE = 44100` Hz (match `--sample-rate` for chunk math); "
        f"causal conv paddings (`cc.get_padding.mode = 'causal'`); `valid_signal_crop = False`."
    )
    print("- **`dataset.split_dataset.max_residual = 1000`**: caps the **validation** split size "
          "(train/val **example counts**, not waveform padding).")
    print()
    print(f"| Chop metric | Value |")
    print(f"| --- | --- |")
    print(f"| `num_signal` / `n_signal` (samples) | {n_signal} |")
    print(f"| LMDB row duration (2×num_signal) | `{chunk_sec:.6f}` s |")
    print(f"| Train crop (num_signal) | `{train_window_sec:.6f}` s |")
    print(f"| Clips shorter than one LMDB row (< `{chunk_sec:.6f}` s) | {short_clips} |")
    print(f"| Sum of floor(duration×sr)//(2×num_signal) over clips | `{chunks_total}` LMDB rows |")
    print(
        f"| Approx train hours (one crop/row) | "
        f"`~{(chunks_total * train_window_sec) / 3600:.4f}` h |"
    )
    print()

    print("### Hours per whitelist tag (ontology token / pseudo-prompt)")
    print()
    print(
        "Each clip contributes its **full** measured duration once **per intersecting whitelist tag** "
        "(multi-label rows add to several tags)."
    )
    print()
    rows = sorted(intersect_labels.items(), key=lambda x: (-x[1], x[0]))
    print(f"| Tag | Clip count | Hours (approx) |")
    print("| --- | ---: | ---: |")
    for tag, sec in rows:
        print(f"| `{tag}` | {tag_clip_count[tag]} | {sec / 3600:.4f} |")

    if args.tag_hours_tsv:
        args.tag_hours_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.tag_hours_tsv.open("w", encoding="utf-8") as fh:
            fh.write("tag\tclip_count\thours\n")
            for tag, sec in rows:
                fh.write(f"{tag}\t{tag_clip_count[tag]}\t{sec / 3600:.6f}\n")
        print()
        print(f"Wrote TSV: `{args.tag_hours_tsv.resolve()}`")


if __name__ == "__main__":
    main()
