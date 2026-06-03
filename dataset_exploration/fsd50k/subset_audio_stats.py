#!/usr/bin/env python3
"""
Estimate FSD50K subset audio **after** vendored RAVE ``preprocess`` (LMDB), not raw WAV totals.

Uses the same whitelist / partition rules as ``build_subset.py``, ffprobe durations, and
simulates ``scripts/preprocess.py`` chunking + optional short-file packing
(``--concat_short``, ``--concat_seed``, ``--pad_short_remainder``, ``--lazy``).

Reports global stored vs discarded seconds and per-whitelist-tag breakdowns (multi-label
clips attribute the same estimated seconds to every intersecting tag, like the old script).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from fsd50k_manifest import iter_manifest_clips
from paths import PARTITION_CHOICES, canonical_partition, partitions_for
from tag_utils import normalize_tag

# Reuse vendored preprocess planning (shuffle + greedy packs).
_RAVE_ROOT = Path(__file__).resolve().parents[2] / "RAVE"
if _RAVE_ROOT.is_dir() and str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

from rave.preprocess_plan import (  # noqa: E402
    AudioProbe,
    build_preprocess_plan,
    expected_chunks_from_samples,
    min_samples_for_mode,
    samples_at_sr,
)


@dataclass
class ClipInfo:
    clip_id: str
    labels: list[str]
    duration_sec: float


@dataclass
class TagTotals:
    stored_sec: float = 0.0
    discarded_sec: float = 0.0
    discarded_full_clip_sec: float = 0.0
    discarded_tail_sec: float = 0.0
    discarded_pack_remainder_sec: float = 0.0
    discarded_pack_chunk_tail_sec: float = 0.0
    clips_with_stored: int = 0
    clips_fully_discarded: int = 0
    lmdb_rows: float = 0.0  # pro-rata row credit per tag


@dataclass
class PreprocessEstimate:
    min_samples: int
    chunk_sec: float
    train_window_sec: float
    input_sec: float = 0.0
    stored_sec: float = 0.0
    discarded_sec: float = 0.0
    lmdb_rows: int = 0
    discarded_full_clip_sec: float = 0.0
    discarded_tail_sec: float = 0.0
    discarded_pack_remainder_sec: float = 0.0
    discarded_pack_chunk_tail_sec: float = 0.0
    short_file_count: int = 0
    long_file_count: int = 0
    concat_packs: int = 0
    clips_fully_discarded: int = 0
    by_tag: dict[str, TagTotals] = field(default_factory=dict)


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


def _worker_duration(payload: tuple[str, str]) -> tuple[str, float | None]:
    cid, wav_s = payload
    return cid, ffprobe_duration_seconds(Path(wav_s))


def _tags_for_clip(labels: list[str], whitelist: set[str]) -> list[str]:
    return sorted(set(labels) & whitelist)


def _get_tag_totals(est: PreprocessEstimate, tag: str) -> TagTotals:
    if tag not in est.by_tag:
        est.by_tag[tag] = TagTotals()
    return est.by_tag[tag]


def _add_to_tags(
    est: PreprocessEstimate,
    tags: list[str],
    *,
    stored: float = 0.0,
    discarded: float = 0.0,
    full_clip: float = 0.0,
    tail: float = 0.0,
    pack_remainder: float = 0.0,
    pack_chunk_tail: float = 0.0,
    lmdb_rows: float = 0.0,
    clip_had_stored: bool = False,
    clip_fully_discarded: bool = False,
) -> None:
    for tag in tags:
        t = _get_tag_totals(est, tag)
        t.stored_sec += stored
        t.discarded_sec += discarded
        t.discarded_full_clip_sec += full_clip
        t.discarded_tail_sec += tail
        t.discarded_pack_remainder_sec += pack_remainder
        t.discarded_pack_chunk_tail_sec += pack_chunk_tail
        t.lmdb_rows += lmdb_rows
        if clip_had_stored:
            t.clips_with_stored += 1
        if clip_fully_discarded:
            t.clips_fully_discarded += 1


def simulate_preprocess_clean(
    clips: list[ClipInfo],
    *,
    sr: int,
    num_signal: int,
    lazy: bool,
    concat_short: bool,
    concat_seed: int,
    pad_short_remainder: bool,
    whitelist: set[str],
) -> PreprocessEstimate:
    """Simulate preprocess and attribute stored/discarded audio per clip and tag."""
    min_samples = min_samples_for_mode(num_signal, lazy)
    chunk_sec = min_samples / sr
    train_window_sec = num_signal / sr

    est = PreprocessEstimate(
        min_samples=min_samples,
        chunk_sec=chunk_sec,
        train_window_sec=train_window_sec,
        input_sec=sum(c.duration_sec for c in clips),
    )

    probes = [
        AudioProbe(path=c.clip_id, length_sec=c.duration_sec, channels=1)
        for c in clips
    ]
    clip_by_path = {c.clip_id: c for c in clips}

    plan = build_preprocess_plan(
        probes,
        sr,
        num_signal,
        lazy=lazy,
        concat_short=concat_short,
        concat_seed=concat_seed,
        pad_short_remainder=pad_short_remainder,
    )

    est.short_file_count = plan.short_file_count
    est.long_file_count = len(plan.long_files)
    est.concat_packs = len(plan.packs)

    clip_stored: dict[str, float] = defaultdict(float)
    clip_discard: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    def _bump_discard(cid: str, kind: str, sec: float) -> None:
        clip_discard[cid][kind] += sec
        est.discarded_sec += sec
        if kind == "full_clip":
            est.discarded_full_clip_sec += sec
        elif kind == "tail":
            est.discarded_tail_sec += sec
        elif kind == "pack_remainder":
            est.discarded_pack_remainder_sec += sec
        elif kind == "pack_chunk_tail":
            est.discarded_pack_chunk_tail_sec += sec

    def _attribute_clip(cid: str, tags: list[str]) -> None:
        stored = clip_stored[cid]
        disc = sum(clip_discard[cid].values())
        fully = stored <= 0 and disc > 0
        if fully:
            est.clips_fully_discarded += 1
        _add_to_tags(
            est,
            tags,
            stored=stored,
            discarded=disc,
            full_clip=clip_discard[cid]["full_clip"],
            tail=clip_discard[cid]["tail"],
            pack_remainder=clip_discard[cid]["pack_remainder"],
            pack_chunk_tail=clip_discard[cid]["pack_chunk_tail"],
            lmdb_rows=stored / chunk_sec if chunk_sec else 0.0,
            clip_had_stored=stored > 0,
            clip_fully_discarded=fully,
        )

    # Long files
    for probe in plan.long_files:
        n = samples_at_sr(probe.length_sec, sr)
        stored_samples = (n // min_samples) * min_samples
        tail_samples = n % min_samples
        clip_stored[probe.path] = stored_samples / sr
        est.stored_sec += stored_samples / sr
        est.lmdb_rows += expected_chunks_from_samples(n, min_samples)
        if tail_samples:
            _bump_discard(probe.path, "tail", tail_samples / sr)

    # Packs
    for pack in plan.packs:
        member_ids = [m.path for m in pack.members]
        member_samples = [samples_at_sr(m.length_sec, sr) for m in pack.members]
        total_raw = sum(member_samples) + pack.pad_samples
        is_remainder = sum(member_samples) < min_samples and pack.pad_samples == 0

        if is_remainder and not pad_short_remainder:
            for mid, ms in zip(member_ids, member_samples):
                _bump_discard(mid, "pack_remainder", ms / sr)
            continue

        stored_samples = (total_raw // min_samples) * min_samples
        tail_samples = total_raw % min_samples
        est.stored_sec += stored_samples / sr
        est.lmdb_rows += expected_chunks_from_samples(total_raw, min_samples)

        if total_raw <= 0:
            continue
        for mid, ms in zip(member_ids, member_samples):
            frac = ms / total_raw
            clip_stored[mid] += (stored_samples / sr) * frac
            if tail_samples:
                _bump_discard(mid, "pack_chunk_tail", (tail_samples / sr) * frac)

    # No concat: every short file fully discarded
    if not concat_short:
        for c in clips:
            if samples_at_sr(c.duration_sec, sr) < min_samples:
                _bump_discard(c.clip_id, "full_clip", c.duration_sec)

    # Per-tag attribution (once per clip)
    for c in clips:
        tags = _tags_for_clip(c.labels, whitelist)
        if not tags:
            continue
        _attribute_clip(c.clip_id, tags)

    return est


def _print_bullet_key(items: list[tuple[str, str]]) -> None:
    """Print column/field definitions as a markdown bullet list."""
    for label, description in items:
        print(f"- **{label}** — {description}")
    print()


def _print_report_intro(compare_concat: bool) -> None:
    print("### How to read this report")
    print()
    print(
        "This script measures your FSD50K WAVs with ffprobe, then **simulates** what "
        "`python RAVE/scripts/preprocess.py` would write into an LMDB database. "
        "It does **not** open a preprocessed LMDB; it predicts row counts and how much "
        "audio is kept vs thrown away."
    )
    print()
    print(
        "RAVE preprocess (non-lazy, the usual BRAVE path) reads each file in fixed-size "
        "chunks. With `--num_signal 131072` and 44100 Hz, each LMDB entry holds "
        "**262144 samples (~5.94 s)**. Training later takes a shorter random crop "
        "(**131072 samples ~2.97 s**) from each entry via `RandomCrop` in `rave/dataset.py`."
    )
    print()
    if compare_concat:
        print(
            "You ran **`--compare-concat`**, so the report shows **two scenarios** for the "
            "same clips:"
        )
        print()
        print(
            "1. **Concat ON** — matches default preprocess (`--concat_short`): clips shorter "
            "than one LMDB row are shuffled (seed `--concat_seed`), concatenated in groups "
            "until the group is long enough, then chunked."
        )
        print(
            "2. **Concat OFF** — matches `--noconcat_short`: every clip shorter than one row "
            "is dropped with no packing."
        )
        print()
        print(
            "Compare **Stored in LMDB** and **Clips fully discarded** between the two sections "
            "to see how much packing recovers short FSD50K clips."
        )
        print()
    print("**Discard types** (used in the summary and per-tag discarded tables):")
    print()
    _print_bullet_key([
        (
            "Full clip",
            "The entire WAV produces **zero** LMDB rows. Typical when concat is off and the "
            "clip is too short, or concat is on but the clip only appears in a **final pack** "
            "that never reached one row (unless `--pad_short_remainder`).",
        ),
        (
            "Long tail",
            "The clip is long enough to process alone. All full rows are saved; any samples "
            "left at the end of the file (less than one row) are trimmed off.",
        ),
        (
            "Pack remainder",
            "**Concat on only.** After grouping short clips, the last group still does not add "
            "up to one row, so the whole group is discarded.",
        ),
        (
            "Pack chunk tail",
            "**Concat on only.** A group was long enough to create at least one LMDB row, but "
            "after concatenating clips and cutting row-sized blocks, the **leftover audio at the "
            "end of that group** is still shorter than one row and is dropped (like a long-tail "
            "trim, but on glued-together shorts).",
        ),
    ])
    print(
        "**Multi-label clips:** FSD50K clips can have several tags. Per-tag tables attribute "
        "the same clip's stored or discarded time to **each** whitelist tag on that clip, so "
        "tag hours can sum to more than the global total."
    )
    print()


def _format_scenario_flags(args: argparse.Namespace) -> str:
    mode = "lazy" if args.lazy else "non-lazy"
    concat = "on" if args.concat_short else "off"
    pad = "yes" if args.pad_short_remainder else "no"
    return (
        f"lazy={mode}, concat_short={concat}, concat_seed={args.concat_seed}, "
        f"pad_short_remainder={pad}"
    )


def _print_estimate(
    est: PreprocessEstimate,
    *,
    title: str,
    scenario: str,
    num_signal: int,
    sr: int,
) -> None:
    util = (
        100.0 * est.stored_sec / est.input_sec if est.input_sec > 0 else 0.0
    )
    train_h = est.lmdb_rows * est.train_window_sec / 3600.0
    lmdb_h = est.lmdb_rows * est.chunk_sec / 3600.0

    print(f"### {title}")
    print()
    print(f"**Preprocess flags simulated:** `{scenario}`")
    print()
    print(
        "This section answers: *if I preprocess this subset with these flags, how much audio "
        "lands in the LMDB, how much is lost, and why?* Use it as a budget before running "
        "`RAVE/scripts/preprocess.py` on disk."
    )
    print()
    print("#### Scenario summary (global totals)")
    print()
    print(
        "One row per quantity below. **Stored** is what survives as full LMDB chunks; "
        "**Discarded** breaks down into the four mechanisms defined in the introduction."
    )
    print()
    _print_bullet_key([
        (
            "`num_signal`",
            f"`--num_signal` / `--n_signal` for training (here {num_signal}). Preprocess "
            f"non-lazy row size is still 2× this value.",
        ),
        (
            "LMDB row size (samples)",
            f"Samples per LMDB entry ({est.min_samples} = 2×num_signal in non-lazy mode).",
        ),
        (
            f"LMDB row duration @ {sr} Hz",
            f"Seconds of audio in each LMDB entry ({est.chunk_sec:.4f} s).",
        ),
        (
            "Train crop duration",
            f"Seconds drawn per training step after `RandomCrop` "
            f"({est.train_window_sec:.4f} s).",
        ),
        (
            "Input audio (measured clips)",
            "Sum of ffprobe durations for all clips in this subset.",
        ),
        (
            "Stored in LMDB",
            "Total seconds that become complete LMDB rows (usable training material).",
        ),
        (
            "Discarded total",
            "Input minus stored; equals the four discard bullets below.",
        ),
        (
            "fully discarded clips",
            "Seconds from files/clips that produced **no** LMDB rows.",
        ),
        (
            "long-file tail (last partial chunk)",
            "Seconds trimmed off the end of **long** files.",
        ),
        (
            "short pack remainder (unpackable)",
            "Seconds in the final short pack that never reached one row.",
        ),
        (
            "pack concat chunk tail",
            "Seconds trimmed after chunking a **successful** short pack.",
        ),
        (
            "LMDB rows (estimated)",
            "Count of LMDB keys (`00000000`, …) written.",
        ),
        (
            "LMDB row-hours (stored window)",
            "`rows × row duration` — hours of waveform actually stored.",
        ),
        (
            "Train crop-hours (1 crop / row)",
            "`rows × train crop` — rough training exposure if each row is seen once.",
        ),
        (
            f"Short files (< one row)",
            f"Clips shorter than {est.chunk_sec:.4f} s by themselves (candidates for packing "
            f"when concat is on).",
        ),
        (
            "Long files (≥ one row alone)",
            "Clips that already fill at least one LMDB row without packing.",
        ),
        (
            "Concat packs",
            "How many short-clip groups were concatenated before chunking (0 if concat off).",
        ),
        (
            "Clips fully discarded (0 stored)",
            "Number of clips with **zero** stored seconds.",
        ),
        (
            "Utilization (stored / input)",
            "Percentage of raw duration that becomes LMDB audio.",
        ),
    ])
    print("| Metric | Value |")
    print(f"| --- | --- |")
    print(f"| `num_signal` (train crop samples) | {num_signal} |")
    print(f"| LMDB row size (samples) | {est.min_samples} |")
    print(f"| LMDB row duration @ {sr} Hz | `{est.chunk_sec:.6f}` s |")
    print(f"| Train crop duration | `{est.train_window_sec:.6f}` s |")
    print(f"| Input audio (measured clips) | `{est.input_sec:.2f}` s (`{est.input_sec / 3600:.4f}` h) |")
    print(f"| **Stored in LMDB** | `{est.stored_sec:.2f}` s (`{est.stored_sec / 3600:.4f}` h) |")
    print(f"| **Discarded total** | `{est.discarded_sec:.2f}` s (`{est.discarded_sec / 3600:.4f}` h) |")
    print(f"| └ fully discarded clips | `{est.discarded_full_clip_sec:.2f}` s |")
    print(f"| └ long-file tail (last partial chunk) | `{est.discarded_tail_sec:.2f}` s |")
    print(f"| └ short pack remainder (unpackable) | `{est.discarded_pack_remainder_sec:.2f}` s |")
    print(f"| └ pack concat chunk tail | `{est.discarded_pack_chunk_tail_sec:.2f}` s |")
    print(f"| LMDB rows (estimated) | `{est.lmdb_rows}` |")
    print(f"| LMDB row-hours (stored window) | `~{lmdb_h:.4f}` h |")
    print(f"| Train crop-hours (1 crop / row) | `~{train_h:.4f}` h |")
    print(f"| Short files (< one row) | {est.short_file_count} |")
    print(f"| Long files (≥ one row alone) | {est.long_file_count} |")
    print(f"| Concat packs | {est.concat_packs} |")
    print(f"| Clips fully discarded (0 stored) | {est.clips_fully_discarded} |")
    print(f"| Utilization (stored / input) | `{util:.1f}%` |")
    print()

    print("#### Per-tag: stored in LMDB (estimated)")
    print()
    print(
        "This table shows **how much training material each ontology tag contributes** after "
        "preprocess. It helps compare tags (e.g. `water` vs `raindrop`) when building class-"
        "conditioned subsets. If a clip has multiple whitelist labels, its stored time is "
        "counted toward **each** of those tags."
    )
    print()
    _print_bullet_key([
        ("Tag", "Label from your `--whitelist` file (FSD50K ontology token)."),
        (
            "Clips w/ stored audio",
            "How many clips carrying this tag still have **some** audio in the LMDB "
            "(not fully discarded).",
        ),
        (
            "LMDB hours (rows×row)",
            "Estimated LMDB row-hours credited to this tag (pro-rata when clips were packed "
            "together).",
        ),
        (
            "Stored hours",
            "Wall-clock hours of waveform from this tag that survive preprocess (should match "
            "LMDB hours up to rounding).",
        ),
    ])
    rows_stored = sorted(est.by_tag.items(), key=lambda x: (-x[1].stored_sec, x[0]))
    print("| Tag | Clips w/ stored audio | LMDB hours (rows×row) | Stored hours |")
    print("| --- | ---: | ---: | ---: |")
    for tag, t in rows_stored:
        row_h = t.lmdb_rows * est.chunk_sec / 3600.0
        print(
            f"| `{tag}` | {t.clips_with_stored} | {row_h:.4f} | {t.stored_sec / 3600:.4f} |"
        )
    print()

    print("#### Per-tag: discarded (estimated)")
    print()
    print(
        "This table shows **where each tag loses audio**, using the same four discard types "
        "as the scenario summary. Use it to see whether a tag suffers mostly from short clips "
        "(Full clip / Pack remainder) or from end-trimming on longer recordings (Long tail / "
        "Pack chunk tail). All values are in **hours**."
    )
    print()
    _print_bullet_key([
        ("Tag", "Whitelist label."),
        (
            "Full clip",
            "Hours lost because clips with this tag never produced any LMDB row.",
        ),
        (
            "Long tail",
            "Hours trimmed from the **end** of long clips tagged with this label.",
        ),
        (
            "Pack remainder",
            "Hours from this tag's clips that were only in a **final** short pack too small "
            "to store (concat on).",
        ),
        (
            "Pack chunk tail",
            "Hours trimmed after chunking a concat pack that included this tag.",
        ),
        (
            "Total discarded h",
            "Sum of the four discard columns above for this tag.",
        ),
    ])
    print("| Tag | Full clip | Long tail | Pack remainder | Pack chunk tail | Total discarded h |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    rows_disc = sorted(est.by_tag.items(), key=lambda x: (-x[1].discarded_sec, x[0]))
    for tag, t in rows_disc:
        print(
            f"| `{tag}` | {t.discarded_full_clip_sec / 3600:.4f} | "
            f"{t.discarded_tail_sec / 3600:.4f} | "
            f"{t.discarded_pack_remainder_sec / 3600:.4f} | "
            f"{t.discarded_pack_chunk_tail_sec / 3600:.4f} | "
            f"{t.discarded_sec / 3600:.4f} |"
        )
    print()

    print("#### Per-tag: clips fully discarded (0 LMDB audio)")
    print()
    print(
        "This table counts **clips** (not hours) that contribute **nothing** to the LMDB for "
        "each tag—every second of that clip is lost. With concat off, this is usually all "
        "clips shorter than one row; with concat on, it should be small unless packing fails "
        "for the final group."
    )
    print()
    _print_bullet_key([
        ("Tag", "Whitelist label."),
        (
            "Clips fully discarded",
            "Number of clips with this tag that have **zero** stored audio after preprocess.",
        ),
    ])
    rows_fd = sorted(
        ((tag, t.clips_fully_discarded) for tag, t in est.by_tag.items()),
        key=lambda x: (-x[1], x[0]),
    )
    print("| Tag | Clips fully discarded |")
    print("| --- | ---: |")
    any_fd = False
    for tag, n in rows_fd:
        if n:
            any_fd = True
            print(f"| `{tag}` | {n} |")
    if not any_fd:
        print("| *(none)* | 0 |")
        print()
        print(
            "_Every clip in this scenario contributes at least some audio to the LMDB._"
        )
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--whitelist", required=True, type=Path)
    p.add_argument(
        "--partition",
        choices=list(PARTITION_CHOICES),
        default="dev_train",
    )
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--wav-root", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-signal", type=int, default=131072)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--tag-hours-tsv", type=Path, default=None)
    p.add_argument(
        "--lazy",
        action="store_true",
        help="Simulate lazy preprocess (min chunk = 1× n_signal, not 2×).",
    )
    p.add_argument(
        "--concat-short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simulate --concat_short (default: true, matches preprocess default).",
    )
    p.add_argument(
        "--concat-seed",
        type=int,
        default=42,
        help="Same as preprocess --concat_seed (affects pack groupings).",
    )
    p.add_argument(
        "--pad-short-remainder",
        action="store_true",
        help="Simulate --pad_short_remainder (zero-pad final short pack).",
    )
    p.add_argument(
        "--compare-concat",
        action="store_true",
        help="Print two scenarios: concat on (default) and concat off.",
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

    manifest_clips: list[tuple[str, list[str], Path]] = []
    for cid, labels in iter_manifest_clips(part):
        if not set(labels) & whitelist:
            continue
        w = wav_root / f"{cid}.wav"
        if not w.is_file():
            continue
        manifest_clips.append((cid, labels, w))

    if not manifest_clips:
        print("No matching clips with WAV files found.")
        sys.exit(1)

    cid_to_duration: dict[str, float] = {}
    probe_errors = 0
    payloads = [(cid, wav.resolve().as_posix()) for cid, _labels, wav in manifest_clips]

    if args.workers == 1:
        it = payloads
        if not args.no_progress:
            it = tqdm(payloads, desc="ffprobe", unit="file", file=sys.stderr)
        for cid, wav_s in it:
            d = ffprobe_duration_seconds(Path(wav_s))
            if d is None or d <= 0:
                probe_errors += 1
                continue
            cid_to_duration[cid] = d
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker_duration, pl): pl[0] for pl in payloads}
            iter_f = as_completed(futures)
            if not args.no_progress:
                iter_f = tqdm(
                    iter_f, total=len(futures), desc="ffprobe", unit="file", file=sys.stderr
                )
            for fut in iter_f:
                cid, d = fut.result()
                if d is None or d <= 0:
                    probe_errors += 1
                    continue
                cid_to_duration[cid] = d

    if not cid_to_duration:
        print("All ffprobe attempts failed.")
        sys.exit(1)

    clip_infos = [
        ClipInfo(cid, labels, cid_to_duration[cid])
        for cid, labels, _ in manifest_clips
        if cid in cid_to_duration
    ]

    scenarios: list[tuple[str, bool, bool]] = []
    if args.compare_concat:
        scenarios = [
            ("Concat ON (default preprocess)", True, args.pad_short_remainder),
            ("Concat OFF (--noconcat_short)", False, False),
        ]
    else:
        scenarios = [
            ("Requested preprocess settings", args.concat_short, args.pad_short_remainder),
        ]

    print("## FSD50K subset — post-preprocess (LMDB) estimates")
    print()
    print(
        "Subset statistics for clips matching your whitelist on disk. "
        "Sections below estimate **post-preprocess** yield (LMDB), not raw WAV totals alone."
    )
    print()
    print("#### Run setup (input subset)")
    print()
    print(
        "This small table describes **which files were measured** before any preprocess "
        "simulation. *Raw input duration* is the sum of ffprobe lengths; everything else in "
        "the report applies preprocess rules to that pile of audio."
    )
    print()
    _print_bullet_key([
        ("Partition", "FSD50K CSV split (`dev_train`, etc.)."),
        ("WAV root", "Folder of `<clip_id>.wav` files analyzed."),
        (
            "Clips in report",
            "Manifest rows with a whitelist tag and an existing WAV.",
        ),
        (
            "ffprobe failures",
            "Files that could not be measured (excluded from simulation).",
        ),
        (
            "Raw input duration",
            "Total seconds of audio **before** preprocess (on-disk WAV).",
        ),
    ])
    print("| Field | Value |")
    print("| --- | --- |")
    print(f"| Partition | `{name}` |")
    print(f"| WAV root | `{wav_root.resolve()}` |")
    print(f"| Clips in report | {len(clip_infos)} |")
    print(f"| ffprobe failures | {probe_errors} |")
    print(f"| Raw input duration | `{sum(c.duration_sec for c in clip_infos):.2f}` s |")
    print()
    _print_report_intro(args.compare_concat)

    all_estimates: list[tuple[str, PreprocessEstimate]] = []

    for title, concat_short, pad_remainder in scenarios:
        est = simulate_preprocess_clean(
            clip_infos,
            sr=args.sample_rate,
            num_signal=args.n_signal,
            lazy=args.lazy,
            concat_short=concat_short,
            concat_seed=args.concat_seed,
            pad_short_remainder=pad_remainder,
            whitelist=whitelist,
        )
        scenario = _format_scenario_flags(
            argparse.Namespace(
                lazy=args.lazy,
                concat_short=concat_short,
                concat_seed=args.concat_seed,
                pad_short_remainder=pad_remainder,
            )
        )
        _print_estimate(
            est,
            title=title,
            scenario=scenario,
            num_signal=args.n_signal,
            sr=args.sample_rate,
        )
        all_estimates.append((title, est))

    if args.tag_hours_tsv and all_estimates:
        args.tag_hours_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.tag_hours_tsv.open("w", encoding="utf-8") as fh:
            fh.write(
                "scenario\ttag\tstored_hours\tdiscarded_hours\t"
                "discarded_full_clip_hours\tdiscarded_tail_hours\t"
                "discarded_pack_remainder_hours\tdiscarded_pack_chunk_tail_hours\t"
                "lmdb_row_hours\tclips_with_stored\tclips_fully_discarded\n"
            )
            for scen_name, est in all_estimates:
                for tag, t in sorted(est.by_tag.items()):
                    fh.write(
                        f"{scen_name}\t{tag}\t{t.stored_sec / 3600:.6f}\t"
                        f"{t.discarded_sec / 3600:.6f}\t"
                        f"{t.discarded_full_clip_sec / 3600:.6f}\t"
                        f"{t.discarded_tail_sec / 3600:.6f}\t"
                        f"{t.discarded_pack_remainder_sec / 3600:.6f}\t"
                        f"{t.discarded_pack_chunk_tail_sec / 3600:.6f}\t"
                        f"{t.lmdb_rows * est.chunk_sec / 3600:.6f}\t"
                        f"{t.clips_with_stored}\t{t.clips_fully_discarded}\n"
                    )
        print(f"Wrote TSV: `{args.tag_hours_tsv.resolve()}`")


if __name__ == "__main__":
    main()
