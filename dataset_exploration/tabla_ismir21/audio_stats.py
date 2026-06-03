#!/usr/bin/env python3
"""
Scan Tabla ISMIR train/test WAV trees: ffprobe durations and optional LMDB yield estimate.

Uses the same RAVE preprocess planning as ``fsd50k/subset_audio_stats.py`` (non-lazy,
``--num_signal`` 131072 @ 44100 Hz by default).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from paths import SPLIT_CHOICES, split_audio_dir

_RAVE_ROOT = Path(__file__).resolve().parents[2] / "RAVE"
if _RAVE_ROOT.is_dir() and str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

from rave.preprocess_plan import (  # noqa: E402
    AudioProbe,
    build_preprocess_plan,
    count_plan_chunks,
    min_samples_for_mode,
)


def ffprobe_duration(path: str) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            text=True,
        ).strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def collect_probes(split: str, wav_root: Path | None, workers: int) -> list[AudioProbe]:
    root = Path(wav_root) if wav_root is not None else split_audio_dir(split)
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        return []

    def _one(p: Path) -> AudioProbe | None:
        dur = ffprobe_duration(str(p))
        if dur is None:
            return None
        return AudioProbe(path=str(p.resolve()), length_sec=dur, channels=1)

    probes: list[AudioProbe] = []
    if workers <= 1:
        for p in tqdm(wavs, desc="ffprobe", unit="file"):
            pr = _one(p)
            if pr is not None:
                probes.append(pr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, p): p for p in wavs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="ffprobe"):
                pr = fut.result()
                if pr is not None:
                    probes.append(pr)
    return probes


def stroke_class_for(path: str, split_root: Path) -> str:
    rel = Path(path).resolve().relative_to(split_root.resolve())
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "(root)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument("--split", choices=list(SPLIT_CHOICES), default="train")
    p.add_argument("--wav-root", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--n-signal", type=int, default=131072)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--compare-concat",
        action="store_true",
        help="Print estimates with concat_short on and off.",
    )
    return p.parse_args()


def print_estimate(
    label: str,
    probes: list[AudioProbe],
    sr: int,
    n_signal: int,
    concat_short: bool,
    by_class: dict[str, float],
) -> None:
    min_samples = min_samples_for_mode(n_signal, lazy=False)
    plan = build_preprocess_plan(
        probes,
        sr,
        n_signal,
        lazy=False,
        concat_short=concat_short,
        concat_seed=42,
        pad_short_remainder=False,
    )
    rows = count_plan_chunks(plan, sr, n_signal, lazy=False)
    stored_sec = rows * (2 * n_signal) / sr
    input_sec = sum(p.length_sec for p in probes)
    print(f"\n=== {label} (concat_short={'on' if concat_short else 'off'}) ===")
    print(f"  clips probed:     {len(probes)}")
    print(f"  raw duration:     {input_sec:.2f} s ({input_sec / 3600:.3f} h)")
    print(f"  LMDB rows:        {rows}")
    print(f"  stored (est.):    {stored_sec:.2f} s ({stored_sec / 3600:.3f} h)")
    print(f"  discarded (est.): {max(0.0, input_sec - stored_sec):.2f} s")
    if by_class:
        print("  per stroke class (raw seconds):")
        for name in sorted(by_class):
            print(f"    {name:24s} {by_class[name]:8.2f} s")


def main() -> None:
    args = parse_args()
    root = Path(args.wav_root) if args.wav_root is not None else split_audio_dir(args.split)
    probes = collect_probes(args.split, args.wav_root, args.workers)
    if not probes:
        print(f"No WAV under {root}", file=sys.stderr)
        sys.exit(1)

    by_class: dict[str, float] = defaultdict(float)
    for pr in probes:
        by_class[stroke_class_for(pr.path, root)] += pr.length_sec

    print(f"Split:      {args.split}")
    print(f"WAV root:   {root.resolve()}")
    print(f"sr={args.sample_rate}  n_signal={args.n_signal}")

    scenarios = [True, False] if args.compare_concat else [True]
    for concat in scenarios:
        print_estimate(
            f"preprocess estimate",
            probes,
            args.sample_rate,
            args.n_signal,
            concat,
            by_class,
        )


if __name__ == "__main__":
    main()
