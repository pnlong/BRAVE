#!/usr/bin/env python3
"""Classify tap WAVs by mic and optionally symlink a training subset.

Writes ``mic_type.yaml`` (path → shotgun | contact | sm57) beside the audio
root. ``--stage-mic shotgun`` builds a directory of symlinks for preprocess.

    python scripts/stage_tap_mic_subset.py \\
      --audio-dir $BRAVE_STORAGE/tap_samples/audio_subset \\
      --stage-mic shotgun
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import yaml

_BRAVE = Path(__file__).resolve().parents[1]
_TAP = _BRAVE / "dataset_exploration" / "tap_samples"
if str(_TAP) not in sys.path:
    sys.path.insert(0, str(_TAP))

from mic_type import (  # noqa: E402
    MIC_TYPES,
    classify_dir,
    counts,
    files_for_mic,
    rename_tap_files,
    select_eval_holdouts,
    split_eval_times,
)

BRAVE_STORAGE = Path(
    os.environ.get("BRAVE_STORAGE", f"/data/hai-res/{os.environ.get('USER', 'p1long')}/BRAVE-data")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audio-dir",
        type=Path,
        default=BRAVE_STORAGE / "tap_samples" / "audio_subset",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="YAML output (default: <audio-dir>/../mic_type.yaml)",
    )
    p.add_argument(
        "--stage-mic",
        choices=MIC_TYPES,
        default=None,
        help="Symlink this mic class into --stage-dir",
    )
    p.add_argument(
        "--rename",
        action="store_true",
        help="Rewrite names to Shotgun- / Contact Mic L|R- / sm57- prefixes",
    )
    p.add_argument(
        "--holdout-eval",
        action="store_true",
        help="Stage one tail crop per style for listening eval; keep the "
        "file prefix in the train dir (see --eval-seconds)",
    )
    p.add_argument(
        "--eval-seconds",
        type=float,
        default=20.0,
        help="Seconds from the end of one take per style for eval "
        "(0 = hold out the whole file). Default: 20",
    )
    p.add_argument(
        "--hold-singletons",
        action="store_true",
        help="Also exclude styles that have only one file (light, sidetap)",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Default: sibling audio_subset_<mic>_eval",
    )
    p.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Default: sibling audio_subset_<mic>",
    )
    return p.parse_args()


def _wav_duration(path: Path) -> float:
    import subprocess

    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def _ffmpeg_pcm_slice(src: Path, dest: Path, *, ss: float, t: float) -> None:
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ss",
        f"{ss:.4f}",
        "-t",
        f"{t:.4f}",
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clear_audio_dir(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for extra in dest.iterdir():
        if extra.is_symlink() or extra.is_file():
            extra.unlink()


def stage_symlinks(files: Iterable[Path], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    keep = set()
    n = 0
    for src in files:
        link = dest / src.name
        keep.add(src.name)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())
        n += 1
    for extra in dest.iterdir():
        if extra.name not in keep and (extra.is_symlink() or extra.is_file()):
            extra.unlink()
    return n


def main() -> None:
    args = parse_args()
    audio_dir = args.audio_dir.resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio dir missing: {audio_dir}")
    if args.rename:
        pairs = rename_tap_files(audio_dir)
        print(f"renamed {len(pairs)} files in {audio_dir}")
        for src, dest in pairs:
            print(f"  {src.name} -> {dest.name}")
    mapping = classify_dir(audio_dir)
    c = counts(mapping)
    manifest = args.manifest or (audio_dir.parent / "mic_type.yaml")
    payload = {
        "audio_dir": str(audio_dir),
        "rules": {
            "shotgun": "filename starts with 'Shotgun-'",
            "contact": "filename starts with 'Contact Mic L-' or 'Contact Mic R-'",
            "sm57": "filename starts with 'sm57-'",
        },
        "counts": c,
        "files": mapping,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    print(f"wrote {manifest}")
    print("counts:", c)

    if args.stage_mic:
        dest = args.stage_dir or (audio_dir.parent / f"audio_subset_{args.stage_mic}")
        files = files_for_mic(audio_dir, args.stage_mic)  # type: ignore[arg-type]
        if not args.holdout_eval:
            n = stage_symlinks(files, dest)
            print(f"staged {n} {args.stage_mic} train files → {dest}")
            return
        eval_seconds = float(args.eval_seconds)
        # One take per style. Tail crop always covers singletons; whole-file
        # mode still skips singletons unless --hold-singletons.
        picks = select_eval_holdouts(
            files, hold_singletons=True if eval_seconds > 0 else args.hold_singletons
        )
        eval_dir = args.eval_dir or (
            audio_dir.parent / f"audio_subset_{args.stage_mic}_eval"
        )
        _clear_audio_dir(dest)
        _clear_audio_dir(eval_dir)
        split_src = {h.path.resolve() for h in picks}
        records: List[dict] = []
        n_eval = 0
        n_train = 0
        for h in picks:
            dur = _wav_duration(h.path)
            train_end, eval_start, eval_dur = split_eval_times(dur, eval_seconds)
            eval_path = eval_dir / h.path.name
            if eval_dur > 0:
                _ffmpeg_pcm_slice(h.path, eval_path, ss=eval_start, t=eval_dur)
                n_eval += 1
            if train_end > 0:
                train_path = dest / h.path.name
                _ffmpeg_pcm_slice(h.path, train_path, ss=0.0, t=train_end)
                n_train += 1
            records.append(
                {
                    "file": h.path.name,
                    "style": h.style,
                    "n_in_style": h.n_in_style,
                    "source_sec": round(dur, 3),
                    "train_prefix_sec": round(train_end, 3),
                    "eval_sec": round(eval_dur, 3),
                    "held_out_tail": True,
                }
            )
            print(
                f"  {h.style}: {h.path.name}  "
                f"train {train_end:.1f}s + eval {eval_dur:.1f}s "
                f"(of {dur:.1f}s)"
            )
        for src in files:
            if src.resolve() in split_src:
                continue
            link = dest / src.name
            link.symlink_to(src.resolve())
            n_train += 1
        manifest_eval = dest.parent / f"{args.stage_mic}_eval_holdout.yaml"
        manifest_eval.write_text(
            yaml.safe_dump(
                {
                    "train_dir": str(dest),
                    "eval_dir": str(eval_dir),
                    "eval_seconds": eval_seconds,
                    "files": records,
                },
                sort_keys=False,
                allow_unicode=True,
            )
        )
        print(f"eval set: {n_eval} tails → {eval_dir}")
        print(f"wrote {manifest_eval}")
        print(f"staged {n_train} {args.stage_mic} train files → {dest}")


if __name__ == "__main__":
    main()
