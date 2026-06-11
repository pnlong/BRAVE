"""
Build lmdb_index_manifest.yaml matching vendored RAVE preprocess LMDB order.

Usage (BRAVE root):
  export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
  python RAVE/scripts/build_lmdb_index_manifest.py \\
    --input_path /path/to/audio_subset \\
    --db_path /path/to/preprocessed_lmdb \\
    --num_signal 131072 --lazy
"""

from __future__ import annotations

import multiprocessing
import os
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import yaml
from absl import app, flags
from tqdm import tqdm

from rave.preprocess_plan import (
    AudioPack,
    AudioProbe,
    build_preprocess_plan,
    count_plan_chunks,
    expected_chunks_from_samples,
    min_samples_for_mode,
    samples_at_sr,
)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("input_path", None, "Audio roots (same as preprocess)", required=True)
flags.DEFINE_string("db_path", None, "LMDB dir; manifest written here", required=True)
flags.DEFINE_list("ext", ["wav", "mp3", "aif", "aiff", "flac"], "Extensions")
flags.DEFINE_integer("num_signal", 131072, "Chunk size (preprocess --num_signal)")
flags.DEFINE_integer("sampling_rate", 44100, "Sample rate")
flags.DEFINE_bool("lazy", False, "Lazy LMDB mode (one row per work item)")
flags.DEFINE_bool("concat_short", True, "Pack short files (preprocess default)")
flags.DEFINE_integer("concat_seed", 0, "Short-file pack shuffle seed")
flags.DEFINE_bool("pad_short_remainder", True, "Pad last pack remainder")
flags.DEFINE_integer(
    "workers",
    0,
    "ffprobe worker processes (0=all logical CPU cores)",
)
flags.DEFINE_bool("no_progress", False, "Disable progress bars")


def _flatten(iterator: Iterable):
    for item in iterator:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from _flatten(item)
        else:
            yield item


def search_for_audios(path_list: Sequence[str], extensions: Sequence[str]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
            audios.append(p.rglob(f"*.{ext.upper()}"))
    return list(_flatten(audios))


def get_audio_length(path: str):
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-i", path, "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            ],
            stderr=subprocess.DEVNULL,
        )
        length = float(out.decode().strip())
        ch_out = subprocess.check_output(
            [
                "ffprobe", "-i", path, "-v", "error", "-show_entries",
                "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1",
            ],
            stderr=subprocess.DEVNULL,
        )
        channels = int(ch_out.decode().strip().split("\n")[0])
        return path, length, channels
    except (subprocess.CalledProcessError, ValueError):
        return None


def clip_id_from_path(path: str) -> str:
    return Path(path).stem


def build_work_items(plan, concat_short: bool):
    items: List[Tuple[str, Union[AudioProbe, AudioPack]]] = [
        ("long", p) for p in plan.long_files
    ]
    items.extend(("pack", pack) for pack in plan.packs)
    return items


def manifest_entries(
    plan,
    work_items,
    sr: int,
    num_signal: int,
    lazy: bool,
    *,
    show_progress: bool = True,
) -> list:
    entries = []
    min_samples = min_samples_for_mode(num_signal, lazy)
    idx = 0
    item_iter = (
        tqdm(work_items, desc="manifest", unit="item")
        if show_progress else work_items
    )

    if lazy:
        for kind, data in item_iter:
            if kind == "long":
                probe: AudioProbe = data
                entries.append({
                    "lmdb_index": idx,
                    "kind": "long",
                    "source_path": probe.path,
                    "clip_id": clip_id_from_path(probe.path),
                    "clip_ids": [clip_id_from_path(probe.path)],
                    "chunk_in_source": 0,
                })
                idx += 1
            else:
                pack: AudioPack = data
                ids = [clip_id_from_path(m.path) for m in pack.members]
                entries.append({
                    "lmdb_index": idx,
                    "kind": "pack",
                    "source_path": pack.members[0].path if pack.members else "",
                    "clip_id": ids[0] if ids else "",
                    "clip_ids": ids,
                    "chunk_in_source": 0,
                })
                idx += 1
        return entries

    for kind, data in item_iter:
        if kind == "long":
            probe = data
            n_samples = samples_at_sr(probe.length_sec, sr)
            n_chunks = expected_chunks_from_samples(n_samples, min_samples)
            cid = clip_id_from_path(probe.path)
            for chunk_i in range(n_chunks):
                entries.append({
                    "lmdb_index": idx,
                    "kind": "long",
                    "source_path": probe.path,
                    "clip_id": cid,
                    "clip_ids": [cid],
                    "chunk_in_source": chunk_i,
                })
                idx += 1
        else:
            pack = data
            ids = [clip_id_from_path(m.path) for m in pack.members]
            n_samples = pack.total_samples(sr)
            n_chunks = expected_chunks_from_samples(n_samples, min_samples)
            primary = ids[0] if ids else ""
            for chunk_i in range(n_chunks):
                entries.append({
                    "lmdb_index": idx,
                    "kind": "pack",
                    "source_path": pack.members[0].path if pack.members else "",
                    "clip_id": primary,
                    "clip_ids": ids,
                    "chunk_in_source": chunk_i,
                })
                idx += 1
    return entries


def main(argv):
    del argv
    show_progress = not FLAGS.no_progress
    audios = search_for_audios(FLAGS.input_path, FLAGS.ext)
    audios = [os.path.abspath(str(p)) for p in audios]
    if not audios:
        print("No audio files found.")
        return

    pool = multiprocessing.Pool(
        processes=max(1, FLAGS.workers) if FLAGS.workers > 0 else None)
    try:
        probe_iter = pool.imap(get_audio_length, audios)
        if show_progress:
            probe_iter = tqdm(
                probe_iter,
                total=len(audios),
                desc="ffprobe",
                unit="file",
            )
        probe_results = list(probe_iter)
    finally:
        pool.close()
        pool.join()

    probe_failures = sum(1 for r in probe_results if r is None)
    probes = []
    for result in probe_results:
        if result is None:
            continue
        p, length, channels = result
        probes.append(AudioProbe(path=p, length_sec=length, channels=channels))

    if probe_failures:
        print(f"ffprobe failed on {probe_failures} file(s).")

    plan = build_preprocess_plan(
        probes,
        FLAGS.sampling_rate,
        FLAGS.num_signal,
        lazy=FLAGS.lazy,
        concat_short=FLAGS.concat_short,
        concat_seed=FLAGS.concat_seed,
        pad_short_remainder=FLAGS.pad_short_remainder,
    )
    work_items = build_work_items(plan, FLAGS.concat_short)
    entries = manifest_entries(
        plan,
        work_items,
        FLAGS.sampling_rate,
        FLAGS.num_signal,
        FLAGS.lazy,
        show_progress=show_progress,
    )

    expected = (
        len(work_items) if FLAGS.lazy else count_plan_chunks(
            plan, FLAGS.sampling_rate, FLAGS.num_signal, FLAGS.lazy)
    )

    manifest = {
        "version": 1,
        "lazy": FLAGS.lazy,
        "sr": FLAGS.sampling_rate,
        "num_signal": FLAGS.num_signal,
        "concat_short": FLAGS.concat_short,
        "concat_seed": FLAGS.concat_seed,
        "expected_lmdb_rows": expected,
        "entries": entries,
    }

    out_path = os.path.join(FLAGS.db_path, "lmdb_index_manifest.yaml")
    os.makedirs(FLAGS.db_path, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Wrote {out_path} ({len(entries)} entries, expected rows {expected})")


if __name__ == "__main__":
    app.run(main)
