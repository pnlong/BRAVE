import functools
import multiprocessing
import os
import pathlib
import random
import sys

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)
import subprocess
from datetime import timedelta
from functools import partial
from itertools import repeat
from typing import Callable, Iterable, List, Sequence, Tuple, Union

import lmdb
import numpy as np
import torch
import yaml
import math
from absl import app, flags
from tqdm import tqdm
from udls.generated import AudioExample

from rave.preprocess_denoise import DenoiseConfig
from rave.preprocess_normalize import NormalizeConfig
from rave.preprocess_pcen import PcenConfig
from rave.preprocess_plan import (
    AudioPack,
    AudioProbe,
    PreprocessStats,
    build_preprocess_plan,
    compute_discarded_seconds,
    count_plan_chunks,
    iter_long_file_chunks,
    iter_pack_chunks,
    min_samples_for_mode,
    print_preprocess_summary,
)

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('input_path',
                          None,
                          help='Path to a directory containing audio files',
                          required=True)
flags.DEFINE_string('output_path',
                    None,
                    help='Output directory for the dataset',
                    required=True)
flags.DEFINE_integer('num_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 1, help="Number of audio channels")
flags.DEFINE_integer('sampling_rate',
                     44100,
                     help='Sampling rate to use during training')
flags.DEFINE_integer('max_db_size',
                     100,
                     help='Maximum size (in GB) of the dataset')
flags.DEFINE_multi_string(
    'ext',
    default=['aif', 'aiff', 'wav', 'opus', 'mp3', 'aac', 'flac', 'ogg'],
    help='Extension to search for in the input directory')
flags.DEFINE_bool('lazy',
                  default=False,
                  help='Decode and resample audio samples.')
flags.DEFINE_bool('dyndb',
                  default=True,
                  help="Allow the database to grow dynamically")
flags.DEFINE_integer('workers',
                     default=0,
                     help='Preprocessor worker processes (0=all logical CPU cores)')
flags.DEFINE_bool('concat_short',
                  default=True,
                  help='Concatenate short files until one full preprocess chunk')
flags.DEFINE_integer('concat_seed',
                     default=42,
                     help='RNG seed for shuffling short files before packing')
flags.DEFINE_bool('pad_short_remainder',
                  default=False,
                  help='Zero-pad the final undersized short-file pack to one chunk')
flags.DEFINE_bool('no_progress', False, 'Disable progress bars')
flags.DEFINE_bool(
    'denoise',
    default=False,
    help='Apply stationary spectral noise reduction before writing LMDB chunks',
)
flags.DEFINE_float(
    'denoise_strength',
    default=0.75,
    help='Noise gate strength in [0, 1] when --denoise is set',
)
flags.DEFINE_float(
    'denoise_noise_sec',
    default=0.0,
    help='Estimate noise floor from the first N seconds only (0 = whole clip)',
)
flags.DEFINE_bool(
    'normalize',
    default=False,
    help='Peak-normalize each LMDB chunk (same rule as train.py --normalize)',
)
flags.DEFINE_float(
    'normalize_max_gain_db',
    default=30.0,
    help='Max gain in dB when --normalize is set',
)
flags.DEFINE_bool(
    'pcen',
    default=False,
    help='Apply PCEN (per-channel energy normalization) before writing LMDB chunks',
)
flags.DEFINE_integer(
    'pcen_n_mels',
    default=128,
    help='Mel bands for --pcen',
)
flags.DEFINE_float(
    'pcen_gain',
    default=0.98,
    help='PCEN AGC strength (librosa pcen gain)',
)
flags.DEFINE_float(
    'pcen_bias',
    default=2.0,
    help='PCEN bias floor',
)
flags.DEFINE_float(
    'pcen_power',
    default=0.5,
    help='PCEN compression exponent',
)
flags.DEFINE_float(
    'pcen_time_constant',
    default=0.4,
    help='PCEN smoothing time constant (seconds)',
)
flags.DEFINE_float(
    'pcen_max_gain',
    default=10.0,
    help='Cap linear STFT gain from PCEN ratio (0 = no cap)',
)
flags.DEFINE_float(
    'max_stored_sec',
    default=0.0,
    help='Stop after storing this many seconds of audio (0 = no limit)',
)
flags.DEFINE_integer(
    'subset_seed',
    default=None,
    help='Shuffle input files before planning (recommended with a stored-sec cap)',
)


def build_denoise_config() -> DenoiseConfig:
    return DenoiseConfig(
        enabled=FLAGS.denoise,
        strength=FLAGS.denoise_strength,
        noise_sec=FLAGS.denoise_noise_sec,
    )


def build_normalize_config() -> NormalizeConfig:
    return NormalizeConfig(
        enabled=FLAGS.normalize,
        max_gain_db=FLAGS.normalize_max_gain_db,
    )


def build_pcen_config() -> PcenConfig:
    return PcenConfig(
        enabled=FLAGS.pcen,
        n_mels=FLAGS.pcen_n_mels,
        gain=FLAGS.pcen_gain,
        bias=FLAGS.pcen_bias,
        power=FLAGS.pcen_power,
        time_constant=FLAGS.pcen_time_constant,
        max_gain=FLAGS.pcen_max_gain,
    )


def load_audio_chunk(path: str, n_signal: int,
                     sr: int, channels: int = 1) -> Iterable[bytes]:
    _, input_channels = get_audio_channels(path)
    if input_channels is None:
        return
    yield from iter_long_file_chunks(path, n_signal, sr, channels, input_channels)


def expected_preprocess_chunks(duration_sec: float, sr: int, num_signal: int) -> int:
    """Full LMDB rows from one file (non-lazy)."""
    if duration_sec <= 0:
        return 0
    samples = int(math.floor(duration_sec * sr))
    return samples // (2 * num_signal)


def get_audio_length(path: str):
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'format=duration'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode:
        return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        length = float(stdout)
        _, channels = get_audio_channels(path)
        return path, float(length), int(channels)
    except Exception:
        return None


def get_audio_channels(path: str):
    process = subprocess.Popen(
        [
            'ffprobe', '-i', path, '-v', 'error', '-show_entries',
            'stream=channels'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = process.communicate()
    if process.returncode:
        return None
    try:
        stdout = stdout.decode().split('\n')[1].split('=')[-1]
        channels = int(stdout)
        return path, int(channels)
    except Exception:
        return None


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def get_metadata(audio_samples, channels: int = 1):
    audio = np.frombuffer(audio_samples, dtype=np.int16)
    audio = audio.astype(float) / (2**15 - 1)
    audio = audio.reshape(channels, -1)
    peak_amplitude = np.amax(np.abs(audio))
    rms_amplitude = np.sqrt(np.mean(audio**2))
    return {
        'peak': str(peak_amplitude),
        'rms': str(rms_amplitude),
    }


def process_audio_array(audio: Tuple[int, bytes],
                        env: lmdb.Environment,
                        channels: int = 1) -> int:
    audio_id, audio_samples = audio
    buffers = {}
    buffers['waveform'] = AudioExample.AudioBuffer(
        shape=(channels, int(len(audio_samples) / (2 * channels))),
        sampling_rate=FLAGS.sampling_rate,
        data=audio_samples,
        precision=AudioExample.Precision.INT16,
    )

    meta = get_metadata(audio_samples, channels)
    ae = AudioExample(buffers=buffers, metadata=meta)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(
            key.encode(),
            ae.SerializeToString(),
        )
    return audio_id


WorkItem = Tuple[str, Union[AudioProbe, AudioPack]]


def load_work_chunks(
    item: WorkItem,
    n_signal: int,
    sr: int,
    channels: int,
    denoise: DenoiseConfig,
    pcen: PcenConfig,
    normalize: NormalizeConfig,
) -> Iterable[bytes]:
    kind, data = item
    if kind == 'long':
        probe: AudioProbe = data
        yield from iter_long_file_chunks(
            probe.path,
            n_signal,
            sr,
            channels,
            probe.channels,
            denoise=denoise,
            pcen=pcen,
            normalize=normalize,
        )
    elif kind == 'pack':
        pack: AudioPack = data
        yield from iter_pack_chunks(
            pack, n_signal, sr, channels,
            denoise=denoise, pcen=pcen, normalize=normalize)


def process_lazy_entry(entry: Tuple[int, WorkItem], env: lmdb.Environment) -> float:
    audio_id, (kind, data) = entry
    if kind == 'long':
        probe: AudioProbe = data
        meta = {
            'path': probe.path,
            'length': str(probe.length_sec),
            'channels': str(probe.channels),
        }
        length = probe.length_sec
    else:
        pack: AudioPack = data
        combined = sum(m.length_sec for m in pack.members)
        if pack.pad_samples:
            combined += pack.pad_samples / FLAGS.sampling_rate
        meta = {
            'paths': ','.join(p.path for p in pack.members),
            'lengths': ','.join(str(p.length_sec) for p in pack.members),
            'length': str(combined),
            'channels': str(pack.channels),
            'packed': 'true',
        }
        length = combined
    ae = AudioExample(metadata=meta)
    key = f'{audio_id:08d}'
    with env.begin(write=True) as txn:
        txn.put(key.encode(), ae.SerializeToString())
    return length


def flatmap(pool: multiprocessing.Pool,
            func: Callable,
            iterable: Iterable,
            chunksize=None):
    queue = multiprocessing.Manager().Queue(maxsize=os.cpu_count())
    pool.map_async(
        functools.partial(flat_mappper, func),
        zip(iterable, repeat(queue)),
        chunksize,
        lambda _: queue.put(None),
        lambda *e: print(e),
    )

    item = queue.get()
    while item is not None:
        yield item
        item = queue.get()


def flat_mappper(func, arg):
    data, queue = arg
    for item in func(data):
        queue.put(item)


def search_for_audios(path_list: Sequence[str], extensions: Sequence[str]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
            audios.append(p.rglob(f'*.{ext.upper()}'))
    audios = flatten(audios)
    return audios


def build_work_items(plan, concat_short: bool) -> List[WorkItem]:
    items: List[WorkItem] = [('long', p) for p in plan.long_files]
    items.extend(('pack', pack) for pack in plan.packs)
    return items


def main(argv):
    if FLAGS.lazy and os.name in ["nt", "posix"]:
        while (answer := input(
                "Using lazy datasets on Windows/macOS might result in slow training. Continue ? (y/n) "
        ).lower()) not in ["y", "n"]:
            print("Answer 'y' or 'n'.")
        if answer == "n":
            print("Aborting...")
            exit()

    output_dir = os.path.join(*os.path.split(FLAGS.output_path)[:-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.max_db_size * 1024**3,
        map_async=not FLAGS.dyndb,
        writemap=not FLAGS.dyndb,
    )
    pool = multiprocessing.Pool(
        processes=max(1, FLAGS.workers) if FLAGS.workers > 0 else None)

    audios = search_for_audios(FLAGS.input_path, FLAGS.ext)
    audios = [os.path.abspath(str(p)) for p in audios]
    if len(audios) == 0:
        print("No valid file found in %s. Aborting" % FLAGS.input_path)
        pool.close()
        env.close()
        return

    min_samples = min_samples_for_mode(FLAGS.num_signal, FLAGS.lazy)
    chunk_seconds = min_samples / FLAGS.sampling_rate

    probe_iter = pool.imap(get_audio_length, audios)
    if not FLAGS.no_progress:
        probe_iter = tqdm(
            probe_iter,
            total=len(audios),
            desc="ffprobe",
            unit="file",
        )
    probe_results = list(probe_iter)
    probe_failures = sum(1 for r in probe_results if r is None)
    probes = []
    file_lengths = []
    for result in probe_results:
        if result is None:
            continue
        path, length, channels = result
        probes.append(AudioProbe(path=path, length_sec=length, channels=channels))
        file_lengths.append(length)

    max_stored_sec = FLAGS.max_stored_sec

    if FLAGS.subset_seed is not None:
        rng = random.Random(FLAGS.subset_seed)
        rng.shuffle(probes)

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
    total_chunks = count_plan_chunks(
        plan, FLAGS.sampling_rate, FLAGS.num_signal, FLAGS.lazy)

    tail_sec, remainder_sec = compute_discarded_seconds(
        probes,
        plan,
        FLAGS.sampling_rate,
        FLAGS.num_signal,
        FLAGS.lazy,
        FLAGS.concat_short,
    )

    stats = PreprocessStats(
        input_files=len(probes),
        probe_failures=probe_failures,
        total_input_sec=sum(file_lengths),
        short_files=plan.short_file_count,
        concat_packs=len(plan.packs),
        files_in_packs=sum(len(p.members) for p in plan.packs),
        tail_discarded_sec=tail_sec,
        remainder_discarded_sec=remainder_sec,
        discarded_sec=tail_sec + remainder_sec,
        file_lengths_sec=file_lengths,
    )

    if probe_failures:
        print(f"ffprobe failed on {probe_failures} file(s).")

    n_seconds = 0.0
    denoise = build_denoise_config()
    pcen = build_pcen_config()
    normalize = build_normalize_config()
    chunk_load = partial(
        load_work_chunks,
        n_signal=FLAGS.num_signal,
        sr=FLAGS.sampling_rate,
        channels=FLAGS.channels,
        denoise=denoise,
        pcen=pcen,
        normalize=normalize,
    )

    if not FLAGS.lazy:
        chunks = flatmap(pool, chunk_load, work_items)
        chunks = enumerate(chunks)

        processed_samples = map(
            partial(process_audio_array, env=env, channels=FLAGS.channels),
            chunks,
        )

        sample_iter = processed_samples
        if FLAGS.no_progress:
            last_id = -1
            stopped_early = False
            for audio_id in sample_iter:
                if (max_stored_sec > 0
                        and (audio_id + 1) * chunk_seconds > max_stored_sec):
                    stopped_early = True
                    break
                last_id = audio_id
                n_seconds = chunk_seconds * (audio_id + 1)
                if total_chunks and audio_id + 1 > total_chunks:
                    pass
            stats.chunks_written = last_id + 1 if last_id >= 0 else 0
            stats.stored_sec = stats.chunks_written * chunk_seconds
            if stopped_early:
                print(
                    f'stored-sec cap reached ({max_stored_sec:.2f} s); '
                    f'wrote {stats.chunks_written} chunks '
                    f'({stats.stored_sec:.2f} s)')
        else:
            pbar = tqdm(
                sample_iter,
                total=total_chunks or None,
                desc="preprocess",
                unit="chunk",
            )
            last_id = -1
            stopped_early = False
            for audio_id in pbar:
                if (max_stored_sec > 0
                        and (audio_id + 1) * chunk_seconds > max_stored_sec):
                    stopped_early = True
                    break
                last_id = audio_id
                n_seconds = chunk_seconds * (audio_id + 1)
                pbar.set_description(
                    f'preprocess ({timedelta(seconds=n_seconds)} audio)')
                if total_chunks and audio_id + 1 > total_chunks:
                    pbar.total = audio_id + 1
                    pbar.refresh()
            pbar.close()
            stats.chunks_written = last_id + 1 if last_id >= 0 else 0
            stats.stored_sec = stats.chunks_written * chunk_seconds
            if stopped_early:
                print(
                    f'stored-sec cap reached ({max_stored_sec:.2f} s); '
                    f'wrote {stats.chunks_written} chunks '
                    f'({stats.stored_sec:.2f} s)')
    else:
        lazy_entries = enumerate(work_items)
        processed = map(partial(process_lazy_entry, env=env), lazy_entries)
        entry_iter = processed
        n_seconds = 0.0
        entries_written = 0
        stopped_early = False
        if FLAGS.no_progress:
            for length in entry_iter:
                if (max_stored_sec > 0
                        and n_seconds + length > max_stored_sec):
                    stopped_early = True
                    break
                n_seconds += length
                entries_written += 1
        else:
            pbar = tqdm(
                entry_iter,
                total=len(work_items),
                desc="preprocess",
                unit="entry",
            )
            for length in pbar:
                if (max_stored_sec > 0
                        and n_seconds + length > max_stored_sec):
                    stopped_early = True
                    break
                n_seconds += length
                entries_written += 1
                pbar.set_description(
                    f'preprocess ({timedelta(seconds=n_seconds)} audio)')
            pbar.close()
        stats.chunks_written = entries_written
        stats.stored_sec = n_seconds
        if stopped_early:
            print(
                f'stored-sec cap reached ({max_stored_sec:.2f} s); '
                f'wrote {entries_written} lazy entries ({n_seconds:.2f} s)')

    print_preprocess_summary(
        stats,
        FLAGS.sampling_rate,
        FLAGS.num_signal,
        FLAGS.lazy,
        denoise=denoise,
        pcen=pcen,
        normalize=normalize,
    )

    meta = {
        'lazy': FLAGS.lazy,
        'channels': FLAGS.channels,
        'n_seconds': n_seconds,
        'sr': FLAGS.sampling_rate,
        'denoise': FLAGS.denoise,
        'denoise_strength': FLAGS.denoise_strength,
        'denoise_noise_sec': FLAGS.denoise_noise_sec,
        'normalize': FLAGS.normalize,
        'normalize_max_gain_db': FLAGS.normalize_max_gain_db,
    }
    if max_stored_sec > 0:
        meta['max_stored_sec'] = max_stored_sec
    if FLAGS.subset_seed is not None:
        meta['subset_seed'] = FLAGS.subset_seed
    with open(os.path.join(FLAGS.output_path, 'metadata.yaml'), 'w') as metadata:
        yaml.safe_dump(meta, metadata)
    pool.close()
    env.close()


if __name__ == '__main__':
    app.run(main)
