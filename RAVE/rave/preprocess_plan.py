"""Planning and PCM concat helpers for RAVE preprocess short-file packing."""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .preprocess_denoise import DenoiseConfig, denoise_pcm


@dataclass(frozen=True)
class AudioProbe:
    path: str
    length_sec: float
    channels: int


def samples_at_sr(length_sec: float, sr: int) -> int:
    if length_sec <= 0:
        return 0
    return int(math.floor(length_sec * sr))


@dataclass
class AudioPack:
    """One or more short files concatenated in decode order."""

    members: List[AudioProbe]
    pad_samples: int = 0

    @property
    def paths(self) -> List[str]:
        return [m.path for m in self.members]

    @property
    def lengths_sec(self) -> List[float]:
        return [m.length_sec for m in self.members]

    @property
    def channels(self) -> int:
        return self.members[0].channels if self.members else 1

    def total_samples(self, sr: int) -> int:
        total = sum(samples_at_sr(m.length_sec, sr) for m in self.members)
        return total + self.pad_samples


@dataclass
class PreprocessPlan:
    long_files: List[AudioProbe]
    packs: List[AudioPack]
    remainder_discarded_samples: int = 0
    short_file_count: int = 0


@dataclass
class PreprocessStats:
    input_files: int = 0
    probe_failures: int = 0
    total_input_sec: float = 0.0
    short_files: int = 0
    concat_packs: int = 0
    files_in_packs: int = 0
    chunks_written: int = 0
    stored_sec: float = 0.0
    discarded_sec: float = 0.0
    tail_discarded_sec: float = 0.0
    remainder_discarded_sec: float = 0.0
    file_lengths_sec: List[float] = field(default_factory=list)


def min_samples_for_mode(num_signal: int, lazy: bool) -> int:
    return num_signal if lazy else 2 * num_signal


def expected_chunks_from_samples(total_samples: int, min_samples: int) -> int:
    if min_samples <= 0:
        return 0
    return total_samples // min_samples


def expected_preprocess_chunks(
    duration_sec: float, sr: int, num_signal: int, lazy: bool = False
) -> int:
    return expected_chunks_from_samples(
        samples_at_sr(duration_sec, sr),
        min_samples_for_mode(num_signal, lazy),
    )


def build_preprocess_plan(
    probes: Sequence[AudioProbe],
    sr: int,
    num_signal: int,
    *,
    lazy: bool,
    concat_short: bool,
    concat_seed: int,
    pad_short_remainder: bool,
) -> PreprocessPlan:
    min_samples = min_samples_for_mode(num_signal, lazy)
    long_files: List[AudioProbe] = []
    short_files: List[AudioProbe] = []

    for probe in probes:
        n = samples_at_sr(probe.length_sec, sr)
        if n >= min_samples:
            long_files.append(probe)
        else:
            short_files.append(probe)

    packs: List[AudioPack] = []
    remainder_discarded = 0

    if concat_short and short_files:
        rng = random.Random(concat_seed)
        shuffled = list(short_files)
        rng.shuffle(shuffled)

        current: List[AudioProbe] = []
        current_samples = 0
        for probe in shuffled:
            current.append(probe)
            current_samples += samples_at_sr(probe.length_sec, sr)
            if current_samples >= min_samples:
                packs.append(AudioPack(members=current))
                current = []
                current_samples = 0

        if current:
            if pad_short_remainder and current_samples > 0:
                pad = min_samples - current_samples
                packs.append(AudioPack(members=current, pad_samples=pad))
            else:
                remainder_discarded = current_samples
    elif short_files and not concat_short:
        remainder_discarded = sum(samples_at_sr(p.length_sec, sr) for p in short_files)

    return PreprocessPlan(
        long_files=long_files,
        packs=packs,
        remainder_discarded_samples=remainder_discarded,
        short_file_count=len(short_files),
    )


def _channel_map(input_channels: int, channels: int) -> List[int]:
    if input_channels >= channels:
        return list(range(channels))
    return (math.ceil(channels / input_channels) * list(range(input_channels)))[:channels]


def decode_path_pcm(
    path: str,
    sr: int,
    channels: int,
    input_channels: int,
    denoise: DenoiseConfig = DenoiseConfig.disabled(),
) -> np.ndarray:
    """Decode file to shape (channels, n_samples) float32 in [-1, 1]."""
    channel_map = _channel_map(input_channels, channels)
    channel_data = []
    for i in channel_map:
        proc = subprocess.Popen(
            [
                'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path,
                '-ar', str(sr),
                '-f', 's16le',
                '-filter_complex', 'channelmap=%d-0' % i,
                '-',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        raw, _ = proc.communicate()
        if proc.returncode != 0:
            return np.zeros((channels, 0), dtype=np.float32)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / (2**15 - 1)
        channel_data.append(pcm)
    if not channel_data:
        return np.zeros((channels, 0), dtype=np.float32)
    n = min(len(c) for c in channel_data)
    pcm = np.stack([c[:n] for c in channel_data], axis=0)
    return denoise_pcm(pcm, sr, denoise)


def concat_pack_pcm(
    pack: AudioPack,
    sr: int,
    channels: int,
    denoise: DenoiseConfig = DenoiseConfig.disabled(),
) -> np.ndarray:
    parts = []
    for member in pack.members:
        pcm = decode_path_pcm(
            member.path, sr, channels, member.channels, denoise=denoise)
        if pcm.shape[-1] > 0:
            parts.append(pcm)
    if not parts:
        total = pack.pad_samples
        return np.zeros((channels, total), dtype=np.float32)
    merged = np.concatenate(parts, axis=-1)
    if pack.pad_samples > 0:
        pad = np.zeros((channels, pack.pad_samples), dtype=np.float32)
        merged = np.concatenate([merged, pad], axis=-1)
    return merged


def pcm_to_chunk_bytes(pcm: np.ndarray, chunk_samples_per_ch: int) -> bytes:
    """Interleave channels into preprocess chunk bytes (n_signal*4 per channel read)."""
    channels, n = pcm.shape
    chunk = pcm[:, :chunk_samples_per_ch]
    if chunk.shape[-1] < chunk_samples_per_ch:
        pad = np.zeros((channels, chunk_samples_per_ch - chunk.shape[-1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=-1)
    int16 = np.floor(chunk * (2**15 - 1)).astype(np.int16)
    # load_audio_chunk joins per-channel reads; each channel contributes chunk_samples*2 bytes
    return b''.join(int16[i].tobytes() for i in range(channels))


def iter_pack_chunks(
    pack: AudioPack,
    n_signal: int,
    sr: int,
    channels: int,
    denoise: DenoiseConfig = DenoiseConfig.disabled(),
) -> Iterable[bytes]:
    """Yield LMDB chunks from a packed group (non-lazy: 2*n_signal samples per chunk)."""
    chunk_samples = 2 * n_signal
    pcm = concat_pack_pcm(pack, sr, channels, denoise=denoise)
    offset = 0
    while offset + chunk_samples <= pcm.shape[-1]:
        yield pcm_to_chunk_bytes(pcm[:, offset:offset + chunk_samples], chunk_samples)
        offset += chunk_samples


def iter_long_file_chunks(
    path: str,
    n_signal: int,
    sr: int,
    channels: int,
    input_channels: int,
    denoise: DenoiseConfig = DenoiseConfig.disabled(),
) -> Iterable[bytes]:
    """Stream chunks from one file (same byte layout as load_audio_chunk)."""
    chunk_samples = 2 * n_signal
    if denoise.enabled:
        pcm = decode_path_pcm(path, sr, channels, input_channels, denoise=denoise)
        offset = 0
        while offset + chunk_samples <= pcm.shape[-1]:
            yield pcm_to_chunk_bytes(
                pcm[:, offset:offset + chunk_samples], chunk_samples)
            offset += chunk_samples
        return

    channel_map = _channel_map(input_channels, channels)
    processes = []
    for i in range(channels):
        proc = subprocess.Popen(
            [
                'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-i', path,
                '-ar', str(sr),
                '-f', 's16le',
                '-filter_complex', 'channelmap=%d-0' % channel_map[i],
                '-',
            ],
            stdout=subprocess.PIPE,
        )
        processes.append(proc)

    read_bytes = n_signal * 4
    chunk = [p.stdout.read(read_bytes) for p in processes]
    while chunk and len(chunk[0]) == read_bytes:
        yield b''.join(chunk)
        chunk = [p.stdout.read(read_bytes) for p in processes]
    for p in processes:
        p.stdout.close()


def count_plan_chunks(plan: PreprocessPlan, sr: int, num_signal: int, lazy: bool) -> int:
    min_samples = min_samples_for_mode(num_signal, lazy)
    total = 0
    for probe in plan.long_files:
        total += expected_chunks_from_samples(samples_at_sr(probe.length_sec, sr), min_samples)
    for pack in plan.packs:
        total += expected_chunks_from_samples(pack.total_samples(sr), min_samples)
    return total


def compute_discarded_seconds(
    probes: Sequence[AudioProbe],
    plan: PreprocessPlan,
    sr: int,
    num_signal: int,
    lazy: bool,
    concat_short: bool,
) -> Tuple[float, float]:
    """Return (tail_discarded_sec, remainder_discarded_sec)."""
    min_samples = min_samples_for_mode(num_signal, lazy)
    tail_samples = 0
    for probe in plan.long_files:
        n = samples_at_sr(probe.length_sec, sr)
        tail_samples += n % min_samples

    remainder_samples = plan.remainder_discarded_samples
    if not concat_short:
        remainder_samples = sum(
            samples_at_sr(p.length_sec, sr)
            for p in probes
            if samples_at_sr(p.length_sec, sr) < min_samples
        )

    return tail_samples / sr, remainder_samples / sr


def print_preprocess_summary(
    stats: PreprocessStats,
    sr: int,
    num_signal: int,
    lazy: bool,
    denoise: DenoiseConfig = DenoiseConfig.disabled(),
) -> None:
    chunk_sec = min_samples_for_mode(num_signal, lazy) / sr
    lengths = sorted(stats.file_lengths_sec)
    n = len(lengths)

    def pct(p: float) -> float:
        if n == 0:
            return float('nan')
        if n == 1:
            return lengths[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return lengths[int(k)]
        return lengths[f] + (lengths[c] - lengths[f]) * (k - f)

    util = (
        100.0 * stats.stored_sec / (stats.total_input_sec - stats.discarded_sec)
        if stats.total_input_sec > stats.discarded_sec
        else 0.0
    )

    print()
    print('=== preprocess summary ===')
    print(f'  input files (ffprobe ok):     {stats.input_files}')
    print(f'  ffprobe failures:             {stats.probe_failures}')
    print(
        f'  total input duration:         {stats.total_input_sec:.2f} s '
        f'({stats.total_input_sec / 60:.2f} min, {stats.total_input_sec / 3600:.4f} h)'
    )
    print(f'  short files (< min chunk):    {stats.short_files}')
    print(f'  concat packs created:         {stats.concat_packs}')
    print(f'  files placed in packs:        {stats.files_in_packs}')
    print(f'  LMDB chunks written:          {stats.chunks_written}')
    print(
        f'  stored audio duration:        {stats.stored_sec:.2f} s '
        f'({stats.stored_sec / 60:.2f} min, {stats.stored_sec / 3600:.4f} h)'
    )
    print(
        f'  discarded duration:           {stats.discarded_sec:.2f} s '
        f'({stats.discarded_sec / 60:.2f} min, {stats.discarded_sec / 3600:.4f} h)'
    )
    print(f'    tail waste (long files):    {stats.tail_discarded_sec:.2f} s')
    print(f'    short remainders:           {stats.remainder_discarded_sec:.2f} s')
    print(f'  chunk size:                   {chunk_sec:.6f} s ({min_samples_for_mode(num_signal, lazy)} samples @ {sr} Hz)')
    if denoise.enabled:
        print(
            f'  denoise:                      on '
            f'(strength={denoise.strength:.2f}, '
            f'noise_sec={denoise.noise_sec:.2f})'
        )
    if n:
        print('  file length distribution (s):')
        print(f'    min / max:                  {lengths[0]:.4f} / {lengths[-1]:.4f}')
        print(f'    median:                     {lengths[n // 2]:.4f}')
        print(f'    p10 / p90:                  {pct(10):.4f} / {pct(90):.4f}')
    print(f'  chunk utilization:            {util:.1f}%')
    print('=== end preprocess summary ===')
    print()
