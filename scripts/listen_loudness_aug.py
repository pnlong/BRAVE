#!/usr/bin/env python3
"""Write raw crops and RandomTargetLoudness draws for listening.

Probes absolute RandomTargetLoudness (default −60…−3 dBFS, the range that
quieted Foundation Y). Training itself uses relative RandomGain in brave.gin.
Phase mangle is left off so the files differ only by the loudness draw.

    python scripts/listen_loudness_aug.py
    python scripts/listen_loudness_aug.py --output-dir /tmp/loudness_aug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_BRAVE = Path(__file__).resolve().parents[1]
_RAVE = _BRAVE / "RAVE"
if str(_RAVE) not in sys.path:
    sys.path.insert(0, str(_RAVE))

from rave.dataset import get_dataset, rms_dbfs  # noqa: E402
from rave.transforms import RandomTargetLoudness  # noqa: E402

THRESHOLDS_DB = (-40.0, -30.0, -20.0)


def _parse_args() -> argparse.Namespace:
    storage = Path("/data/hai-res/p1long/BRAVE-data")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=storage / "listening" / "loudness_aug",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=["rain_sounds", "fire_crackling"],
        help="Playlist names under yt_playlists/<name>/preprocessed",
    )
    p.add_argument("--storage", type=Path, default=storage)
    p.add_argument("--n-crops", type=int, default=4)
    p.add_argument("--n-draws", type=int, default=8, help="WAV draws saved per crop")
    p.add_argument(
        "--n-stat-draws",
        type=int,
        default=200,
        help="Extra in-memory draws per crop for the level counts",
    )
    p.add_argument("--n-signal", type=int, default=131072)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loudness-min", type=float, default=-60.0)
    p.add_argument("--loudness-max", type=float, default=-3.0)
    return p.parse_args()


def _spread_indices(n_items: int, n_crops: int) -> list[int]:
    if n_items < 1:
        raise ValueError("dataset is empty")
    n_crops = min(n_crops, n_items)
    if n_crops == 1:
        return [0]
    return [int(round(i * (n_items - 1) / (n_crops - 1))) for i in range(n_crops)]


def _as_mono_float(audio) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    return np.ascontiguousarray(x)


def _clipped_fraction(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) >= 0.999))


def _threshold_counts(levels_db: np.ndarray) -> dict[str, float]:
    out = {"n": int(levels_db.size), "median_db": float(np.median(levels_db))}
    for threshold in THRESHOLDS_DB:
        key = f"frac_below_{int(abs(threshold))}"
        out[key] = float(np.mean(levels_db < threshold))
    return out


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def _level_tag(rms_db: float) -> str:
    return f"rms_{rms_db:+.1f}dB".replace("+", "p").replace("-", "m")


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    aug = RandomTargetLoudness(
        loudness_range=(args.loudness_min, args.loudness_max),
        prob=1.0,
        hard_clip=True,
        max_gain_db=60.0,
    )
    summary = {
        "loudness_range_db": [args.loudness_min, args.loudness_max],
        "hard_clip": True,
        "max_gain_db": 60.0,
        "n_signal": args.n_signal,
        "sr": args.sr,
        "domains": {},
    }

    for name in args.domains:
        db_path = args.storage / "yt_playlists" / name / "preprocessed"
        if not db_path.is_dir():
            raise FileNotFoundError(db_path)
        dataset = get_dataset(
            str(db_path),
            args.sr,
            args.n_signal,
            augmentations=[],
            stochastic=False,
            show_progress=False,
        )
        indices = _spread_indices(len(dataset), args.n_crops)
        domain_dir = args.output_dir / name
        all_draw_db = []
        all_clip = []
        crops = []
        for crop_i, index in enumerate(indices):
            raw = _as_mono_float(dataset[index])
            raw_db = rms_dbfs(raw)
            crop_dir = domain_dir / f"crop{crop_i:02d}_idx{index}"
            _write_wav(crop_dir / f"raw_{_level_tag(raw_db)}.wav", raw, args.sr)
            saved = []
            for draw_i in range(args.n_draws):
                drawn = _as_mono_float(aug(raw))
                drawn_db = rms_dbfs(drawn)
                clip = _clipped_fraction(drawn)
                _write_wav(
                    crop_dir / f"draw{draw_i:02d}_{_level_tag(drawn_db)}.wav",
                    drawn,
                    args.sr,
                )
                saved.append({"rms_db": drawn_db, "clipped_frac": clip})
                all_draw_db.append(drawn_db)
                all_clip.append(clip)
            for _ in range(args.n_stat_draws):
                drawn = _as_mono_float(aug(raw))
                all_draw_db.append(rms_dbfs(drawn))
                all_clip.append(_clipped_fraction(drawn))
            crops.append({
                "index": index,
                "raw_rms_db": raw_db,
                "saved_draws": saved,
            })
        levels = np.asarray(all_draw_db, dtype=np.float64)
        summary["domains"][name] = {
            "db_path": str(db_path),
            "n_lmdb_rows": len(dataset),
            "crops": crops,
            "draws": _threshold_counts(levels),
            "mean_clipped_frac": float(np.mean(all_clip)),
        }
        counts = summary["domains"][name]["draws"]
        print(
            f"{name}: median {counts['median_db']:.1f} dB, "
            f"below -40/-30/-20: "
            f"{counts['frac_below_40']:.0%} / "
            f"{counts['frac_below_30']:.0%} / "
            f"{counts['frac_below_20']:.0%} "
            f"(n={counts['n']})"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
