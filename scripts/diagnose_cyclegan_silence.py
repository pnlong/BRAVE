#!/usr/bin/env python3
"""Offline CycleGAN diagnosis: does silence decode as ringing?

Does **not** peak-normalize (dBFS must stay meaningful). Full-context Python
transfer is the primary path; 512-block transfer is a pad/cache control.

    export PYTHONPATH="${PWD}/RAVE:${PWD}/latent_exploration:${PYTHONPATH}"
    python scripts/diagnose_cyclegan_silence.py \\
      --tap-backbone path/to/tap.ckpt \\
      --y-backbone path/to/y.ckpt \\
      --cyclegan runs/.../cyclegan_latent.ckpt \\
      --output-dir /tmp/silence_diag [--input tap.wav] [--y-db-path lmdb]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_BRAVE = Path(__file__).resolve().parents[1]
_LATENT = _BRAVE / "latent_exploration"
_RAVE = _BRAVE / "RAVE"
for p in (_LATENT, _RAVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from load_model import load_audio, load_model, save_audio  # noqa: E402
from rave.canonicalizer.cycle_inference import (  # noqa: E402
    _encode_mean,
    load_cyclegan_warps_from_checkpoint,
    transfer_waveform_blocked,
    transfer_x_to_y,
)
from rave.canonicalizer.silence_gate import (  # noqa: E402
    as_btc,
    frame_rms_dbfs,
    global_rms_dbfs,
    latent_hop,
    mute_by_input_rms,
    replace_latent_on_quiet,
    sidechain_by_input_loudness,
)
from rave.dataset import AudioDataset, LazyAudioDataset  # noqa: E402
from rave import transforms  # noqa: E402
import yaml  # noqa: E402

SILENT_FRAME_DB = -40.0
LEAK_DB = -50.0
GATE_QUIET_DB = -60.0
LOUD_MARGIN_DB = 6.0
DEFAULT_RF_SEC = 0.6
DEFAULT_PROBE_SEC = 3.0
BURST_SEC = 0.05
GAP_SEC = 1.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tap-backbone", type=Path, required=True)
    p.add_argument("--y-backbone", type=Path, required=True)
    p.add_argument("--cyclegan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None, help="Optional real tap WAV")
    p.add_argument("--y-db-path", type=Path, default=None, help="Optional Y LMDB for quiet-Y z")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--threshold-db", type=float, default=SILENT_FRAME_DB)
    p.add_argument("--hold-frames", type=int, default=2)
    p.add_argument(
        "--sidechain-smooth-frames",
        type=int,
        default=8,
        help="Causal hop smoothing for gated_sidechain (Gate C)",
    )
    p.add_argument("--rf-drop-sec", type=float, default=DEFAULT_RF_SEC)
    p.add_argument("--probe-sec", type=float, default=DEFAULT_PROBE_SEC)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--left-context", type=int, default=32768)
    return p.parse_args()


def decode_from_z(backbone, z: torch.Tensor, target_len: int) -> torch.Tensor:
    return backbone.decode(z)[..., :target_len]


def noise_at_dbfs(
    shape: Tuple[int, ...],
    dbfs: float,
    *,
    device,
    dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    rms = x.pow(2).mean().sqrt().clamp_min(1e-12)
    target = 10.0 ** (dbfs / 20.0)
    return x * (target / rms)


def synthetic_burst_gap(
    n_channels: int,
    sr: int,
    *,
    rf_samples: int,
    device,
    dtype,
    burst_db: float = -6.0,
) -> torch.Tensor:
    burst_n = max(1, int(round(BURST_SEC * sr)))
    gap_n = max(1, int(round(GAP_SEC * sr)))
    burst = noise_at_dbfs((n_channels, burst_n), burst_db, device=device, dtype=dtype)
    gap = torch.zeros(n_channels, gap_n, device=device, dtype=dtype)
    prefix = torch.zeros(n_channels, rf_samples, device=device, dtype=dtype)
    return torch.cat([prefix, burst, gap, burst], dim=-1)


def pad_rf_prefix(x: torch.Tensor, rf_samples: int) -> torch.Tensor:
    if rf_samples <= 0:
        return x
    zeros = torch.zeros(*x.shape[:-1], rf_samples, device=x.device, dtype=x.dtype)
    return torch.cat([zeros, x], dim=-1)


def mean_l2(z: torch.Tensor) -> float:
    return float(z.pow(2).mean().sqrt().item())


def mean_l2_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    t = min(a.shape[-1], b.shape[-1])
    return float((a[..., :t] - b[..., :t]).pow(2).mean().sqrt().item())


def spectral_centroid_hz(x: torch.Tensor, sr: int, n_fft: int = 2048) -> float:
    """Centroid of mean magnitude spectrum (Hz). Empty/zero → 0."""
    xb = as_btc(x).mean(dim=1)[0]
    if float(xb.abs().max()) < 1e-12:
        return 0.0
    spec = torch.fft.rfft(xb, n=n_fft)
    mag = spec.abs()
    freqs = torch.linspace(0, sr / 2, mag.numel(), device=mag.device)
    w = mag.sum().clamp_min(1e-12)
    return float((freqs * mag).sum().item() / float(w))


def peak_bin_hz(x: torch.Tensor, sr: int, n_fft: int = 2048) -> float:
    xb = as_btc(x).mean(dim=1)[0]
    if float(xb.abs().max()) < 1e-12:
        return 0.0
    mag = torch.fft.rfft(xb, n=n_fft).abs()
    idx = int(torch.argmax(mag).item())
    return float(idx * (sr / 2) / max(mag.numel() - 1, 1))


def crop_after_rf(x: torch.Tensor, rf_samples: int) -> torch.Tensor:
    if x.shape[-1] <= rf_samples:
        return x
    return x[..., rf_samples:]


def gather_frame_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    hop: int,
    *,
    silent_db: float,
    rf_frames: int,
    sr: int,
) -> Dict[str, Any]:
    t = min(x.shape[-1], y.shape[-1])
    x = x[..., :t]
    y = y[..., :t]
    xin = frame_rms_dbfs(x, hop)
    yout = frame_rms_dbfs(y, hop)
    n = min(xin.shape[-1], yout.shape[-1])
    xin = xin[0, :n]
    yout = yout[0, :n]
    rf_frames = min(max(0, rf_frames), n)
    late = slice(rf_frames, n)
    silent = xin[late] < silent_db
    loud = xin[late] >= silent_db
    silent_out = yout[late][silent]
    loud_out = yout[late][loud]
    silent_in = xin[late][silent]
    y_late = crop_after_rf(y, rf_frames * hop)
    x_late_silent_audio = None
    if silent.any():
        # Concat hop windows that are silent for spectral stats.
        idxs = torch.where(silent)[0] + rf_frames
        chunks = [y[..., int(i) * hop : (int(i) + 1) * hop] for i in idxs.tolist()]
        x_late_silent_audio = torch.cat(chunks, dim=-1)
    return {
        "n_frames": int(n),
        "n_late_frames": int(n - rf_frames),
        "n_silent_frames": int(silent.sum().item()),
        "n_loud_frames": int(loud.sum().item()),
        "input_rms_dbfs": float(global_rms_dbfs(crop_after_rf(x, rf_frames * hop))),
        "output_rms_dbfs": float(global_rms_dbfs(y_late)),
        "silent_input_mean_dbfs": _mean_or_nan(silent_in),
        "silent_output_mean_dbfs": _mean_or_nan(silent_out),
        "loud_output_mean_dbfs": _mean_or_nan(loud_out),
        "silent_spectral_centroid_hz": (
            spectral_centroid_hz(x_late_silent_audio, sr)
            if x_late_silent_audio is not None
            else float("nan")
        ),
        "silent_peak_hz": (
            peak_bin_hz(x_late_silent_audio, sr)
            if x_late_silent_audio is not None
            else 0.0
        ),
    }


def _mean_or_nan(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(t.mean().item())


def save_rms_plot(
    path: Path,
    x: torch.Tensor,
    series: Sequence[Tuple[str, torch.Tensor]],
    hop: int,
    sr: int,
    threshold_db: float,
) -> None:
    t = x.shape[-1]
    n = t // hop
    times = (np.arange(n) + 0.5) * hop / sr
    fig, ax = plt.subplots(figsize=(10, 4))
    xin = frame_rms_dbfs(x, hop)[0, :n].cpu().numpy()
    ax.plot(times, xin, label="input", color="#333333", lw=1.2)
    for name, wav in series:
        yw = wav[..., :t]
        ydb = frame_rms_dbfs(yw, hop)[0, :n].cpu().numpy()
        ax.plot(times, ydb, label=name, lw=1.0, alpha=0.9)
    ax.axhline(threshold_db, color="#888888", ls="--", lw=0.8, label=f"gate {threshold_db:.0f} dBFS")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frame RMS (dBFS)")
    ax.set_ylim(-90, 0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def encode_quiet_y(
    backbone_y,
    db_path: Path,
    *,
    device: torch.device,
    n_try: int = 32,
    n_signal: int = 65536,
) -> Optional[torch.Tensor]:
    meta_path = db_path / "metadata.yaml"
    if not meta_path.is_file():
        return None
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    lazy = bool(meta.get("lazy", False))
    transform = transforms.Compose([
        lambda x: x.astype(np.float32),
        transforms.RandomCrop(n_signal),
        transforms.Dequantize(16),
        lambda x: x.astype(np.float32),
    ])
    if lazy:
        ds = LazyAudioDataset(
            str(db_path), n_signal, int(meta.get("sr", 44100)), transform, backbone_y.n_channels
        )
    else:
        ds = AudioDataset(
            str(db_path), transforms=transform, n_channels=backbone_y.n_channels, show_progress=False
        )
    best_z = None
    best_db = 0.0
    rng = np.random.default_rng(0)
    n = min(n_try, len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False)
    for i in idxs:
        crop = torch.from_numpy(np.asarray(ds[int(i)])).float().to(device)
        if crop.dim() == 1:
            crop = crop.unsqueeze(0)
        db = global_rms_dbfs(crop)
        if best_z is None or db < best_db:
            best_db = db
            best_z = _encode_mean(backbone_y, crop.unsqueeze(0))
    if best_z is None or best_db > SILENT_FRAME_DB:
        return None
    return best_z


def verdict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    z = metrics["zeros"]
    synth = metrics.get("burst_gap")
    enc_x_nonzero = z["z_x_l2"] > 0.05
    far_from_zy0 = z["dist_zxy_enc_y0"] > 0.2 * max(z["z_y0_l2"], 1e-3)
    transfer_leaks = z["transfer_rms_dbfs"] >= LEAK_DB
    controls_quieter = (
        z["enc_y0_decode_rms_dbfs"] <= z["transfer_rms_dbfs"] - LOUD_MARGIN_DB
        or z["dec_y0_rms_dbfs"] <= z["transfer_rms_dbfs"] - LOUD_MARGIN_DB
    )
    blocked_louder = z["blocked_rms_dbfs"] > z["transfer_rms_dbfs"] + 3.0
    late_gap_leaks = False
    gate_kills = False
    loud_also_ring = False
    if synth is not None:
        s_out = synth["silent_output_mean_dbfs"]
        l_out = synth["loud_output_mean_dbfs"]
        late_gap_leaks = (not math.isnan(s_out)) and s_out >= LEAK_DB
        g = synth.get("gated_mute_silent_dbfs", float("nan"))
        gate_kills = (not math.isnan(g)) and (
            g < GATE_QUIET_DB or (not math.isnan(s_out) and g <= s_out - 12.0)
        )
        loud_also_ring = (
            (not math.isnan(s_out))
            and (not math.isnan(l_out))
            and s_out >= LEAK_DB
            and (l_out - s_out) < LOUD_MARGIN_DB
        )
    hyp = (
        enc_x_nonzero
        and far_from_zy0
        and transfer_leaks
        and controls_quieter
        and late_gap_leaks
        and gate_kills
        and not loud_also_ring
    )
    cache_not_latent = (not transfer_leaks) and blocked_louder
    return {
        "enc_x_silence_nonzero": enc_x_nonzero,
        "z_xy_far_from_enc_y_silence": far_from_zy0,
        "transfer_silence_leaks": transfer_leaks,
        "controls_quieter_than_transfer": controls_quieter,
        "late_gap_leaks": late_gap_leaks,
        "gate_kills_gap": gate_kills,
        "blocked_louder_than_full_context": blocked_louder,
        "loud_regions_also_ring": loud_also_ring,
        "hypothesis_supported": hyp,
        "likely_pad_cache_not_silent_latent": cache_not_latent,
    }


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_probe_wavs(
    out: Path,
    sr: int,
    x: torch.Tensor,
    stems: Dict[str, torch.Tensor],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_audio(out / "probe_in.wav", x.cpu(), sr)
    for name, wav in stems.items():
        save_audio(out / f"{name}.wav", wav.cpu(), sr)


@torch.no_grad()
def run_probe(
    name: str,
    x: torch.Tensor,
    *,
    backbone_x,
    backbone_y,
    warp_xy,
    mode: str,
    hop: int,
    sr: int,
    rf_samples: int,
    threshold_db: float,
    hold_frames: int,
    sidechain_smooth_frames: int,
    block_size: int,
    left_context: int,
    z_y0: torch.Tensor,
    z_y_quiet: Optional[torch.Tensor],
    out_dir: Path,
) -> Dict[str, Any]:
    device = x.device
    x_b = as_btc(x)
    z_x = _encode_mean(backbone_x, x_b)
    z_xy = warp_xy(z_x)
    y = decode_from_z(backbone_y, z_xy, x.shape[-1])[0]
    y_ae = decode_from_z(backbone_y, _encode_mean(backbone_y, x_b), x.shape[-1])[0]
    t_lat = z_xy.shape[-1]
    z0 = torch.zeros_like(z_xy)
    y_dec0 = decode_from_z(backbone_y, z0, x.shape[-1])[0]
    y_enc_y0 = decode_from_z(
        backbone_y,
        z_y0[..., :1].expand(-1, -1, t_lat) if z_y0.shape[-1] != t_lat else z_y0,
        x.shape[-1],
    )[0]

    def _gxy(wave):
        return transfer_x_to_y(wave, backbone_x, backbone_y, warp_xy, mode=mode)  # type: ignore[arg-type]

    y_blocked = transfer_waveform_blocked(
        _gxy, x_b, block_size=block_size, left_context=left_context
    )[0, ..., : x.shape[-1]]

    rms_db = frame_rms_dbfs(x, hop)
    tmin = min(z_xy.shape[-1], rms_db.shape[-1])
    fill = z_y0
    if fill.shape[-1] == 1 or fill.shape[-1] != tmin:
        fill = fill[..., :1]
    z_gated = replace_latent_on_quiet(
        z_xy, fill, rms_db, threshold_db=threshold_db, hold_frames=hold_frames
    )
    y_zy0 = decode_from_z(backbone_y, z_gated, x.shape[-1])[0]
    y_mute = mute_by_input_rms(
        y, x, hop, threshold_db=threshold_db, hold_frames=hold_frames
    )
    y_sc = sidechain_by_input_loudness(
        y, x, hop, threshold_db=threshold_db, smooth_frames=sidechain_smooth_frames
    )

    hop_use = hop if hop > 0 else latent_hop(x.shape[-1], z_x.shape[-1])
    rf_frames = rf_samples // hop_use
    stats = gather_frame_stats(
        x, y, hop_use, silent_db=threshold_db, rf_frames=rf_frames, sr=sr
    )
    mute_stats = gather_frame_stats(
        x, y_mute, hop_use, silent_db=threshold_db, rf_frames=rf_frames, sr=sr
    )
    zy0_stats = gather_frame_stats(
        x, y_zy0, hop_use, silent_db=threshold_db, rf_frames=rf_frames, sr=sr
    )
    sc_stats = gather_frame_stats(
        x, y_sc, hop_use, silent_db=threshold_db, rf_frames=rf_frames, sr=sr
    )

    y_rf = crop_after_rf(y, rf_samples)
    stems = {
        "transfer": y,
        "control_y_ae": y_ae,
        "control_dec_y0": y_dec0,
        "control_enc_y0_decode": y_enc_y0,
        "blocked_512": y_blocked,
        "gated_mute": y_mute,
        "gated_zy0": y_zy0,
        "gated_sidechain": y_sc,
    }
    probe_dir = out_dir / name
    write_probe_wavs(probe_dir, sr, x, stems)
    save_rms_plot(
        probe_dir / "rms.png",
        x,
        [
            ("transfer", y),
            ("y_ae", y_ae),
            ("gated_mute", y_mute),
            ("gated_zy0", y_zy0),
            ("gated_sidechain", y_sc),
        ],
        hop_use,
        sr,
        threshold_db,
    )

    z_y0_exp = z_y0[..., :1].expand_as(z_xy) if z_y0.shape[-1] != z_xy.shape[-1] else z_y0
    row: Dict[str, Any] = {
        "name": name,
        "n_samples": int(x.shape[-1]),
        "hop": hop_use,
        "z_x_l2": mean_l2(z_x),
        "z_xy_l2": mean_l2(z_xy),
        "z_y0_l2": mean_l2(z_y0),
        "dist_zxy_enc_y0": mean_l2_diff(z_xy, z_y0_exp),
        "dist_zxy_enc_y_quiet": (
            mean_l2_diff(z_xy, z_y_quiet) if z_y_quiet is not None else None
        ),
        "transfer_rms_dbfs": float(global_rms_dbfs(y_rf)),
        "enc_y0_decode_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_enc_y0, rf_samples))),
        "dec_y0_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_dec0, rf_samples))),
        "y_ae_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_ae, rf_samples))),
        "blocked_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_blocked, rf_samples))),
        "gated_mute_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_mute, rf_samples))),
        "gated_zy0_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_zy0, rf_samples))),
        "gated_sidechain_rms_dbfs": float(global_rms_dbfs(crop_after_rf(y_sc, rf_samples))),
        "silent_spectral_centroid_hz": stats["silent_spectral_centroid_hz"],
        "silent_peak_hz": stats["silent_peak_hz"],
        **{f"frames_{k}": v for k, v in stats.items()},
        "gated_mute_silent_dbfs": mute_stats["silent_output_mean_dbfs"],
        "gated_zy0_silent_dbfs": zy0_stats["silent_output_mean_dbfs"],
        "gated_sidechain_silent_dbfs": sc_stats["silent_output_mean_dbfs"],
        "device": str(device),
    }
    row.update(stats)
    return row


def build_probes(
    *,
    n_channels: int,
    sr: int,
    rf_samples: int,
    probe_sec: float,
    device,
    dtype,
    real_wav: Optional[torch.Tensor],
) -> List[Tuple[str, torch.Tensor]]:
    n = int(round(probe_sec * sr)) + rf_samples
    probes: List[Tuple[str, torch.Tensor]] = []
    zeros = torch.zeros(n_channels, n, device=device, dtype=dtype)
    probes.append(("zeros", zeros))
    for db in (-60.0, -50.0, -40.0, -30.0):
        noise = noise_at_dbfs((n_channels, n), db, device=device, dtype=dtype)
        probes.append((f"noise_{int(-db)}db", noise))
    probes.append(
        (
            "burst_gap",
            synthetic_burst_gap(n_channels, sr, rf_samples=rf_samples, device=device, dtype=dtype),
        )
    )
    if real_wav is not None:
        probes.append(("real", pad_rf_prefix(real_wav, rf_samples)))
    return probes


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_x = load_model(args.tap_backbone, use_gpu=args.gpu)
    backbone_y = load_model(args.y_backbone, use_gpu=args.gpu)
    warp_xy, _warp_yx, manifest = load_cyclegan_warps_from_checkpoint(
        str(args.cyclegan),
        backbone_x=backbone_x,
        backbone_y=backbone_y,
    )
    warp_xy.eval()
    mode = manifest.canonicalizer_type
    device = next(backbone_x.parameters()).device
    dtype = next(backbone_x.parameters()).dtype
    sr = int(backbone_x.sr)
    n_ch = int(backbone_x.n_channels)
    rf_samples = max(1, int(round(args.rf_drop_sec * sr)))

    dummy = torch.zeros(1, n_ch, rf_samples + int(sr * 0.25), device=device, dtype=dtype)
    z_dummy = _encode_mean(backbone_x, dummy)
    hop = latent_hop(dummy.shape[-1], z_dummy.shape[-1])

    z_y0 = _encode_mean(
        backbone_y,
        torch.zeros(1, n_ch, dummy.shape[-1], device=device, dtype=dtype),
    )
    z_y_quiet = None
    if args.y_db_path is not None:
        z_y_quiet = encode_quiet_y(backbone_y, args.y_db_path, device=device)

    real = None
    if args.input is not None:
        real = load_audio(args.input, backbone_x, device=device)

    probes = build_probes(
        n_channels=n_ch,
        sr=sr,
        rf_samples=rf_samples,
        probe_sec=args.probe_sec,
        device=device,
        dtype=dtype,
        real_wav=real,
    )

    rows: Dict[str, Any] = {}
    for name, x in probes:
        print(f"probe {name}: {x.shape[-1]} samples")
        rows[name] = run_probe(
            name,
            x,
            backbone_x=backbone_x,
            backbone_y=backbone_y,
            warp_xy=warp_xy,
            mode=mode,
            hop=hop,
            sr=sr,
            rf_samples=rf_samples,
            threshold_db=args.threshold_db,
            hold_frames=args.hold_frames,
            sidechain_smooth_frames=args.sidechain_smooth_frames,
            block_size=args.block_size,
            left_context=args.left_context,
            z_y0=z_y0,
            z_y_quiet=z_y_quiet,
            out_dir=out_dir,
        )

    metrics = {
        "cyclegan": str(args.cyclegan),
        "tap_backbone": str(args.tap_backbone),
        "y_backbone": str(args.y_backbone),
        "canonicalizer_type": mode,
        "hop": hop,
        "sr": sr,
        "rf_drop_sec": args.rf_drop_sec,
        "threshold_db": args.threshold_db,
        "hold_frames": args.hold_frames,
        "sidechain_smooth_frames": args.sidechain_smooth_frames,
        "probes": rows,
        "zeros": rows["zeros"],
        "burst_gap": rows.get("burst_gap"),
    }
    metrics["verdict"] = verdict(metrics)
    report_path = out_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(_json_ready(metrics), f, indent=2)
    print(json.dumps(metrics["verdict"], indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
