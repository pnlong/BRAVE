# Canonicalizer training loss

Stage-1 canonicalizer training adapts a small warp (waveform or latent) on a **frozen**
RAVE / FaderRAVE backbone so out-of-domain (OOD) audio reconstructs with statistics
closer to the in-domain corpus. The objective is a one-way CycleGAN-style setup:
push OOD reconstructions toward in-domain audio while keeping self-reconstruction as a
light regularizer.

Implementation: [`RAVE/rave/canonicalizer/trainer.py`](../../RAVE/rave/canonicalizer/trainer.py).
Gin defaults: [`configs/brave_canonicalizer.gin`](../../configs/brave_canonicalizer.gin).

## Problem framing

| Symbol | Meaning |
|--------|---------|
| **X** | Out-of-domain batch (e.g. tap LMDB) |
| **Y** | In-domain batch (backbone training LMDB) |
| **G** | Canonicalizer + frozen Enc/Dec → reconstructed waveform |
| **D** | `InDomainAudioDiscriminator` — real Y vs fake G(X) |

Mixed batches use stratified sampling via
[`build_canonicalizer_dataloader`](../../RAVE/rave/canonicalizer/dataset.py)
(`in_domain_fraction` in gin, default `0.5`).

## Total loss (generator / warp step)

Raw terms (STFT, RMS, GAN, FM) live on very different scales. Each is divided by an
**empirical reference scale** measured on your run (see below) so typical values are ~1,
then combined with explicit λ weights.

```
L_total = λ_rec · L_recon + λ_gan · L̃_gan + λ_fm · L̃_fm
```

where `L̃_x = L_x / scale_x` (normalized), and:

```
L_recon = w_stft · L̃_stft + w_rms · L̃_rms
```

| Gin parameter | Default | Role |
|---------------|---------|------|
| `lambda_rec` | `1.0` | Top-level reconstruction bundle |
| `lambda_gan` | `1.0` | Normalized GAN generator loss |
| `lambda_feature_matching` | `2.0` | Normalized feature-matching loss |
| `recon_stft_weight` | `0.9` | Sub-weight on normalized STFT inside `L_recon` |
| `recon_rms_weight` | `0.1` | Sub-weight on normalized RMS inside `L_recon` |
| `calibrate_loss_scales` | `True` | Measure scales from data at startup |
| `calibration_batches` | `16` | Stratified train batches used for calibration |
| `stft_loss_scale` | `45.0` | Fallback STFT scale if calibration disabled |
| `rms_loss_scale` | `0.3` | Fallback RMS scale if calibration disabled |
| `gan_loss_scale` | `1.0` | Fallback GAN scale if calibration disabled |
| `fm_loss_scale` | `0.5` | Fallback FM scale if calibration disabled |

### Calibrating loss scales (default)

Before training, `train_canonicalizer.py` runs **identity-warp calibration** on the
train loader:

1. Warp at initialization (near-identity for waveform; zero residual for latent).
2. Frozen backbone encode/decode on stratified in-domain + OOD batches.
3. Mean raw STFT, RMS, GAN, and FM over `calibration_batches` (default 16).
4. Set `scale_x = max(mean, loss_scale_min)` for STFT/RMS.
5. For GAN/FM: use the measured mean only if it is meaningfully above zero;
   otherwise keep gin fallback scales (identity warp often yields ~0 adversarial
   loss before the GAN ramp, which would inflate normalized GAN/FM later).

This is the recommended approach: scales come from **your** backbone checkpoint, **your**
in-domain LMDB, and **your** OOD tap corpus — not hand-waved constants. Results are saved
to `loss_scales.json` in the run directory and logged to W&B config.

Disable with `--no_calibrate_scales` (uses gin fallback scales only).

**Tuning intuition:** after calibration, each normalized term is ~1 at step 0, so
`λ_rec ≈ λ_gan` means comparable recon vs adversarial pull. Sub-weights `w_stft` / `w_rms`
set the mix inside recon (e.g. 90% spectral, 10% envelope).

Discriminator steps optimize raw `L_D` only (hinge GAN on D), on a separate Adam optimizer,
every `update_discriminator_every` batches (default `2`) once GAN is active.

## Reconstruction loss

Both STFT and RMS are computed **per domain** (in-domain self-recon + OOD cycle proxy),
normalized, weighted, and summed:

```
L_recon = w_stft · (L_stft / s_stft) + w_rms · (L_rms / s_rms)
```

Each mode is gin-selectable per domain: `"stft"`, `"rms"`, or `"both"` (current default).

### STFT term (`L_stft`)

Uses the frozen backbone's `AudioDistanceV1` (same metric as RAVE pretraining):

```
L_stft = multiband_audio_distance(x_mb, y_mb)
       + audio_distance(x_raw, y_raw)
```

**Typical raw magnitude: ~40–50.** After normalization (`/ 45`), ~1.

### RMS term (`L_rms`)

Differentiable frame-wise RMS envelope L1 ([`rms_recon_l1`](../../RAVE/rave/canonicalizer/losses.py)).

**Typical raw magnitude: ~0.2–0.4.** After normalization (`/ 0.3`), ~1.

### Effective contribution at defaults

| Component | Raw (typical) | Normalized | × λ / w | Effective in `L_total` |
|-----------|---------------|------------|---------|------------------------|
| STFT | ~45 | ~1 | `λ_rec · w_stft` (0.9) | **~0.9** |
| RMS | ~0.3 | ~1 | `λ_rec · w_rms` (0.1) | **~0.1** |
| GAN | ~1 | ~1 | `λ_gan` (1.0) | **~1** |
| Feature matching | ~0.5 | ~1 | `λ_fm` (2.0) | **~2** |

## GAN loss

Active only when all of the following hold:

1. `gan_factor > 0` (set each step by `CanonicalizerGanRampCallback`)
2. Batch contains both in-domain and OOD samples (guaranteed with stratified batching)
3. `in_domain_disc` is configured

### GAN ramp schedule (default)

| Phase | Steps | `gan_factor` | Training |
|-------|-------|--------------|----------|
| Recon-only | `0 … phase_1_duration-1` (default **1000**) | `0` | Only `λ_rec · L_recon` |
| Ramp | `phase_1_duration … phase_1_duration + gan_ramp_duration - 1` (default **1000–5999**) | linear `0 → 1` | GAN/FM weight scaled by `gan_factor` |
| Full GAN | `≥ phase_1_duration + gan_ramp_duration` (default **6000+**) | `1` | Full adversarial training |

Effective generator adversarial terms:

```
λ_gan_eff = gan_factor · λ_gan
λ_fm_eff  = gan_factor · λ_fm
```

| Gin parameter | Default | Role |
|---------------|---------|------|
| `phase_1_duration` | `1000` | Recon-only steps before GAN ramp starts |
| `gan_ramp_duration` | `5000` | Linear ramp length for `gan_factor` |

Set `gan_ramp_duration = 0` for an immediate step from recon-only to full GAN (hard gate).

**During ramp:**

- **G step**: `λ_rec · L_recon + gan_factor · (λ_gan · L̃_gan + λ_fm · L̃_fm)`
- **D step** (every `update_discriminator_every` batches, once `gan_factor > 0`): raw hinge D loss

## WandB / log metrics

**Normalized** (`*_norm`): comparable ~1-scale terms that enter `L_total` (after
empirical scale division). Use these when tuning λ weights or comparing recon vs GAN vs FM.

**Raw** (`*_raw`): unnormalized magnitudes for absolute-quality diagnostics.

| Metric | What it shows |
|--------|----------------|
| `canon/loss` | Total warp loss (or D loss on disc steps) |
| `canon/recon_norm` | Normalized weighted recon bundle (before `λ_rec`) |
| `canon/recon_in_norm` | Normalized weighted recon, in-domain only |
| `canon/recon_ood_norm` | Normalized weighted recon, OOD only |
| `canon/gan_norm` | GAN generator loss / `gan_loss_scale` |
| `canon/fm_norm` | Feature-matching loss / `fm_loss_scale` |
| `canon/recon_stft_raw` | Raw STFT recon (diagnostic — expect ~40–50) |
| `canon/recon_rms_raw` | Raw RMS recon (diagnostic — expect ~0.2–0.4) |
| `canon/gan_raw` | Raw GAN generator loss |
| `canon/fm_raw` | Raw feature-matching loss |
| `canon/audio_disc` | D loss (logged on discriminator steps only) |
| `canon/gan_factor` | GAN ramp weight in `[0, 1]` (linear ramp after recon-only phase) |
| `canon/warmed_up` | `1.0` once `gan_factor` reaches `1.0` |
| `val/recon_ood` | Raw STFT recon on OOD validation |
| `val/rms_ood` | Raw RMS recon on OOD validation |
| `val/disc_ood` | Mean fake logit on OOD (sanity check D is learning) |

Calibration scales are saved to `loss_scales.json` and W&B run config (not logged per step).

## Tuning guide

### Less reconstruction, more timbre shift

Lower `lambda_rec` (e.g. `0.5` or `0.25`).

### More envelope vs spectral inside recon

Raise `recon_rms_weight` and lower `recon_stft_weight` (they need not sum to 1, but ~1 is
intuitive):

```gin
recon_stft_weight = 0.7
recon_rms_weight = 0.3
```

### Recalibrate if corpus or backbone changes

Scales are measured once per run. If you change tap LMDB, birdsong ckpt, or `n_signal`,
recalibration happens automatically on the next run. To reuse fixed scales (e.g. ablation
sweeps with identical data), pass `--no_calibrate_scales` and set gin fallbacks explicitly.

### RMS-only recon (drop STFT from training)

```gin
recon_ood_mode = "rms"
recon_in_domain_mode = "rms"
```

### Disable recon entirely

```gin
lambda_rec = 0.0
```

### Longer recon-only before GAN ramp

```gin
phase_1_duration = 2000
gan_ramp_duration = 5000
```

### Instant GAN (no ramp)

```gin
gan_ramp_duration = 0
```

## CycleGAN mapping

| CycleGAN | Canonicalizer |
|----------|---------------|
| Generator X→Y | Warp + frozen decode path on OOD |
| Discriminator on Y | `InDomainAudioDiscriminator` |
| Cycle consistency | `L_rec` on OOD (and in-domain) |
| Identity | In-domain `L_rec` + identity init of warp |

Full one-way transfer: we do **not** train a reverse mapper or an OOD discriminator.
