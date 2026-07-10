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
L_total = λ_rec · L_recon
        + gan_factor · (λ_gan · L̃_gan + λ_fm · L̃_fm)
        + spread_factor · λ_spread · L̃_spread
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

## Latent spread (enabled by default)

Audio GAN + feature matching are **mode-seeking**: OOD latents tend to collapse into a
small region of latent space. The spread term counteracts that by matching the **per-channel
variance** of OOD post-warp latents to in-domain encoder latents, plus optional
training-only noise on OOD `z`.

**Inference is deterministic** — noise and spread are training-only; export uses the same
warp without sampling.

### Spread ramp schedule

Uses the same ramp math as GAN (`compute_gan_ramp_factor`). By default,
`spread_phase_1_duration` and `spread_ramp_duration` copy `phase_1_duration` and
`gan_ramp_duration`.

| Phase | Steps | `spread_factor` | Training |
|-------|-------|-----------------|----------|
| Recon-only | `0 … spread_phase_1_duration−1` | `0` | No spread, no noise |
| Ramp | linear over `spread_ramp_duration` | `0 → 1` | Spread + noise scaled |
| Full spread | after ramp completes | `1` | Full `λ_spread` + noise |

Effective spread terms:

```
λ_spread_eff = spread_factor · λ_spread
noise_std_eff = spread_factor · latent_noise_std_fraction · latent_in_std_mean
```

### Spread loss modes

Reference variance `v_ref[c]` is an EMA of in-domain **pre-warp** encoder `z` (per-channel).
Target is OOD **post-warp** `z`.

**`var_match`** (default):

```
L_spread = mean_c | log Var(z_ood[:,c,:]) − log v_ref[c] |
```

**`var_floor`** (anti-collapse safety net):

```
L_spread = mean_c ReLU( log v_ref[c] − log Var(z_ood[:,c,:]) )
```

### Training-only noise

When `latent_noise_std_fraction > 0`, Gaussian noise is added to OOD post-warp `z`
before decode during training only:

```
ε ~ N(0, noise_std_eff²),   noise_std_eff = spread_factor · fraction · latent_in_std_mean
```

`latent_in_std_mean` is measured at startup calibration from in-domain encoder latents.
Tune the **fraction** (e.g. `0.10` = 10% of natural in-domain spread), not an absolute std.

### Gin parameters

| Parameter | Default (`brave_canonicalizer.gin`) | Role |
|-----------|-----------------------------------|------|
| `lambda_latent_spread` | `0.25` | Top-level spread weight; **`0.0` disables spread loss** |
| `latent_noise_std_fraction` | `0.10` | Noise as fraction of `latent_in_std_mean`; **`0.0` disables noise** |
| `latent_spread_mode` | `"var_match"` | `var_match` or `var_floor` |
| `latent_spread_scale` | `1.0` | Fallback normalization scale |
| `latent_spread_use_ema_ref` | `True` | EMA in-domain variance reference |
| `latent_spread_ema_decay` | `0.99` | EMA decay for `v_ref` |
| `spread_phase_1_duration` | *(unset → `phase_1_duration`)* | Recon-only steps before spread ramps |
| `spread_ramp_duration` | *(unset → `gan_ramp_duration`)* | Linear ramp length for `spread_factor` |

Disable spread entirely:

```gin
lambda_latent_spread = 0.0
latent_noise_std_fraction = 0.0
```

Override spread timing independently of GAN:

```gin
spread_phase_1_duration = 30000
spread_ramp_duration = 10000
```

### Calibration

Startup calibration (when spread or noise is active) also measures:

- `latent_in_std_mean` — mean per-channel std of in-domain pre-warp `z`
- `latent_spread_scale` — mean raw `L_spread` at identity init (when `lambda_latent_spread > 0`)

Saved to `loss_scales.json` alongside STFT/RMS/GAN/FM scales.

### Tuning spread (use metrics, not PCA axis scale)

PCA/t-SNE plots show 2D geometry only. Tune using W&B scalars:

| Metric | Target | Action if off |
|--------|--------|---------------|
| `canon/latent_ood_in_var_ratio` | ≈ `1.0` | Raise `lambda_latent_spread` or `latent_noise_std_fraction` if ≪ 1 |
| `val/recon_ood` | stable | Lower spread weights if recon degrades |
| `canon/latent_spread_norm` | ~0.5–2.0 | Compare to other normalized terms |

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
| `canon/spread_factor` | Spread ramp weight in `[0, 1]` |
| `canon/spread_warmed_up` | `1.0` once `spread_factor` reaches `1.0` |
| `canon/latent_spread_norm` | Normalized spread loss / `latent_spread_scale` |
| `canon/latent_spread_raw` | Raw spread loss |
| `canon/latent_ood_in_var_ratio` | Mean per-channel Var_ood / Var_in_ref (primary spread dial) |
| `canon/latent_noise_std_effective` | Effective OOD noise std this step |
| `canon/latent_in_std_mean` | Calibrated in-domain latent std reference |
| `val/recon_ood` | Raw STFT recon on OOD validation |
| `val/rms_ood` | Raw RMS recon on OOD validation |
| `val/disc_ood` | Mean fake logit on OOD (sanity check D is learning) |
| `val/ood_by_class` | Single grouped bar chart: mean fake D logit + STFT recon per assigned OOD class (Fader) |

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

### Disable latent spread

```gin
lambda_latent_spread = 0.0
latent_noise_std_fraction = 0.0
```

### More OOD latent spread

```gin
lambda_latent_spread = 0.35
latent_noise_std_fraction = 0.15
```

Watch `canon/latent_ood_in_var_ratio` → 1.0 and guard with `val/recon_ood`.

## CycleGAN mapping

| CycleGAN | Canonicalizer |
|----------|---------------|
| Generator X→Y | Warp + frozen decode path on OOD |
| Discriminator on Y | `InDomainAudioDiscriminator` |
| Cycle consistency | `L_rec` on OOD (and in-domain) |
| Identity | In-domain `L_rec` + identity init of warp |

Full one-way transfer: we do **not** train a reverse mapper or an OOD discriminator.

## Conditional FaderRAVE (attribute-conditioned D)

For Fader backbones (`num_attributes > 0`), **Y** is a family of distributions
`Y | a` — one per attribute setting. The discriminator conditions on discretized
attributes (`attr_cls`): continuous rows use quantile bins; discrete rows use class
indices (same representation as the Fader latent discriminator).

```
L_D = E_{y,a~p_Y} [hinge(D(y|a), real)] + E_{x,a~p_target} [hinge(D(G(x|a), fake)]
```

Plain BRAVE (`num_attributes = 0`) passes `attr_cls=None` and recovers the
audio-only MSD.

### OOD attribute policy (training)

| Kind | OOD (tap) | In-domain (Y) |
|------|-----------|---------------|
| Continuous (RMS, …) | **Tap-extracted** (keep from descriptor provider) | Natural from dataset |
| Discrete (`texture_class`, …) | **Sampled target class** (Y marginal by default; `uniform` via gin) | Natural from sidecar |

Decode always uses `attr_norm`. D and warp FiLM use `attr_cls` by default.

### Phase 3 training options (Fader)

| Gin / flag | Default | Effect |
|------------|---------|--------|
| `ood_discrete_sampling` | `"marginal"` | OOD discrete rows drawn from train-split class histogram in `attribute_sidecar.yaml` |
| `class_stratified_batches` | `True` | In-domain slots in each batch round-robin across discrete classes |
| `val/ood_by_class` | — | One W&B figure: grouped bars of mean fake D logit + STFT recon per OOD class |

Use `ood_discrete_sampling = "uniform"` to restore equal class exposure regardless of Y frequency.

### Inference attribute modes (nn~)

See [`docs/fader_host_controls.md`](../fader_host_controls.md). Users can adjust
continuous knobs (`attr_mode=0/1`) or use extract + manual discrete (`attr_mode=2`).
Training defaults match extract-continuous + sampled-discrete-class.

### D conditioning: `attr_cls` vs `attr_norm`

| Criterion | `attr_cls` (default) | `attr_norm` (ablation) |
|-----------|----------------------|------------------------|
| Mechanism | Per-attribute `nn.Embedding`; pool; projection onto D | Pool `attr_norm`; linear projection |
| Continuous | Coarse (16 bins) | Fine-grained trajectories |
| Discrete | Categorical class buckets | Decoder floats in [-1, 1] (interpolable) |
| Best for | `texture_class` / `water_scene` | Continuous-only ablation |

Gin: `condition_on = "attr_cls"` or `"attr_norm"` on
`InDomainAudioDiscriminator`.

Plan reference: [`scratchpaper/conditional_canonicalizer_plan.md`](../../scratchpaper/conditional_canonicalizer_plan.md).
