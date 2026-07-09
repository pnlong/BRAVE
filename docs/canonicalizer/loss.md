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

Mixed batches are built by [`build_canonicalizer_dataset`](../../RAVE/rave/canonicalizer/dataset.py)
(`in_domain_fraction` in gin, default `0.5`).

## Total loss (generator / warp step)

On warp optimizer steps:

```
L_total = λ_rec · L_rec + λ_gan · L_GAN + λ_fm · L_feature_matching
```

| Gin parameter | Default | Role |
|---------------|---------|------|
| `lambda_rec` | `0.05` | Scales all reconstruction terms |
| `lambda_gan` | `1.0` | Hinge GAN generator loss on OOD fakes |
| `lambda_feature_matching` | `2.0` | L1 match of D intermediate features (real Y vs fake G(X)) |
| `lambda_rms_recon` | `1.0` | Extra weight on RMS envelope term *inside* `L_rec` |

There is **no** separate `lambda_stft` — STFT recon enters only through `lambda_rec`.

Discriminator steps optimize `L_D` only (hinge GAN on D), on a separate Adam optimizer,
every `update_discriminator_every` batches (default `2`) once GAN is active.

## Reconstruction loss

```
L_rec = L_stft + λ_rms_recon · L_rms
```

Both STFT and RMS are computed **per domain** and summed:

- **In-domain** (`recon_in_domain_mode`): self-reconstruction on Y — G(y) ≈ y
- **OOD** (`recon_ood_mode`): cycle-identity proxy on X — G(x) ≈ x

Each mode is gin-selectable: `"stft"`, `"rms"`, or `"both"` (current default).

### STFT term (`L_stft`)

Uses the frozen backbone's `AudioDistanceV1` (same metric as RAVE pretraining):

```
L_stft = multiband_audio_distance(x_mb, y_mb)
       + audio_distance(x_raw, y_raw)
```

Each distance sums **5 multi-scale STFT** scales; per scale it adds relative L2 (linear
spec) + L1 (log spec). Multiband + fullband are both included.

**Typical raw magnitude: ~40–50** for imperfect reconstructions. This is expected — not a
bug and not comparable in scale to GAN losses (~0–2).

### RMS term (`L_rms`)

Differentiable frame-wise RMS envelope L1 ([`rms_recon_l1`](../../RAVE/rave/canonicalizer/losses.py)),
aligned to latent frame count.

**Typical raw magnitude: ~0–1.**

### Effective recon contribution

With defaults and typical running values:

| Component | Raw | × weights | Effective in `L_total` |
|-----------|-----|-----------|------------------------|
| STFT | ~45 | `λ_rec` (0.05) | **~2.2** |
| RMS | ~0.3 | `λ_rec × λ_rms_recon` (0.05) | **~0.015** |
| GAN | ~1 | `λ_gan` (1.0) | **~1** |
| Feature matching | ~0.5 | `λ_fm` (2.0) | **~1** |

**Important:** `lambda_rec = 1.0` makes STFT dominate (~45 vs ~1 for GAN). Earlier runs
that looked successful under `λ_rec = 1.0` were largely driven by reconstruction, not
adversarial timbre shift. Default is now `0.05` so GAN + feature matching lead.

## GAN loss

Active only when all of the following hold:

1. `global_step >= phase_1_duration` (`WarmupCallback` sets `warmed_up`; default **2000** steps)
2. Batch contains both in-domain and OOD samples
3. `in_domain_disc` is configured

**Phase 1 (not warmed up):** only `λ_rec · L_rec` trains the warp — recon-only warmup.

**Phase 2 (warmed up):**

- **D step** (every `update_discriminator_every` batches): classify real Y vs fake G(X)
- **G step** (other batches): fool D + feature matching on intermediate activations

GAN loss uses the same hinge / LS / nonsaturating callables as RAVE (`gan_loss` gin string).

## WandB / log metrics

| Metric | What it shows |
|--------|----------------|
| `canon/loss` | Total warp loss (or D loss on disc steps) |
| `canon/recon_stft` | Raw STFT recon (**unweighted** — expect ~40–50) |
| `canon/recon_rms` | Raw RMS recon (**unweighted**) |
| `canon/recon` | `L_stft + λ_rms_recon · L_rms` (before `λ_rec`) |
| `canon/gan` | Raw GAN generator loss |
| `canon/feature_matching` | Raw feature-matching loss |
| `canon/audio_disc` | D loss (logged on discriminator steps) |
| `canon/warmed_up` | `1.0` once `phase_1_duration` elapsed |
| `val/recon_ood` | STFT recon on OOD validation |
| `val/rms_ood` | RMS recon on OOD validation |
| `val/disc_ood` | Mean fake logit on OOD (sanity check D is learning) |

Do not compare `canon/recon_stft` directly to `canon/gan` without applying the λ weights.

## Tuning guide

### Less reconstruction, more timbre shift

Lower `lambda_rec`:

```bash
--override 'CanonicalizerTrainer.lambda_rec=0.01'
```

### RMS-only recon (drop STFT from training)

```gin
recon_ood_mode = "rms"
recon_in_domain_mode = "rms"
```

### Disable recon entirely

```gin
lambda_rec = 0.0
```

### Longer recon-only warmup before GAN

```gin
phase_1_duration = 5000
```

### Stronger envelope matching relative to STFT (within recon)

Raise `lambda_rms_recon` (e.g. `5.0`). There is no `lambda_stft_recon` knob; STFT weight
is fixed at `1.0` inside `L_rec`.

## CycleGAN mapping

| CycleGAN | Canonicalizer |
|----------|---------------|
| Generator X→Y | Warp + frozen decode path on OOD |
| Discriminator on Y | `InDomainAudioDiscriminator` |
| Cycle consistency | `L_rec` on OOD (and in-domain) |
| Identity | In-domain `L_rec` + identity init of warp |

Full one-way transfer: we do **not** train a reverse mapper or an OOD discriminator.
