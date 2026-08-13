# CycleGAN: Tap ↔ Water / Birdsong

Bidirectional CycleGAN between two **frozen** plain BRAVE backbones (X = tap, Y = water or birdsong).

Implementation: [`RAVE/rave/canonicalizer/cycle_trainer.py`](../../RAVE/rave/canonicalizer/cycle_trainer.py).  
Config: [`configs/brave_cyclegan.gin`](../../configs/brave_cyclegan.gin).  
Warp module: [`LatentCanonicalizer`](../canonicalizer/latent.md).  
One-way Stage-1 (not CycleGAN): [`docs/canonicalizer/README.md`](../canonicalizer/README.md).

## Geometry (Approach 2)

Warps are **cross-space**. Inference (and the main train path) encode a clip with **its** domain encoder; warp only when transferring:

| Intent | Path |
|--------|------|
| Stay tap | `Enc_X → Dec_X` (no warp) |
| Stay birdsong | `Enc_Y → Dec_Y` (no warp) |
| Tap → birdsong | `Enc_X(x) → W_xy → Dec_Y` |
| Birdsong → tap | `Enc_Y(y) → W_yx → Dec_X` |

Waveform **warps** (EQ/reverb before a single backbone) are **shelved**. Always pass `--canonicalizer_type latent`.

Do not confuse **warp type** with **cycle/GAN domain** (`cycle_domain` below).

---

## A. CycleGAN modes: waveform vs latent

`CycleGANTrainer.cycle_domain` chooses where **cycle consistency** and the **discriminator** live. Both modes still use the same cross-space latent warps.

| `cycle_domain` | Cycle loss | Discriminator | Decode in train? | Warmup |
|----------------|------------|---------------|------------------|--------|
| `"waveform"` (default) | STFT + RMS on audio round-trip | Audio MSD (`D_X`, `D_Y` on waveforms) | yes | 50k |
| `"latent"` | L1 on `z` | Latent D on `z` | AE-aware: yes; direct: **no** | AE-aware: 50k; direct: **0** |

```bash
# waveform cycle + audio D (default)
# (no override)

# latent cycle + latent D
OVERRIDE='CycleGANTrainer.cycle_domain="latent"'
```

SLURM: `CYCLE_DOMAIN=waveform` or `CYCLE_DOMAIN=latent`.

---

## B. Loss components

Total warp/G step (terms that are off for a given mode are zero):

```
L = λ_cycle · L_wave
  + λ_latent_cycle · L_z
  + gan_factor · (λ_gan · L_gan + λ_fm · L_fm)
  + λ_identity · L_id
  + spread_factor · λ_spread · L_spread
```

Each raw term is divided by an empirical scale (startup calibration) before λ. Defaults from [`brave_cyclegan.gin`](../../configs/brave_cyclegan.gin):

| Term | Gin | Default λ | When it is on | What it measures |
|------|-----|-----------|---------------|------------------|
| Waveform cycle `L_wave` | `lambda_cycle` | 10 | `cycle_domain="waveform"` | `x ≈ G_yx(G_xy(x))` and `y ≈ G_xy(G_yx(y))` via STFT (`cycle_stft_weight=0.9`) + RMS (`cycle_rms_weight=0.1`) |
| Latent cycle `L_z` | `lambda_latent_cycle` | 10 | `cycle_domain="latent"` | L1 on `z` — AE-aware or direct (section C) |
| GAN `L_gan` | `lambda_gan` | 1 | after warmup, `gan_factor>0` | Hinge: fool `D_Y` / `D_X`. **Audio D:** real `y` vs `Dec_Y(W_xy(Enc_X(x)))`. **Latent D:** real `Enc_Y(y)` vs `W_xy(Enc_X(x))` (and vice versa) |
| Feature matching `L_fm` | `lambda_feature_matching` | 0.5 | with GAN | Match D intermediate features (real vs fake) |
| Identity `L_id` | `lambda_identity` | **0** (off) | only if you set λ>0 | See below. Not used at inference. |
| Latent spread `L_spread` | `lambda_latent_spread` | 0.25 | latent warps, after GAN ramp | Match per-channel variance of post-warp `z` to in-domain encoder `z` |

D step (separate optimizer, every `update_discriminator_every=2` batches once GAN is active): hinge on real vs fake only.

**Identity (optional, default off).** Classic CycleGAN idea: don’t let \(G\) rewrite samples already in the target domain. Under Approach 2 that means a path you **do not** use at inference: encode \(y\) with `Enc_X` (wrong encoder), then ask \(W_{xy}(\mathrm{Enc}_X(y)) \approx \mathrm{Enc}_Y(y)\) (latent mode) or `Dec_Y(W_xy(Enc_X(y))) ≈ y` (waveform mode). Main cycle + D never need this.

---

## C. Latent cycle: AE-aware vs direct

Only when `cycle_domain="latent"`. Switch: `CycleGANTrainer.latent_cycle_mode` / SLURM `LATENT_CYCLE_MODE`.

### AE-aware (`"ae_aware"`, default)

\[
z_x \approx W_{yx}\big(\mathrm{Enc}_Y(\mathrm{Dec}_Y(W_{xy}(z_x)))\big)
\]

\(P_Y = \mathrm{Enc}_Y \circ \mathrm{Dec}_Y\) sits in the middle. \(P_Y\) is ~identity only on **valid Y codes**. Off-manifold it is a lossy projection, so the round-trip only closes if \(W_{xy}(z_x)\) lands on the Y decoder manifold **and** still carries tap content. Uses the frozen Y AE as a pretrained “is this a real Y latent?” critic. Decode every train step. Warmup 50k.

### Direct (`"direct"`)

\[
z_x \approx W_{yx}(W_{xy}(z_x))
\]

Warps must be inverses. No Dec/Enc in the cycle — **no decode in `training_step`** (val still decodes for audio). Warmup forced to **0**. Relies on the **latent D** to put \(W_{xy}(z_x)\) on the `Enc_Y(y)` cloud. Faster; weaker if latent D only matches prior-like \(\mathcal{N}(0,I)\) blobs.

| | AE-aware | Direct |
|--|----------|--------|
| Constraint | invertible *through Y’s codec* | warps invertible |
| Train decode | yes | no |
| Warmup | 50k | 0 |
| Speed | slower | faster |
| Defines “valid Y \(z\)” | frozen Enc∘Dec | learned latent D |

Both directions: same formulas with \(X\leftrightarrow Y\).

```bash
OVERRIDE='CycleGANTrainer.cycle_domain="latent"'
OVERRIDE='CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.latent_cycle_mode="direct"'
```

---

## D. Warp initialization: identity vs random

Same module (`LatentCanonicalizer`); `init_mode` changes **init and forward**.

| `init_mode` | Who uses it | Forward | Init |
|-------------|-------------|---------|------|
| `"identity"` | **Stage-1** one-way canonicalizer | residual \(L(z)=z+\sigma(\alpha)(f(z)-z)\) | \(L(z)=z\) exactly (1×1 = I, or last conv zero) |
| `"random"` | **All CycleGAN** latent warps | \(L(z)=f(z)\) (no residual gate) | Orthogonal 1×1 or Kaiming MLP; output \(z\) ~unit variance |

Identity is the right prior for **within-space** Stage-1 (nudge OOD codes inside Y). Cross-space CycleGAN (\(Z_X \not\approx Z_Y\) as coordinates) uses random; `train_cyclegan.py` always constructs warps with `init_mode="random"` regardless of gin `LatentCanonicalizer.init_mode` (gin default stays `"identity"` for Stage-1).

Manifest field `init_mode` is required at load so inference uses the same forward (residual vs not).

---

## Recipes

From BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# A — waveform cycle + audio D
python RAVE/scripts/train_cyclegan.py \
  --config configs/brave_cyclegan.gin \
  --backbone_x_config configs/brave.gin \
  --ckpt_x /path/to/tap_run.ckpt --db_path_x /path/to/tap_lmdb \
  --backbone_y_config configs/brave.gin \
  --ckpt_y /path/to/water_run.ckpt --db_path_y /path/to/water_lmdb \
  --canonicalizer_type latent \
  --name tap_water_wf_cycle

# B — latent D + AE-aware z cycle
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent"' \
  --name tap_water_lat_ae

# C — latent D + direct compose-L1
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.latent_cycle_mode="direct"' \
  --name tap_water_lat_direct
```

SLURM (`scripts/train_cyclegan.sbatch`, 1 GPU): set `CKPT_X`, `DB_PATH_X`, `CKPT_Y`, `DB_PATH_Y`, `RUN_NAME`; optional `CYCLE_DOMAIN`, `LATENT_CYCLE_MODE`, `OVERRIDE`. `CANONICALIZER_TYPE` defaults to `latent`.

## Training schedule

1. **Cycle warmup** (`cycle_warmup_duration`, default 50k; **0** for direct): cycle only, no D
2. **GAN ramp** (`gan_ramp_duration`, default 5k): `gan_factor` 0→1 (spread ramps with GAN)
3. **Full**: cycle + hinge GAN + FM (+ spread)

## Outputs

`runs/<name>_<hash>/`:

```
cyclegan_latent.ckpt
cyclegan_latent.manifest.json   # cycle_domain, latent_cycle_mode, init_mode, …
config.gin
loss_scales.json
viz/x_val_step*.wav             # input | G_xy(x) | x_cycle
viz/y_val_step*.wav
viz/x_to_y_val_step*.wav        # transfer only
viz/y_to_x_val_step*.wav
```

## Inference

```python
from rave.canonicalizer.cycle_inference import (
    load_cyclegan_warps_from_checkpoint,
    transfer_x_to_y,
    transfer_y_to_x,
)

warp_xy, warp_yx, manifest = load_cyclegan_warps_from_checkpoint(
    "runs/tap_water_lat_cyclegan_abc/cyclegan_latent.ckpt",
    backbone_x=backbone_x,
    backbone_y=backbone_y,
)
y = transfer_x_to_y(x_tap, backbone_x, backbone_y, warp_xy, mode="latent")
```

Always `Enc_src → warp → Dec_tgt`.

## W&B metrics

| Key | Meaning |
|-----|---------|
| `cycle/gan_factor` | 0 during warmup, ramps to 1 |
| `cycle/cycle_x_norm`, `cycle/cycle_y_norm` | Normalized waveform cycle |
| `cycle/latent_cycle_norm` | Normalized z cycle (AE-aware or direct) |
| `cycle/adv_norm`, `cycle/fm_norm` | Normalized GAN / FM |
| `cycle/spread_norm` | Normalized spread |
| `val/cycle_x`, `val/cycle_y` | Val waveform cycle |
| `val/latent_cycle_x`, `val/latent_cycle_y` | Val z cycle |
| `val/disc_x_fake`, `val/disc_y_fake` | Fake logits (audio or latent D) |
| `val/audio_x`, `val/audio_y` | `input \| transfer \| cycle` |
| `val/audio_x_to_y`, `val/audio_y_to_x` | Transfer only |
| `val/latent_x_pca` | `Enc_X(x)` vs `z_yx` |
| `val/latent_y_pca` | `Enc_Y(y)` vs `z_xy` |

## Other gin overrides

```bash
OVERRIDE='LatentCanonicalizer.n_layers=2'
OVERRIDE='CycleGANTrainer.cycle_warmup_duration=0'
OVERRIDE='CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'
```

Legacy `use_waveform_cycle` / `use_latent_cycle` still apply if `cycle_domain` is unset (audio D always).
