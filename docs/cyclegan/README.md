# CycleGAN: Tap ↔ Water / Birdsong

Bidirectional CycleGAN for unpaired domain transfer (X = tap, Y = water or birdsong).

Implementation: [`RAVE/rave/canonicalizer/cycle_trainer.py`](../../RAVE/rave/canonicalizer/cycle_trainer.py).  
Configs:
- Approach 2 (separate codecs): [`configs/brave_cyclegan_separate.gin`](../../configs/brave_cyclegan_separate.gin) — [`brave_cyclegan.gin`](../../configs/brave_cyclegan.gin) is a back-compat include
- Approach 3 (joint embedding): [`configs/brave_cyclegan_joint.gin`](../../configs/brave_cyclegan_joint.gin)

Warp module: [`LatentCanonicalizer`](../canonicalizer/latent.md).  
One-way Stage-1 (not CycleGAN): [`docs/canonicalizer/README.md`](../canonicalizer/README.md).  
Design notes: [`scratchpaper/joint_embedding_cyclegan.md`](../../scratchpaper/joint_embedding_cyclegan.md).  
Project arc (zero-shot → manual → learned → joint): [`docs/domain_adaptation_arc.md`](../domain_adaptation_arc.md).

## Geometry (Approach 2 — separate)

Warps are **cross-space**. Inference (and the main train path) encode a clip with **its** domain encoder; warp only when transferring:

| Intent | Path |
|--------|------|
| Stay tap | `Enc_X → Dec_X` (no warp) |
| Stay birdsong | `Enc_Y → Dec_Y` (no warp) |
| Tap → birdsong | `Enc_X(x) → W_xy → Dec_Y` |
| Birdsong → tap | `Enc_Y(y) → W_yx → Dec_X` |

## Geometry (Approach 3 — joint)

Train **one** plain BRAVE on both domains (stratified dual-LMDB: `--db_path` + `--db_path_y`), freeze it, then CycleGAN with **within-space** warps (`CycleGANTrainer.shared_backbone=True`):

| Intent | Path |
|--------|------|
| Stay X / stay Y | `Enc → Dec` (no warp) |
| X → Y | `Enc(x) → W_xy → Dec` |
| Y → X | `Enc(y) → W_yx → Dec` |

Still uses **two** unpaired LMDBs for cycle batches. Warp init is **identity** residual (Stage-1 prior); identity loss λ is on by default in the joint gin. Latent AE-aware cycle still runs Enc∘Dec, but that is the shared AE manifold (not a foreign codec) — domain signal comes mainly from latent/audio D.

```bash
python RAVE/scripts/train_cyclegan.py \
  --config configs/brave_cyclegan_joint.gin \
  --backbone_x_config configs/brave.gin \
  --ckpt_x /path/to/joint_run.ckpt \
  --db_path_x /path/to/tap_lmdb \
  --db_path_y /path/to/water_lmdb \
  --canonicalizer_type latent \
  --name tap_water_joint_wf
```

`--ckpt_y` / `--backbone_y_config` are optional under joint (default to X).

Waveform **warps** (EQ/reverb before a single backbone) are **shelved**. Always pass `--canonicalizer_type latent`.

Do not confuse **warp type** with **cycle/GAN domain** (`cycle_domain` below).

---

## A. CycleGAN modes: waveform vs latent

`CycleGANTrainer.cycle_domain` chooses where **cycle consistency** and the **primary discriminator** live. Both modes still use the same cross-space latent warps.

| `cycle_domain` | Cycle loss | Discriminator | Decode in train? | Warmup |
|----------------|------------|---------------|------------------|--------|
| `"waveform"` (default) | STFT + RMS on audio round-trip | Audio MSD (`D_X`, `D_Y` on waveforms) | yes | 50k |
| `"latent"` | L1 on `z` | Latent D on `z` | AE-aware: yes; direct: **no** | AE-aware: 50k; direct: **0** |

**Manifold note.** Waveform cycle + audio D can sound strong while `z_xy = W_xy(Enc_X(x))` sits **off** the support of `Enc_Y(y)`. Good STFT / Max audio ≠ PCA overlap. Latent AE-aware pressures codes onto Y’s codec manifold (stronger PCA), but decoded audio may ring until polished. **Spread** (`λ_latent_spread`) only matches per-channel variance — it is **not** a manifold membership loss.

**Dual-track recipes** (asymmetric curricula):

| Track | Goal | Schedule |
|-------|------|----------|
| **A — latent + phase-2 audio polish** | Fix ringing while keeping PCA overlap | Phase 1: latent cycle + latent D. Phase 2: ramp STFT(+RMS) on cycles (`audio_polish_*`) with latent D / `λ_latent_cycle` kept on |
| **B — waveform hybrid latent D** | Pull waveform runs onto Enc manifolds | No extra phase: after cycle warmup, `gan_factor` ramps **audio D and latent D together** (`λ_latent_gan > 0`) |

```bash
# waveform cycle + audio D (default)
# (no override)

# Track B — waveform + hybrid latent D (manifold critic)
OVERRIDE='CycleGANTrainer.lambda_latent_gan=1.0'

# Track A — latent cycle + latent D, then audio polish
OVERRIDE='CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.audio_polish_start_step=100000'
```

SLURM: `CYCLE_DOMAIN=waveform|latent`, `LATENT_CYCLE_MODE=…`, `LAMBDA_LATENT_GAN=1.0`, `AUDIO_POLISH_START=100000`.
---

## B. Loss components

Total warp/G step (terms that are off for a given mode are zero):

```
L = λ_cycle · L_wave
  + λ_latent_cycle · L_z
  + polish_factor · λ_audio_polish · L_wave_polish
  + gan_factor · (λ_gan · L_gan + λ_fm · L_fm)
  + gan_factor · (λ_latent_gan · L_z_gan + λ_latent_fm · L_z_fm)
  + λ_identity · L_id
  + spread_factor · λ_spread · L_spread
```

Each raw term is divided by an empirical scale (startup calibration) before λ. Defaults from [`brave_cyclegan.gin`](../../configs/brave_cyclegan.gin):

| Term | Gin | Default λ | When it is on | What it measures |
|------|-----|-----------|---------------|------------------|
| Waveform cycle `L_wave` | `lambda_cycle` | 10 | `cycle_domain="waveform"` | `x ≈ G_yx(G_xy(x))` and `y ≈ G_xy(G_yx(y))` via STFT (`cycle_stft_weight=0.9`) + RMS (`cycle_rms_weight=0.1`) |
| Latent cycle `L_z` | `lambda_latent_cycle` | 10 | `cycle_domain="latent"` | L1 on `z` — AE-aware or direct (section C) |
| Audio polish `L_wave_polish` | `lambda_audio_polish` | 10 | latent domain, after `audio_polish_start_step` | Same STFT+RMS cycle as `L_wave`; latent D / `L_z` stay on (Track A phase 2) |
| GAN `L_gan` | `lambda_gan` | 1 | after warmup, `gan_factor>0` | Hinge: fool primary `D_Y` / `D_X`. **Audio D:** real `y` vs `Dec_Y(…)`. **Latent D:** real `Enc_Y(y)` vs `W_xy(Enc_X(x))` |
| Latent GAN (hybrid) `L_z_gan` | `lambda_latent_gan` | **0** (off) | waveform domain + `λ>0`, with `gan_factor` | Extra latent D on `z` while audio D stays primary (Track B) |
| Feature matching `L_fm` | `lambda_feature_matching` | 0.5 | with GAN | Match D intermediate features (real vs fake) |
| Latent FM (hybrid) | `lambda_latent_feature_matching` | 0.5 | with hybrid latent GAN | FM on latent D features |
| Identity `L_id` | `lambda_identity` | **0** (off) | only if you set λ>0 | See below. Not used at inference. |
| Latent spread `L_spread` | `lambda_latent_spread` | 0.25 | latent warps, after GAN ramp | Per-channel variance only — **not** manifold membership |

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
| `"identity"` | **Stage-1** and **Approach 3 joint** CycleGAN | residual \(L(z)=z+\sigma(\alpha)(f(z)-z)\) | \(L(z)=z\) exactly (1×1 = I, or last conv zero) |
| `"random"` | **Approach 2 separate** CycleGAN latent warps | \(L(z)=f(z)\) (no residual gate) | Orthogonal 1×1 or Kaiming MLP; output \(z\) ~unit variance |

Identity is the right prior for **within-space** maps (Stage-1; joint CycleGAN). Cross-space separate CycleGAN (\(Z_X \not\approx Z_Y\) as coordinates) uses random; `train_cyclegan.py` forces `init_mode="random"` when `shared_backbone=False`, and honors gin `LatentCanonicalizer.init_mode` (default `"identity"`) when joint.

Manifest field `init_mode` is required at load so inference uses the same forward (residual vs not). Manifest also stores `geometry` (`separate`|`joint`) and `shared_backbone`.

---

## Recipes

From BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# A — waveform cycle + audio D (Approach 2 separate)
python RAVE/scripts/train_cyclegan.py \
  --config configs/brave_cyclegan_separate.gin \
  --backbone_x_config configs/brave.gin \
  --ckpt_x /path/to/tap_run.ckpt --db_path_x /path/to/tap_lmdb \
  --backbone_y_config configs/brave.gin \
  --ckpt_y /path/to/water_run.ckpt --db_path_y /path/to/water_lmdb \
  --canonicalizer_type latent \
  --name tap_water_wf_cycle

# A2 — waveform + hybrid latent D (Track B)
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.lambda_latent_gan=1.0' \
  --name tap_water_wf_hybrid

# B — latent D + AE-aware z cycle
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent"' \
  --name tap_water_lat_ae

# B2 — latent AE-aware + phase-2 audio polish (Track A)
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.audio_polish_start_step=100000' \
  --name tap_water_lat_polish

# C — latent D + direct compose-L1
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.latent_cycle_mode="direct"' \
  --name tap_water_lat_direct
```

SLURM (`scripts/train_cyclegan.sbatch`, 1 GPU): set `CKPT_X`, `DB_PATH_X`, `CKPT_Y`, `DB_PATH_Y`, `RUN_NAME`; optional `CYCLE_DOMAIN`, `LATENT_CYCLE_MODE`, `LAMBDA_LATENT_GAN`, `AUDIO_POLISH_START`, `OVERRIDE`. `CANONICALIZER_TYPE` defaults to `latent`.

## Training schedule

1. **Cycle warmup** (`cycle_warmup_duration`, default 50k; **0** for direct): cycle only, no D
2. **GAN ramp** (`gan_ramp_duration`, default 5k): `gan_factor` 0→1 (spread + hybrid latent D ramp with GAN)
3. **Full**: cycle + hinge GAN + FM (+ spread / hybrid latent D)
4. **Track A only — audio polish** (after `audio_polish_start_step`): ramp STFT+RMS cycle weight while latent cycle + latent D stay active

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

## Silence / ringing diagnosis

Older runs rejected train crops quieter than **−40 dBFS**, so warps never saw `Enc(silence)`. Live tap is sparse. Current `brave.gin` **keeps** quiet crops by default (`maybe_reject_silent.enabled = False`); re-enable with `--reject_silent`. Offline (Python, not Max) you can test whether `Enc_X(0) → W_xy → Dec_Y` rings:

```bash
export PYTHONPATH="${PWD}/RAVE:${PWD}/latent_exploration:${PYTHONPATH}"
python scripts/diagnose_cyclegan_silence.py \
  --tap-backbone /path/to/tap/epoch_1500000.ckpt \
  --y-backbone /path/to/y/epoch_1500000.ckpt \
  --cyclegan runs/tap_<domain>_.../cyclegan_latent.ckpt \
  --output-dir /tmp/cyclegan_silence_diag \
  --input /path/to/tap.wav \
  --y-db-path /path/to/y/preprocessed
```

Do **not** peak-normalize this path. Each probe writes `probe_in.wav`, `transfer.wav`, `control_y_ae.wav`, `gated_mute.wav`, `gated_zy0.wav`, `gated_sidechain.wav`, `rms.png`. `metrics.json` includes `verdict`:

| Flag | Meaning |
|------|---------|
| `hypothesis_supported` | Silent tap codes leak energy; Y-AE / `Dec_Y(0)` are quieter; a loudness gate kills **late** gaps without flattening onsets |
| `likely_pad_cache_not_silent_latent` | Full-context zeros stay quiet but 512-block transfer is louder → Max/`cached_conv` pads, not silent latents |
| `loud_regions_also_ring` | Gap energy is almost as loud as onsets → off-manifold / Track A ringing, not silence-only |

Gated stems are a **test intervention**, not a production gate. `gated_sidechain` scales decode by a smoothed input RMS envelope (`rms / (rms + knee)` at −40 dBFS) instead of hard mute.

## Export for Max / nn~

X→Y timbre transfer only (`Enc_X → W_xy → Dec_Y`, or shared Enc/Dec when joint). Uses `cyclegan_latent.ckpt`, **not** Lightning `last.ckpt`.

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python scripts/export_model.py \
  --model runs/tap_rain_sounds_wf_cyc_mlp2_<hash> \
  --output_dir exports/tap_rain_sounds_cyc

# Joint (Approach 3) — same command; manifest geometry=joint loads the AE once
python scripts/export_model.py \
  --model runs/tap_water_joint_wf_<hash> \
  --output_dir exports/tap_water_joint_wf
```

Writes:

```
exports/tap_rain_sounds_cyc/
  model.ts
  cyclegan_latent.manifest.json
  play.maxpat
```

Copy the folder to `~/Documents/Max 9/Packages/nn_tilde/models/`, open `play.maxpat`, set Max audio to **44100 Hz**. Same nn~ setup as [Fader Max bundles](../../RAVE/rave/fader/export/README.md#max-9--nn-bundles).

`forward` is live tap (or file) in → Y-domain audio out. **Loudness sidechain is on by default** (`set sidechain 1`): hop RMS of the input ducks the output with `g = rms / (rms + knee)` at −40 dBFS, causal across 512-sample nn~ blocks. Toggle in `play.maxpat`, or `set sidechain 0` on the nn~ inlet. Re-export after pulling this so the `.ts` includes the attribute.

Offline listen: `scripts/diagnose_cyclegan_silence.py` still writes `gated_sidechain.wav` for qualitative A/B.

Do not pass `--canonicalizer`; Stage-1 attach is a different graph (same encoder and decoder).

Export zeros causal `cached_conv` pad/cache buffers after the warmup pass before writing `model.ts`. If those rings are left dirty, Max buzzes on load and on silence even when offline block recon of real audio sounds fine. Re-export after pulling that fix.

## W&B metrics

| Key | Meaning |
|-----|---------|
| `cycle/gan_factor` | 0 during warmup, ramps to 1 |
| `cycle/audio_polish_factor` | 0 until `audio_polish_start_step`, then ramps to 1 (Track A) |
| `cycle/cycle_x_norm`, `cycle/cycle_y_norm` | Normalized waveform cycle (waveform domain or polish) |
| `cycle/latent_cycle_norm` | Normalized z cycle (AE-aware or direct) |
| `cycle/adv_norm`, `cycle/fm_norm` | Normalized primary GAN / FM |
| `cycle/latent_adv_norm`, `cycle/latent_fm_norm` | Hybrid latent D / FM (Track B) |
| `cycle/spread_norm` | Normalized spread |
| `val/cycle_x`, `val/cycle_y` | Val waveform cycle |
| `val/latent_cycle_x`, `val/latent_cycle_y` | Val z cycle |
| `val/disc_x_fake`, `val/disc_y_fake` | Fake logits (audio or primary latent D) |
| `val/disc_latent_x_fake`, `val/disc_latent_y_fake` | Hybrid latent D fake logits (Track B) |
| `val/audio_x`, `val/audio_y` | `input \| transfer \| cycle` |
| `val/audio_x_to_y`, `val/audio_y_to_x` | Transfer only (full-clip) |
| `val/audio_x_to_y_nn512` | X→Y transfer in 512-sample blocks (Max/nn~-like); only this direction |
| `val/latent_x_pca` | `Enc_X(x)` vs `z_yx` |
| `val/latent_y_pca` | `Enc_Y(y)` vs `z_xy` |

## Other gin overrides

```bash
OVERRIDE='LatentCanonicalizer.n_layers=2'
OVERRIDE='CycleGANTrainer.cycle_warmup_duration=0'
OVERRIDE='CycleGANTrainer.lambda_latent_gan=1.0'
OVERRIDE='CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.audio_polish_start_step=100000'
OVERRIDE='CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'
```

Legacy `use_waveform_cycle` / `use_latent_cycle` still apply if `cycle_domain` is unset (audio D always).
