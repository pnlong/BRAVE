# CycleGAN: Tap ↔ Water

Bidirectional CycleGAN between two **frozen** plain BRAVE backbones (domain X = tap, domain Y = water / birdsong). Latent warps are **cross-space**: `Enc_src → W → Dec_tgt`. Waveform warps are **shelved**.

Implementation: [`RAVE/rave/canonicalizer/cycle_trainer.py`](../../RAVE/rave/canonicalizer/cycle_trainer.py).  
Config: [`configs/brave_cyclegan.gin`](../../configs/brave_cyclegan.gin).

## Two axes

| Axis | Switch | Meaning |
|------|--------|---------|
| Warp | `--canonicalizer_type latent` | Cross-space 128-D warp (CycleGAN always **random** init). Stage-1 canonicalizer stays identity-init. |
| Cycle + GAN | `CycleGANTrainer.cycle_domain` | `"waveform"` (default) or `"latent"` — cycle loss **and** discriminator live in that domain |

When `cycle_domain="latent"`, `latent_cycle_mode` selects how z-cycle is measured:

| `latent_cycle_mode` | Cycle | Decode in train? | Warmup |
|---------------------|-------|------------------|--------|
| `"ae_aware"` (default) | \(z_x \approx W_{yx}(\mathrm{Enc}_Y(\mathrm{Dec}_Y(W_{xy}(z_x))))\) | yes | 50k |
| `"direct"` | \(z_x \approx W_{yx}(W_{xy}(z_x))\) | **no** (val only) | **0** |

Inference is always `Enc_src → warp → Dec_tgt` (decode for audio).

## Recipes

From BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# A — waveform cycle + audio D (default)
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

# C — latent D + direct compose-L1 (no warmup, no decode in train)
python RAVE/scripts/train_cyclegan.py ... --canonicalizer_type latent \
  --override 'CycleGANTrainer.cycle_domain="latent" CycleGANTrainer.latent_cycle_mode="direct"' \
  --name tap_water_lat_direct
```

SLURM: `CANONICALIZER_TYPE=latent` (default), optional `CYCLE_DOMAIN` / `LATENT_CYCLE_MODE`, then `sbatch scripts/train_cyclegan.sbatch`.

## Objective (mode A)

| Term | Meaning |
|------|---------|
| `G_{X→Y}` | `Enc_X → warp_xy → Dec_Y` |
| `G_{Y→X}` | `Enc_Y → warp_yx → Dec_X` |
| `D_Y` / `D_X` | Audio MSD: real Y vs `G_xy(x)`, real X vs `G_yx(y)` |
| Waveform cycle | `x ≈ G_yx(G_xy(x))` via STFT + RMS |

## Objective (modes B / C)

| Term | Meaning |
|------|---------|
| `D_Y` / `D_X` | Latent D: real `Enc_Y(y)` vs `W_xy(Enc_X(x))`, and vice versa |
| AE-aware cycle | L1 through the other AE (`P_Y = Enc_Y ∘ Dec_Y` in the middle) |
| Direct cycle | L1 `W_yx(W_xy(z)) ≈ z` (no Dec/Enc in the cycle) |

Val always decodes for listening + PCA (`Enc_Y(y)` vs `z_xy`, `Enc_X(x)` vs `z_yx`).

## Training schedule

1. **Cycle warmup** (`cycle_warmup_duration`, default 50k; **0** for direct): cycle only, no D steps
2. **GAN ramp** (`gan_ramp_duration`, default 5k): `gan_factor` 0→1
3. **Full CycleGAN**: cycle + hinge GAN + feature matching

## Outputs

Run directory `runs/<name>_<hash>/`:

```
cyclegan_latent.ckpt
cyclegan_latent.manifest.json   # includes cycle_domain, latent_cycle_mode, init_mode
config.gin
loss_scales.json
viz/x_val_step*.wav             # input | G_xy(x) | x_cycle
viz/y_val_step*.wav
viz/x_to_y_val_step*.wav
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

## W&B metrics

| Key | Meaning |
|-----|---------|
| `cycle/gan_factor` | 0 during warmup, ramps to 1 |
| `cycle/cycle_x_norm`, `cycle/cycle_y_norm` | Normalized waveform cycle |
| `cycle/latent_cycle_norm` | Normalized z cycle (AE-aware or direct) |
| `val/cycle_x`, `val/cycle_y` | Validation waveform cycle |
| `val/latent_cycle_x`, `val/latent_cycle_y` | Validation z cycle |
| `val/disc_x_fake`, `val/disc_y_fake` | Fake logits (audio or latent D) |
| `val/audio_x`, `val/audio_y` | `input \| transfer \| cycle` |
| `val/audio_x_to_y`, `val/audio_y_to_x` | Transfer only |
| `val/latent_x_pca` | Enc_X(x) vs `z_yx` |
| `val/latent_y_pca` | Enc_Y(y) vs `z_xy` |

## Ablations (gin overrides)

```bash
# Unfreeze decoders only
OVERRIDE='CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'

# Unfreeze encoders + decoders
OVERRIDE='CycleGANTrainer.unfreeze_encoders=True CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'

# 2-layer warp
OVERRIDE='LatentCanonicalizer.n_layers=2'
```

**Identity loss:** waveform STFT when `gan_domain=audio`; z-space \(W_{xy}(\mathrm{Enc}_X(y)) \approx \mathrm{Enc}_Y(y)\) when latent. Default `lambda_identity=0`.

Legacy flags `use_waveform_cycle` / `use_latent_cycle` still apply if `cycle_domain` is unset (audio D always).
