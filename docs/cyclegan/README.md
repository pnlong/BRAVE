# CycleGAN: Tap ↔ Water

Bidirectional CycleGAN between two **frozen** plain BRAVE backbones (domain X = tap, domain Y = water). Trains lightweight warp modules in **latent** or **waveform** space with hinge GAN discriminators and cycle consistency.

Implementation: [`RAVE/rave/canonicalizer/cycle_trainer.py`](../../RAVE/rave/canonicalizer/cycle_trainer.py).  
Config: [`configs/brave_cyclegan.gin`](../../configs/brave_cyclegan.gin).

## Objective

| Term | Meaning |
|------|---------|
| `G_{X→Y}` | `warp_xy` + frozen Enc/Dec (tap → water) |
| `G_{Y→X}` | `warp_yx` + frozen Enc/Dec (water → tap) |
| `D_Y` | Real water vs `G_{X→Y}(x)` |
| `D_X` | Real tap vs `G_{Y→X}(y)` |
| Waveform cycle | `x ≈ G_{Y→X}(G_{X→Y}(x))` via STFT + RMS (`use_waveform_cycle`) |
| Latent cycle | `z_x ≈ W_{yx}(W_{xy}(z_x))` L1, no re-encode (`use_latent_cycle`; latent warps only) |

Relation to one-way canonicalizer: see [`docs/canonicalizer/loss.md`](../canonicalizer/loss.md) CycleGAN mapping — canonicalizer is **one-way** (X→Y + `D_Y` only).

Enable latent cycle (optionally drop waveform cycle):

```bash
OVERRIDE='CycleGANTrainer.use_latent_cycle=True'
# latent-only cycle:
# OVERRIDE='CycleGANTrainer.use_latent_cycle=True CycleGANTrainer.use_waveform_cycle=False'
```

2-layer latent warp (Conv → LeakyReLU → Conv; still identity at init):

```bash
OVERRIDE='CycleGANTrainer.use_latent_cycle=True LatentCanonicalizer.n_layers=2'
```

## Training schedule

1. **Cycle warmup** (`cycle_warmup_duration`, default 50k steps): cycle loss only; **no discriminator steps**
2. **GAN ramp** (`gan_ramp_duration`, default 5k): `gan_factor` 0→1
3. **Full CycleGAN**: cycle + hinge GAN + feature matching

## Commands

From BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# Latent CycleGAN
python RAVE/scripts/train_cyclegan.py \
  --config configs/brave_cyclegan.gin \
  --backbone_x_config configs/brave.gin \
  --ckpt_x /path/to/tap_run.ckpt \
  --db_path_x /path/to/tap_lmdb \
  --backbone_y_config configs/brave.gin \
  --ckpt_y /path/to/water_run.ckpt \
  --db_path_y /path/to/water_lmdb \
  --canonicalizer_type latent \
  --name tap_water_lat_cyclegan

# Waveform CycleGAN
python RAVE/scripts/train_cyclegan.py \
  ... \
  --canonicalizer_type waveform \
  --name tap_water_wf_cyclegan
```

## Outputs

Run directory `runs/<name>_<hash>/`:

```
cyclegan_latent.ckpt      # or cyclegan_waveform.ckpt
cyclegan_latent.manifest.json
config.gin
loss_scales.json
viz/x_val_step*.wav        # input | G_xy(x) | x_cycle
viz/y_val_step*.wav        # input | G_yx(y) | y_cycle
viz/x_to_y_val_step*.wav   # G_xy(x) only
viz/y_to_x_val_step*.wav   # G_yx(y) only
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
| `cycle/cycle_x_norm`, `cycle/cycle_y_norm` | Normalized waveform cycle losses |
| `cycle/latent_cycle_norm` | Normalized latent cycle (when enabled) |
| `val/cycle_x`, `val/cycle_y` | Validation waveform cycle |
| `val/latent_cycle_x`, `val/latent_cycle_y` | Validation latent cycle |
| `val/disc_x_fake`, `val/disc_y_fake` | Fake logits after GAN phase |
| `val/audio_x`, `val/audio_y` | `input \| transfer \| cycle` |
| `val/audio_x_to_y`, `val/audio_y_to_x` | Transfer only |
| `val/latent_x_pca` | X-space: Enc_X(x) vs G_yx(y), colored by domain |
| `val/latent_y_pca` | Y-space: Enc_Y(y) vs G_xy(x), colored by domain |

## Ablations (gin overrides)

```bash
# Unfreeze decoders only (recommended first unfreeze)
OVERRIDE='CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'

# Unfreeze encoders + decoders
OVERRIDE='CycleGANTrainer.unfreeze_encoders=True CycleGANTrainer.unfreeze_decoders=True CycleGANTrainer.backbone_lr=1e-5 CycleGANTrainer.lambda_identity=1.0'
```

**Identity loss:** `λ_id · (‖G_xy(y)−y‖ + ‖G_yx(x)−x‖)`. Encourages each mapper to act as identity on samples already in its *output* domain, so GAN pressure does not needlessly rewrite in-domain audio. Especially useful when unfreezing Enc/Dec.
