# Canonicalizer (`rave.canonicalizer`)

Stage-1 **input adapter** for frozen RAVE / FaderRAVE backbones. Trains a small warp so
**out-of-domain** audio (e.g. tap) reconstructs with timbre closer to the **in-domain** corpus
the backbone was trained on — without retraining the autoencoder.

Loss derivation and CycleGAN mapping: [`scratchpaper/canonicalizer_loss.md`](../../../scratchpaper/canonicalizer_loss.md).

## Problem framing

| Term | In this repo |
|------|----------------|
| **In-domain (Y)** | LMDB the backbone was trained on (`DOMAIN_IN`) |
| **Out-of-domain (X)** | OOD LMDB, e.g. tap (`DOMAIN_OOD`) |
| **G** | Canonicalizer + frozen Enc/Dec → reconstructed audio |
| **D** | `InDomainAudioDiscriminator` — real Y audio vs OOD reconstructions |

## Package layout

| File / dir | Role |
|------------|------|
| [`latent_canonicalizer.py`](latent_canonicalizer.py) | `LatentCanonicalizer` — residual 1×1 conv on content `z` after encode |
| [`waveform_canonicalizer.py`](waveform_canonicalizer.py) | `WaveformCanonicalizer` — EQ + optional causal reverb before encode |
| [`in_domain_discriminator.py`](in_domain_discriminator.py) | `InDomainAudioDiscriminator` — multi-scale audio GAN (real in-domain vs OOD fake) |
| [`trainer.py`](trainer.py) | `CanonicalizerTrainer` — Lightning Stage-1 loop |
| [`losses.py`](losses.py) | RMS recon helpers, GAN loss resolver |
| [`dataset.py`](dataset.py) | Mixed in-domain / OOD datasets, IR aug, collate |
| [`config.py`](config.py) | `TrainingProfile`, manifest, checkpoint save/load |
| [`backbone.py`](backbone.py) | Attach warp weights to `RAVE` or `FaderRAVE` |
| [`callbacks.py`](callbacks.py) | Validation PCA/t-SNE + W&B audio |
| [`viz.py`](viz.py) | Scatter plots and audio triplet helpers |
| [`ir_augmentation.py`](ir_augmentation.py) | Optional room IR on OOD clips |
| [`export/`](export/) | Ckpt resolve + attach for nn~ / TorchScript export |

## Warp modules

### Waveform (`canonicalizer_type=waveform`)

```
x → C(x) [EQ + reverb] → Enc → Dec → y
```

`C` is identity at init. Applied on the backbone via `waveform_canonicalizer` slot
(see [`backbone.py`](backbone.py), [`RAVE/rave/model.py`](../model.py)).

### Latent (`canonicalizer_type=latent`)

```
x → Enc → L(z) → Dec → y
```

`L` is a gated residual 1×1 conv, identity at init. Applied via `latent_canonicalizer` slot.

## Training loss

```
L_total = λ_gan · L_GAN + λ_rec · L_rec
```

- **L_GAN** (OOD only): fool `InDomainAudioDiscriminator` — push `G(x)` toward in-domain audio statistics
- **L_rec** (optional, `λ_rec=0` disables): self-reconstruction `G(x) ≈ x` (STFT and/or RMS envelope)

Two optimizers: warp (+ optional unfrozen encoder) vs `InDomainAudioDiscriminator`.

Works on **plain BRAVE** (`num_attributes=0`) and **FaderRAVE** (conditional decode uses batch attrs when present).

## Train

From BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# Plain BRAVE backbone
python RAVE/scripts/train_canonicalizer.py \
  --config configs/brave_canonicalizer.gin \
  --backbone_config configs/brave.gin \
  --ckpt runs/brave.ckpt \
  --db_path /path/to/in_domain_lmdb \
  --ood_db_path /path/to/tap_lmdb \
  --canonicalizer_type waveform \
  --name canon_run

# FaderRAVE backbone
python RAVE/scripts/train_canonicalizer.py \
  --backbone_config configs/brave_fader_birdsong.gin \
  ...
```

Writes `waveform_canonicalizer.ckpt` or `latent_canonicalizer.ckpt` plus `.manifest.json` sidecar.

Key gin: [`configs/brave_canonicalizer.gin`](../../../configs/brave_canonicalizer.gin).

## Export

Canonicalizer checkpoints embed into realtime bundles through [`export/`](export/):

- `resolve_canonicalizer_ckpt` — find `*canonicalizer.ckpt` in run dir or accept explicit path
- `attach_canonicalizer_for_export` — load warp onto backbone before TorchScript

Used by [`scripts/export_model.py`](../../../scripts/export_model.py) for both vanilla RAVE and FaderRAVE.
Fader-specific export UI (stats, host controls, `play.maxpat`) stays in [`rave/fader/export/`](../fader/export/).

## Checkpoints

```
waveform_canonicalizer.ckpt
waveform_canonicalizer.manifest.json
latent_canonicalizer.ckpt
latent_canonicalizer.manifest.json
```

Manifest records backbone config, ckpt, db paths, and `backbone_kind` (`RAVE` or `FaderRAVE`).
