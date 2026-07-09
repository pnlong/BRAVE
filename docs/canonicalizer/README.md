# Canonicalizer documentation

Stage-1 **input adapters** for frozen BRAVE / FaderRAVE backbones. A canonicalizer is a small trainable warp that runs **before** (waveform) or **after** (latent) the frozen encoder, so out-of-domain audio (e.g. body-tap recordings) reconstructs with timbre closer to the in-domain corpus the backbone was trained on — without retraining the autoencoder.

Implementation lives in [`RAVE/rave/canonicalizer/`](../../RAVE/rave/canonicalizer/). Training config: [`configs/brave_canonicalizer.gin`](../../configs/brave_canonicalizer.gin).

## Which canonicalizer?

| | [Waveform](waveform.md) | [Latent](latent.md) |
|---|-------------------------|---------------------|
| **Where it acts** | On raw audio, before PQMF encode | On content latent `z`, after encode |
| **Mechanism** | Per-input knob encoder → causal EQ + optional reverb | Gated residual 1×1 conv on `z` |
| **Interpretability** | Explicit DSP knobs (EQ bands, reverb wet/comb/allpass) | Learned linear mix in latent space |
| **Typical use** | Timbre/room correction you can reason about in Hz and wet/dry | Broader latent remapping when waveform DSP is too constrained |
| **Realtime caveat** | Per-block knob estimates; use EMA smoothing in export | Stateless per latent frame; no knob smoothing needed |
| **Checkpoint** | `waveform_canonicalizer.ckpt` | `latent_canonicalizer.ckpt` |

You train **one** type per run (`--canonicalizer_type waveform` or `latent`). They can be composed in principle (waveform slot + latent slot on the same backbone), but each Stage-1 run produces a single warp checkpoint.

## Problem framing

| Term | Meaning |
|------|---------|
| **In-domain (Y)** | LMDB the backbone was trained on |
| **Out-of-domain (X)** | OOD LMDB (e.g. tap on a surface) |
| **G** | Canonicalizer + frozen Enc/Dec → reconstructed audio |
| **D** | `InDomainAudioDiscriminator` — real Y vs OOD reconstructions |

Loss details: [loss.md](loss.md).

## Shared training pipeline

From the BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

# Waveform canonicalizer (plain BRAVE backbone)
python RAVE/scripts/train_canonicalizer.py \
  --config configs/brave_canonicalizer.gin \
  --backbone_config configs/brave.gin \
  --ckpt runs/brave.ckpt \
  --db_path /path/to/in_domain_lmdb \
  --ood_db_path /path/to/tap_lmdb \
  --canonicalizer_type waveform \
  --name wf_canon

# Latent canonicalizer (FaderRAVE backbone example)
python RAVE/scripts/train_canonicalizer.py \
  --config configs/brave_canonicalizer.gin \
  --backbone_config configs/brave_fader_birdsong.gin \
  --ckpt runs/birdsong.ckpt \
  --db_path /path/to/birdsong_lmdb \
  --ood_db_path /path/to/tap_lmdb \
  --canonicalizer_type latent \
  --name latent_canon
```

### Loss (both types)

See [loss.md](loss.md) for the full derivation, metric scales, and tuning guide.

Summary:

```
L_total = λ_gan · L_GAN + λ_rec · L_rec + λ_fm · L_feature_matching
```

Default gin ([`brave_canonicalizer.gin`](../../configs/brave_canonicalizer.gin)):

- **L_GAN** (`λ_gan = 1.0`): fool the in-domain audio discriminator on OOD batches
- **L_rec** (`λ_rec = 0.05`): light self-reconstruction (STFT + RMS envelope)
- **Feature matching** (`λ_feature_matching = 2.0`): match discriminator intermediate features

Two optimizers: warp (+ optional unfrozen backbone encoder) vs discriminator.

Works on **plain BRAVE** (`num_attributes=0`) and **FaderRAVE** (decode uses batch attributes when present).

### Outputs

Each run writes under `runs/<name>_<gin_hash>/`:

```
waveform_canonicalizer.ckpt   # or latent_canonicalizer.ckpt
waveform_canonicalizer.manifest.json
config.gin
```

The manifest records backbone config, backbone ckpt, db paths, and `backbone_kind` (`RAVE` or `FaderRAVE`).

## Export

Canonicalizer weights embed into realtime bundles via [`scripts/export_model.py`](../../scripts/export_model.py):

```bash
python scripts/export_model.py \
  --model runs/wf_canon_abc123 \
  --db_path /path/to/lmdb \
  --output_dir exports/wf_canon
```

- `--canonicalizer auto` (default): picks `*canonicalizer.ckpt` from the run directory
- `--canonicalizer none`: backbone only
- `--waveform_canonicalizer` / `--latent_canonicalizer`: explicit ckpt path

See [waveform export notes](waveform.md#realtime-export-and-knob-ema) for streaming-specific behavior.

## Further reading

- [Training loss](loss.md) — STFT scale, λ weights, warmup phases, WandB metrics, tuning
- [Waveform canonicalizer](waveform.md) — knob encoder, DSP layout, gin knobs
- [Latent canonicalizer](latent.md) — residual warp, identity init, when to prefer latent
- Package README: [`RAVE/rave/canonicalizer/README.md`](../../RAVE/rave/canonicalizer/README.md)
