# Latent canonicalizer

The latent canonicalizer (`LatentCanonicalizer`) is a lightweight warp applied to the **content latent** `z` **after** the frozen encoder (and variational reparameterization), **before** decode.

Source: [`RAVE/rave/canonicalizer/latent_canonicalizer.py`](../../RAVE/rave/canonicalizer/latent_canonicalizer.py)

## Signal flow

```
x (B, C, T)
  → [optional waveform canonicalizer]
  → PQMF → Enc → reparametrize → z (B, latent_size, T_lat)
  → LatentCanonicalizer L(z)
  → Dec → y
```

Training Stage-1 with `--canonicalizer_type latent` optimizes only `L`; the backbone encoder and decoder stay frozen (encoder may be unfrozen via gin if desired).

## Architecture

`L` is a **gated residual** 1×1 convolution:

```
L(z) = z + σ(α) · (Conv1d(z) − z)
```

| Parameter | Shape | Role |
|-----------|-------|------|
| `conv` | `(latent_size, latent_size, 1)` | Linear mix across latent channels per time step |
| `α` | scalar | Gate in `[0, 1]` via `sigmoid`; starts at 0 |

### Identity initialization

At init:

- `conv.weight` is **identity** (ones on the diagonal, zeros elsewhere)
- `conv.bias` is zero
- `α = 0` → `sigmoid(α) = 0.5`, but `conv(z) = z`, so `L(z) = z` exactly

## Gin configuration

```gin
rave.canonicalizer.latent_canonicalizer.LatentCanonicalizer:
    latent_size = %LATENT_SIZE   # 128 for BRAVE
```

`latent_size` must match the backbone (`RAVE.latent_size` / FaderRAVE content latent dim).

## Training

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/train_canonicalizer.py \
  --config configs/brave_canonicalizer.gin \
  --backbone_config configs/brave.gin \
  --ckpt runs/brave.ckpt \
  --db_path /path/to/in_domain_lmdb \
  --ood_db_path /path/to/tap_lmdb \
  --canonicalizer_type latent \
  --name latent_canon
```

Same mixed in-domain / OOD dataset and discriminator as the waveform path. Only the warp module and checkpoint name differ.

### Trainer notes

- `encode_use_mean = True` (default): encode OOD clips with the posterior **mean** for stabler GAN targets
- `unfreeze_encoder = False` (default): only `L` (and discriminator) train; set `True` + `encoder_lr` for joint encoder adaptation
- Reconstruction modes: `recon_ood_mode` / `recon_in_domain_mode` ∈ `{stft, rms, both}`

## Integration on the backbone

The warp attaches to the `latent_canonicalizer` slot on `RAVE` / `FaderRAVE`:

```python
# encode_with_warp (simplified)
z = backbone.encode(x)
z, reg = backbone.encoder.reparametrize(z)[:2]
if latent_canonicalizer is not None:
    z = latent_canonicalizer(z)
y = backbone.decode(z)
```

See [`RAVE/rave/model.py`](../../RAVE/rave/model.py) (`encode_with_warp`) and [`RAVE/rave/fader/model.py`](../../RAVE/rave/fader/model.py) for Fader attribute handling on decode.

## Export

**Checkpoint file:** `latent_canonicalizer.ckpt` + `latent_canonicalizer.manifest.json`

```bash
python scripts/export_model.py \
  --model runs/latent_canon_abc123 \
  --db_path /path/to/lmdb \
  --output_dir exports/latent_canon
```

`--canonicalizer auto` resolves `latent_canonicalizer.ckpt` in the run directory. The warp is loaded via [`attach_canonicalizer_for_export`](../../RAVE/rave/canonicalizer/export/load.py) before TorchScript tracing.

Unlike the waveform path, there is **no per-block knob smoothing** — `L` operates on latent frames already computed from each audio block.

## Combining with waveform canonicalizer

A backbone can hold **both** slots (`waveform_canonicalizer` and `latent_canonicalizer`), but each Stage-1 run trains **one** warp type. To use both:

1. Train waveform Stage-1 → `waveform_canonicalizer.ckpt`
2. Train latent Stage-1 on the same backbone (or load waveform ckpt manually) → `latent_canonicalizer.ckpt`
3. Export with both checkpoints specified:

```bash
python scripts/export_model.py \
  --model runs/combined \
  --waveform_canonicalizer runs/wf_canon/waveform_canonicalizer.ckpt \
  --latent_canonicalizer runs/latent_canon/latent_canonicalizer.ckpt \
  ...
```

(Only one canonicalizer is auto-resolved; explicit paths are required for both.)

## When to choose latent

- OOD gap is **not** well captured by EQ + short reverb (e.g. complex excitation / playing style)
- You want a **compact** warp (~`latent_size²` params in the 1×1 conv plus one gate)
- You do not need interpretable DSP knobs

Prefer the [waveform canonicalizer](waveform.md) when corrections should stay in an explicit audio-effects chain or when you want per-clip knob telemetry.

## Parameter count (BRAVE default)

With `latent_size = 128`:

- `conv`: 128 × 128 + 128 bias ≈ **16.5k** parameters
- `α`: 1 parameter

Total warp is small relative to the backbone (~4.9M params for BRAVE).

## Related files

| File | Role |
|------|------|
| [`latent_canonicalizer.py`](../../RAVE/rave/canonicalizer/latent_canonicalizer.py) | `LatentCanonicalizer` module |
| [`trainer.py`](../../RAVE/rave/canonicalizer/trainer.py) | Shared Stage-1 training loop |
| [`backbone.py`](../../RAVE/rave/canonicalizer/backbone.py) | Attach warp at export |
| [`config.py`](../../RAVE/rave/canonicalizer/config.py) | Manifest save/load |
