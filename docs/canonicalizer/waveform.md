# Waveform canonicalizer

The waveform canonicalizer (`WaveformCanonicalizer`) is a **per-input** adapter applied to raw audio **before** PQMF encode. Each clip is mapped to a **K-dimensional knob vector** that drives differentiable causal EQ and optional reverb, then the result is fed to the frozen RAVE encoder.

Source: [`RAVE/rave/canonicalizer/waveform_canonicalizer.py`](../../RAVE/rave/canonicalizer/waveform_canonicalizer.py)

## Signal flow

```
x (B, C, T)
  → WaveformKnobEncoder
  → knobs (B, K)
  → split into eq_knobs (B, n_eq), rev_knobs (B, n_rev)
  → BiquadBank(x, eq_knobs)
  → CausalReverb(x, rev_knobs)   [optional]
  → canonicalized audio
  → PQMF → Enc → Dec → y
```

At initialization the encoder outputs **neutral knobs**, so `C(x) ≈ x` (identity pass-through). Training then learns input-dependent corrections that push OOD reconstructions toward in-domain timbre.

## Components

### `WaveformKnobEncoder`

Small **causal** conv stack (left-padded `Conv1d` + stride-2 downsampling) that pools over time and projects to `K` scalars.

| Gin parameter | Default | Role |
|---------------|---------|------|
| `hidden_channels` | 64 | Conv channel width |
| `n_layers` | 4 | Number of causal conv blocks |
| `max_gain_db` | 12.0 | EQ gain range via `tanh(raw) * max_gain_db` |
| `in_channels` | 1 | Match backbone `n_channels` (set at build time) |

**Range mapping** (applied in the encoder head):

- **EQ slots**: `gain_db = tanh(raw) * max_gain_db` (neutral = 0 dB)
- **Reverb wet** (slot 0 of reverb section): raw logit → `sigmoid` in `CausalReverb` (neutral bias ≈ −20 → dry)
- **Comb feedback / allpass gain**: raw pre-sigmoid values (same activations as legacy internal params)

### `BiquadBank` (EQ)

Causal peaking biquads at log-spaced center frequencies between `min_freq` and `max_freq`.

| Gin parameter | Default |
|---------------|---------|
| `n_bands` | 6 |
| `min_freq` | 80 Hz |
| `max_freq` | 12000 Hz |
| `max_gain_db` | 12 dB |
| `q` | 1.0 |

Each band applies a soft bypass: at 0 dB gain the filter is an exact pass-through but gradients still flow.

### `CausalReverb`

Schroeder-style **causal** wet/dry chain: parallel comb filters → averaged → allpass chain → wet mix.

Default structure (7 reverb knobs):

| Index | Knob | Activation in DSP |
|------:|------|-------------------|
| 0 | wet | `sigmoid(wet_logit)` |
| 1–4 | comb feedback | `sigmoid(raw) * 0.85` |
| 5–6 | allpass gain | `sigmoid(raw) * 0.7` |

Comb delay times (ms): 29.7, 37.1, 41.1, 43.7. Allpass delays: 5.0, 1.7.

## Knob layout (`WaveformKnobLayout`)

The full knob vector order is:

```
[eq_gain_0, …, eq_gain_{n_eq-1}, reverb_0, …, reverb_{n_rev-1}]
```

**Default K = 13** when `n_eq_bands = 6` and `n_reverb_knobs = 7`.

To change K, keep these gin bindings **consistent**:

```gin
rave.canonicalizer.waveform_canonicalizer.WaveformKnobLayout:
    n_eq_bands = 6      # must match dsp.BiquadBank.n_bands
    n_reverb_knobs = 7  # must match CausalReverb.n_knobs (1 + n_combs + n_allpasses)

dsp.BiquadBank:
    n_bands = 6
```

Example alternate layout: 15 EQ bands + 4 reverb scalars → **K = 19** (requires matching `CausalReverb` comb/allpass count or a custom reverb module).

### API

```python
knobs = canonicalizer.predict_knobs(x)   # (B, K)
eq_k, rev_k = canonicalizer.layout.split(knobs)
y = canonicalizer(x)
```

## Training

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/train_canonicalizer.py \
  --config configs/brave_canonicalizer.gin \
  --backbone_config configs/brave.gin \
  --ckpt runs/brave.ckpt \
  --db_path /path/to/in_domain_lmdb \
  --ood_db_path /path/to/tap_lmdb \
  --canonicalizer_type waveform \
  --name wf_canon
```

Key gin overrides ([`configs/brave_canonicalizer.gin`](../../configs/brave_canonicalizer.gin)):

```gin
rave.canonicalizer.waveform_canonicalizer.WaveformCanonicalizer:
    use_reverb = True
    knob_ema_decay = 0.95

rave.canonicalizer.trainer.CanonicalizerTrainer:
    lr = 3e-4
    lambda_gan = 1.0
    lambda_rec = 0.05
```

Set `use_reverb = False` for EQ-only canonicalization (`n_reverb_knobs = 0` in layout).

### Optional IR augmentation on OOD clips

```gin
rave.canonicalizer.dataset.make_ir_augment:
    ir_path = "/path/to/irs"
    ir_prob = 0.3
```

Or pass `--ir_path` / `--ir_prob` to `train_canonicalizer.py`.

## Realtime export and knob EMA

TorchScript / nn~ export calls `waveform_canonicalizer(x)` **once per streaming block**. With per-input knobs, estimates can change block-to-block and cause audible modulation.

**Mitigation:** `knob_ema_decay` (default `0.95` in gin) enables exponential smoothing of knob vectors during **eval/export** (`model.eval()`). Training uses raw per-clip knobs (no EMA).

```gin
rave.canonicalizer.waveform_canonicalizer.WaveformCanonicalizer:
    knob_ema_decay = 0.95   # set null / omit binding for no smoothing
```

Disable streaming cached conv for offline-only TorchScript:

```bash
python scripts/export_model.py --model runs/wf_canon --host ts --nostreaming ...
```

## Checkpoints and compatibility

**Checkpoint file:** `waveform_canonicalizer.ckpt` + `waveform_canonicalizer.manifest.json`

State dict includes `encoder.*`, `eq.*`, and `reverb.*` (DSP internal params exist for legacy fallback but are **not** the primary learnable path).

**Breaking change:** Checkpoints from the older **global-DSP** design (learnable scalars only inside `BiquadBank` / `CausalReverb`, no `encoder`) are **not compatible**. Retrain Stage-1 after upgrading to the per-input encoder architecture.

## When to choose waveform

- You want **interpretable** timbre correction (EQ curve + room wetness)
- OOD error is mostly **spectral balance / damping / short room coloration**
- You plan to inspect or log `predict_knobs(x)` during evaluation

Prefer the [latent canonicalizer](latent.md) when the needed shift is not well modeled by a small EQ+reverb chain.

## Related files

| File | Role |
|------|------|
| [`waveform_canonicalizer.py`](../../RAVE/rave/canonicalizer/waveform_canonicalizer.py) | Encoder + canonicalizer |
| [`biquad_bank.py`](../../RAVE/rave/dsp/biquad_bank.py) | Causal peaking EQ |
| [`causal_reverb.py`](../../RAVE/rave/dsp/causal_reverb.py) | Causal reverb DSP |
| [`backbone.py`](../../RAVE/rave/canonicalizer/backbone.py) | Attach warp at export |
| [`RAVE/rave/model.py`](../../RAVE/rave/model.py) | `waveform_canonicalizer` slot in `encode()` |
