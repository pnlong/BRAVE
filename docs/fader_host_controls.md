# Fader host controls (TorchScript / nn~ inference)

Fader models widen the decoder input to `128 + D` by concatenating **normalized attributes** `attr_norm` with content latent `z`.

## Pipeline

```
audio block → encode → z (B, 128, T_lat)
audio block → extract raw attr (D, T_lat)  [nn~: torch extractors in forward]
manual knobs → broadcast raw values
blend / scale / override
normalize_all → attr_norm (B, D, T_lat)
decode(cat(z, attr_norm)) → audio
```

## Export paths

**Default:** [`scripts/export_model.py`](../scripts/export_model.py) — auto-detects BRAVE vs FaderRAVE, embeds canonicalizer when present, writes a Max bundle with `play.maxpat`.

| Script | Host | Attribute knobs |
|--------|------|-----------------|
| [`scripts/export_model.py`](../scripts/export_model.py) | Max **nn~** bundle | Yes (Fader) — pre-wired patch |
| [`RAVE/scripts/export_fader_nn.py`](../RAVE/scripts/export_fader_nn.py) | Max/Pd **nn~** | Yes — `register_attribute` per control |
| [`RAVE/scripts/export_fader_ts.py`](../RAVE/scripts/export_fader_ts.py) | Python demos | External — pass `attr` to `forward(x, attr)` |
| [`RAVE/scripts/export.py`](../RAVE/scripts/export.py) | nn~ (128-D RAVE only) | No Fader support |

### nn~ export (recommended for Max)

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python scripts/export_model.py \
  --model runs/brave_fader_texture \
  --db_path /path/to/lmdb \
  --output_dir exports/brave_fader_texture
```

Copy the bundle folder to `~/Documents/Max 9/Packages/nn_tilde/models/` and open `play.maxpat`.

## nn~ attributes

Per attribute row (order = `attribute_names` in stats):

| Attribute | Type | Default | Role |
|-----------|------|---------|------|
| `{name}` | float or int | midpoint / 0 | Manual raw value (broadcast over `T_lat`) |
| `{name}_scale` | float | 1.0 | Multiplier on normalized row after `normalize_all` |
| `{name}_override` | bool | false | Replace extracted row with manual `{name}` |
| `attr_mode` | int | 2 | `0`=extract+scale/override, `1`=manual-only, `2`=extract-only |

Max message syntax:

```
set rms 0.05
set rms_scale 1.2
set texture_class 3
set texture_class_override 1
set attr_mode 1
```

Example patch wiring: slider → `prepend set rms` → **same inlet** as audio on `[nn~ model.ts forward 512]`.

## Attribute order

Row order matches `attribute_names` in `attribute_stats.yaml` (continuous rows first, then discrete).

## Continuous sliders

Each continuous name has global min/max from stats. `{name}` is the **raw** training-unit value (not normalized). After `normalize_all`, values lie in **[-1, 1]** per frame; `{name}_scale` multiplies that normalized trajectory.

## Discrete controls (`texture_class`, etc.)

Set `{name}` to class index `0 … K-1`. After precompute, human-readable names are stored in `attribute_stats.yaml` as `discrete_class_labels` (and copied to `*_host_controls.json` at export for Max `live.menu` entries). Discrete rows are always taken from manual menus in `attr_mode=2` (extract-only); in modes `0` and `1` they come from the `{name}` attribute.

## Realtime nn~ extract

TorchScript extractors match **librosa** descriptors (`rms`, `flatness`, `centroid`, …). **Timbral** attrs (`roughness`, `brightness`, …) stay at training midpoints in `attr_mode=2` until dedicated JIT extractors exist — bogus STFT proxies caused out-of-range normalized attrs (buzzing).

Extracted rows are clamped to dataset min/max, normalized to **[-1, 1]**, and only applied after the rolling buffer fills (~3 s @ 44.1 kHz, same as training `n_signal`).

## Realtime demo (plain TorchScript)

```bash
python scripts/realtime_fader_demo.py \
  --ts exports/fader.ts \
  --input clip.wav --output out.wav \
  --attr-scales rms=1.2,centroid=0.9
```

See [`scripts/realtime_fader_demo.py`](../scripts/realtime_fader_demo.py).

## Minifusion / HDF5

Vanilla BRAVE plugin export (`export_brave_plugin.py`) does not include Fader attributes.
