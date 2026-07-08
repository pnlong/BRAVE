# Fader export (`rave.fader.export`)

Stripped inference graph for TorchScript and nn~ — no latent discriminator, no GAN, no training stats mutation.

## Files

| File | Role |
|------|------|
| [`trace_model.py`](trace_model.py) | `FaderTraceModel`, `build_trace_model` |
| [`nn_module.py`](nn_module.py) | `ScriptedFaderRAVE` — nn_tilde wrapper with attribute knobs |
| [`torch_descriptors.py`](torch_descriptors.py) | JIT-safe continuous descriptor extractors |
| [`host_controls.py`](host_controls.py) | `*_host_controls.json` writer |
| [`max_patch.py`](max_patch.py) | Auto-generated `play.maxpat` for Max 9 / nn~ |
| [`bundle.py`](bundle.py) | Bundle sidecars + copy instructions |
| [`load_for_export.py`](load_for_export.py) | FaderRAVE load + stats for export |
| [`__init__.py`](__init__.py) | Package exports |

## Max 9 / nn~ bundles

Exports from [`scripts/export_model.py`](../../../../scripts/export_model.py) write a self-contained folder:

```
exports/<run_name>/
  model.ts                      # nn~ TorchScript model
  model_attribute_stats.yaml    # Fader attribute ranges (FaderRAVE only)
  model_host_controls.json      # Attribute metadata (FaderRAVE only)
  play.maxpat                     # Pre-wired patch — open this in Max
```

### One-time setup (macOS)

1. Install [nn~ v1.6.0](https://github.com/acids-ircam/nn_tilde/releases) → unpack to `~/Documents/Max 9/Packages/`
2. Set Max **Options → Audio** to **44100 Hz**

### Load a model (every time)

1. Copy the bundle folder to `~/Documents/Max 9/Packages/nn_tilde/models/`
2. Open `play.maxpat` from that folder
3. Click **ezdac~** to enable audio

The patch is laid out in **numbered steps** on the right:

1. **INPUT SOURCE** — `live in` vs `file`
2. **LIVE INPUT** — `adc~` + **meter~** (confirms mic is working)
3. **FILE INPUT** — **Choose audio file…** button (reliable), drag-and-drop zone, **play** toggle; drops auto-play
4. **LEVEL & MODEL** — **input gain** slider → `nn~`
5. **OUTPUT** — **ezdac~** speaker (this is Max’s “audio on” — there is no separate mic button)

Model attributes go to nn~ **right inlet**; audio to **left**.

See [`../../../../docs/fader_host_controls.md`](../../../../docs/fader_host_controls.md) for Fader attribute semantics.

## Inference API (`FaderTraceModel`)

```
audio x  → encode → z (B, 128, T_lat)
attr raw → normalize_all → attr_norm (B, D, T_lat)
y = decode(cat(z, attr_norm))
```

Buffers embed per-attribute min/max for continuous rows, discrete class counts, and (after precompute) **`discrete_class_labels`** in `attribute_stats.yaml` for Max menu text.

## CLI

### Unified router (recommended)

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python scripts/export_model.py \
  --model runs/brave_fader_run \
  --db_path /path/to/lmdb \
  --output_dir exports/brave_fader_run
```

Writes a bundle: `model.ts`, sidecars, and pre-wired `play.maxpat` (see **Max 9 / nn~ bundles** above).

### nn~ (Max / Pd) — attribute knobs

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python RAVE/scripts/export_fader_nn.py \
  --model runs/brave_fader_run \
  --db_path /path/to/lmdb \
  --output_path exports/fader.ts \
  --canonicalizer auto \
  --write_play_patch
```

Omit `--streaming` unless you need minimum-latency cached convs; with Fader models, streaming exports can go silent after ~1 s of nn~ blocks (512 samples).

### Plain TorchScript — Python / external attr concat

```bash
python RAVE/scripts/export_fader_ts.py \
  --model runs/brave_fader_run \
  --db_path /path/to/lmdb \
  --output_path exports/fader.ts
```

Both write:

- `*_attribute_stats.yaml` (copy of training stats)
- `*_host_controls.json` (names, kinds, min/max; nn export adds `nn_attributes`)

## vs vanilla RAVE export

| | [`export.py`](../../../scripts/export.py) | `export_fader_nn.py` | `export_fader_ts.py` |
|---|------------------------------------------|----------------------|----------------------|
| Model | `RAVE` 128-D | `FaderRAVE` 128+D | `FaderRAVE` 128+D |
| Backend | `nn_tilde` → Max **nn~** | `nn_tilde` → Max **nn~** | `torch.jit.script` |
| Attributes | N/A | nn~ knobs + torch extract | External concat |

## Related

- [`../realtime/README.md`](../realtime/README.md) — librosa `AttributeStream` (offline / non-nn~)
- [`../../../../docs/fader_host_controls.md`](../../../../docs/fader_host_controls.md) — nn~ attribute names and Max messages
