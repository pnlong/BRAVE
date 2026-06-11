# Fader export (`rave.fader.export`)

Stripped inference graph for TorchScript and nn~ — no latent discriminator, no GAN, no training stats mutation.

## Files

| File | Role |
|------|------|
| [`trace_model.py`](trace_model.py) | `FaderTraceModel`, `build_trace_model` |
| [`nn_module.py`](nn_module.py) | `ScriptedFaderRAVE` — nn_tilde wrapper with attribute knobs |
| [`torch_descriptors.py`](torch_descriptors.py) | JIT-safe continuous descriptor extractors |
| [`host_controls.py`](host_controls.py) | `*_host_controls.json` writer |
| [`__init__.py`](__init__.py) | Package exports |

## Inference API (`FaderTraceModel`)

```
audio x  → encode → z (B, 128, T_lat)
attr raw → normalize_all → attr_norm (B, D, T_lat)
y = decode(cat(z, attr_norm))
```

Buffers embed per-attribute min/max for continuous rows and discrete class counts from `attribute_stats.yaml`.

## CLI

### nn~ (Max / Pd) — attribute knobs

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python RAVE/scripts/export_fader_nn.py \
  --model runs/brave_fader_run \
  --db_path /path/to/lmdb \
  --output_path exports/fader.ts \
  --streaming
```

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
