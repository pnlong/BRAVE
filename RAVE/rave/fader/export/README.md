# Fader export (`rave.fader.export`)

Stripped inference graph for TorchScript — no latent discriminator, no GAN, no training stats mutation.

## Files

| File | Role |
|------|------|
| [`trace_model.py`](trace_model.py) | `FaderTraceModel`, `build_trace_model` |
| [`__init__.py`](__init__.py) | Package marker |

## Inference API (`FaderTraceModel`)

```
audio x  → encode → z (B, 128, T_lat)
attr raw → normalize_all → attr_norm (B, D, T_lat)
y = decode(cat(z, attr_norm))
```

Buffers embed per-attribute min/max for continuous rows and discrete class counts from `attribute_stats.yaml`.

## CLI

From BRAVE root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
python RAVE/scripts/export_fader_ts.py \
  --model runs/brave_fader_run \
  --db_path /path/to/lmdb \
  --output_path exports/fader.ts
```

Also writes:

- `fader_attribute_stats.yaml` (copy of training stats)
- `fader_host_controls.json` (names, kinds, min/max, `T_lat`, `sr`)

## vs vanilla RAVE export

| | [`export.py`](../../../scripts/export.py) | `export_fader_ts.py` |
|---|------------------------------------------|----------------------|
| Model | `RAVE` 128-D | `FaderRAVE` 128+D |
| Backend | `nn_tilde` → Max **nn~** | `torch.jit.script` |
| Attributes | N/A | External concat required |

## Related

- [`../realtime/README.md`](../realtime/README.md) — live attribute extraction
- [`../../../../docs/fader_host_controls.md`](../../../../docs/fader_host_controls.md) — host UI metadata
