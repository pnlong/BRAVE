# Realtime attributes (`rave.fader.realtime`)

Block-wise attribute extraction for **inference** (TorchScript decode, demos, future host patches).

Training uses [`FaderAttributeDataset`](../dataset.py) + full-chunk crops; this package handles **streaming-sized blocks** at the same `T_lat` grid.

## Files

| File | Role |
|------|------|
| [`stream.py`](stream.py) | `AttributeStream` — extract + min/max normalize per block |
| [`__init__.py`](__init__.py) | Package marker |

## `AttributeStream`

```python
stream = AttributeStream.from_stats_path("path/to/attribute_stats.yaml", sr=44100)
attr_norm = stream.push(mono_block, latent_length=T_lat)  # (D_cont, T_lat)
# Host / trace: cat(z, attr_norm) → decode
```

Uses the same [`AudioDescriptorProvider`](../providers/audio.py) and min/max tables as training. Optional `discrete_norm` tensor can be appended for fixed discrete controls (e.g. `water_scene` selector).

## Demo

[`BRAVE/scripts/realtime_fader_demo.py`](../../../../scripts/realtime_fader_demo.py):

- Load exported `.ts` + sibling `*_attribute_stats.yaml`
- `--attr-scales rms=1.2,centroid=0.9` or legacy `--modulate-attr rms`

## Host integration

See [`BRAVE/docs/fader_host_controls.md`](../../../../docs/fader_host_controls.md) for attribute order, slider ranges, and discrete class mapping. Export writes `*_host_controls.json` from [`export_fader_ts.py`](../../../scripts/export_fader_ts.py).

**Note:** Vanilla RAVE Max/nn~ export ([`export.py`](../../../scripts/export.py)) does not include Fader attributes. Use [`export_fader_nn.py`](../../../scripts/export_fader_nn.py) for nn~ with knobs, or [`export_fader_ts.py`](../../../scripts/export_fader_ts.py) for plain TorchScript.
