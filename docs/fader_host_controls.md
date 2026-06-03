# Fader host controls (TorchScript inference)

Fader models widen the decoder input to `128 + D` by concatenating **normalized attributes** `attr_norm` with content latent `z`.

## Pipeline

```
audio block → encode → z (B, 128, T_lat)
audio block → AttributeStream.push → raw → normalize → attr_norm (B, D, T_lat)
decode(cat(z, attr_norm)) → audio
```

Export with [`RAVE/scripts/export_fader_ts.py`](../RAVE/scripts/export_fader_ts.py). The script writes:

- `fader.ts` — TorchScript module
- `fader_attribute_stats.yaml` — min/max and metadata
- `fader_host_controls.json` — host-facing slider metadata

## Attribute order

Row order matches `attribute_names` in `attribute_stats.yaml` (continuous rows first, then discrete).

## Continuous sliders

Each continuous name has global min/max from stats. After `normalize_attributes`, values lie in **[-1, 1]** per frame. Host UI can map slider `s ∈ [0, 1]` to `2*s - 1`.

## Discrete controls (`water_scene`, etc.)

Discrete rows use class index `0 … K-1` constant over `T_lat` in the sidecar. The traced model maps indices to decoder floats via `discrete_index_to_decoder_float` inside `normalize_all`.

For clip-level tags at inference, set the discrete row manually (e.g. water_scene `0|1|2`) instead of re-extracting from audio.

## Realtime demo

```bash
python scripts/realtime_fader_demo.py \
  --ts exports/fader.ts \
  --input clip.wav --output out.wav \
  --attr-scales rms=1.2,centroid=0.9
```

See [`scripts/realtime_fader_demo.py`](../scripts/realtime_fader_demo.py).

## Minifusion / HDF5

Vanilla BRAVE plugin export (`export_brave_plugin.py`) does not include Fader attributes. Use the TorchScript path above for 128+D concat.
