# Attribute providers (`rave.fader.providers`)

Answer one question per training sample: **where does `attr_raw` come from?**

The dataset calls [`AttributeLoader.load(index, audio)`](loader.py) and receives `(D_total, T_lat)` — continuous rows first, then discrete, matching gin attribute order.

## Modules

| File | Class | Kind |
|------|-------|------|
| [`base.py`](base.py) | `ContinuousAttributeProvider`, `DiscreteAttributeProvider` | ABCs |
| [`loader.py`](loader.py) | `AttributeLoader`, `build_attribute_loader` | Gin factory + facade |
| [`audio.py`](audio.py) | `AudioDescriptorProvider`, `CachingAudioDescriptorProvider` | On-the-fly librosa/timbral |
| [`sidecar.py`](sidecar.py) | `SidecarAttributeProvider` | `{db_path}/attribute_sidecar.yaml` |
| [`null.py`](null.py) | `NullDiscreteProvider` | Zero discrete placeholder |
| [`midi_cc.py`](midi_cc.py) | `MidiCCSidecarProvider` | Stub: CC trajectories from `midi_cc_sidecar.yaml` |
| [`learned.py`](learned.py) | `LearnedFeatureProvider` | Stub: `{db_path}/learned_features/{index:08d}.npy` |

Public imports: [`__init__.py`](__init__.py) re-exports the above for `from rave.fader.providers import ...`.

## Default wiring (`build_attribute_loader`)

Configured in [`brave_fader.gin`](../../../../configs/brave_fader.gin):

```
continuous  → AudioDescriptorProvider (cropped audio)
discrete    → NullDiscreteProvider + SidecarAttributeProvider if YAML exists
```

When `attribute_sidecar.yaml` is present, discrete values are read from the sidecar. Continuous still comes from audio unless `use_audio_descriptors=False`.

## Gin flags (loader)

| Flag | Default | Effect |
|------|---------|--------|
| `use_audio_descriptors` | `True` | Librosa/timbral continuous extraction |
| `cache_descriptors` | `False` | Wrap audio provider with LRU on `(index, hash(audio))` |
| `cache_max_entries` | `4096` | LRU size when caching enabled |
| `use_midi_cc` | `False` | Use MIDI CC sidecar **only if** no audio provider |
| `use_learned_features` | `False` | Use learned `.npy` **only if** no audio provider |

MIDI CC and learned providers are **stubs** — no live host capture, no merging with audio descriptors yet.

## Sidecar YAML shape

```yaml
index_key: lmdb_index
attributes:
  water_scene:
    values:
      "00000042": 1      # scalar → broadcast over T_lat
      "00000043": [0, 0, 1, 1]  # optional per-frame list
```

Built by [`RAVE/scripts/build_attribute_sidecar.py`](../../../scripts/build_attribute_sidecar.py) for FSD50k (`water_scene`, etc.).

## Continuous vs discrete

| | Continuous | Discrete |
|---|------------|----------|
| Provider | Usually `AudioDescriptorProvider` | `SidecarAttributeProvider` (+ `NullDiscreteProvider`) |
| Raw values | Float trajectories | Integer class indices |
| `FaderRAVE` | min/max → `attr_norm`; bins → `attr_cls` | index→float → `attr_norm`; same index → `attr_cls` |
| Example names | `rms`, `centroid`, `sharpness` | `water_scene` (0/1/2) |

**MIDI CC** is modeled as **continuous** (knob trajectories), not note numbers. Note-based discrete labels belong in `attribute_sidecar.yaml` like `water_scene`.
