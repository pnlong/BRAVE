# Timbral models (`audio_descriptors.timbral_models`)

Vendored [Audio Commons](https://github.com/AudioCommons/timbral-models) timbral descriptors, adapted to return **time series** (not only clip-level scalars) for Fader conditioning.

Called from [`features.compute_timbral`](../features.py) when a timbral name appears in the gin `continuous_attributes` list.

## Models

| Module | Gin name | Quantity (typical) |
|--------|----------|-------------------|
| `Timbral_Booming.py` | `booming` | Low-frequency emphasis |
| `Timbral_Brightness.py` | `brightness` | High-frequency emphasis |
| `Timbral_Depth.py` | `depth` | Spectral depth |
| `Timbral_Hardness.py` | `hardness` | Attack / transient hardness |
| `Timbral_Reverb.py` | `reverb` | Reverberation decay estimate |
| `Timbral_Roughness.py` | `roughness` | Sensory roughness |
| `Timbral_Sharpness.py` | `sharpness` | Perceptual sharpness |
| `Timbral_Warmth.py` | `warmth` | Low-mid warmth |

Shared utilities: [`timbral_util.py`](timbral_util.py) (envelopes, loudness helpers, segmentation).

## Sample rate

If input `sr < 44100`, audio is resampled to 44.1 kHz before timbral functions run (see `compute_timbral` in [`features.py`](../features.py)).

## Editing

When adding a new timbral export:

1. Implement or expose a frame-wise function in the appropriate `Timbral_*.py`
2. Register the gin name in `features_dict` inside [`features.py`](../features.py) `compute_timbral`
3. Re-run `precompute_descriptors.py` and training

`Timbral_Extractor.py` is legacy scaffolding; Fader uses the individual model modules via `features.py`.
