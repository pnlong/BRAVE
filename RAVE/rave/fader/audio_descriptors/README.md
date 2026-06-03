# Audio descriptors (`rave.fader.audio_descriptors`)

Frame-wise features extracted from **mono** audio and resampled to latent length `T_lat`. Used by [`AudioDescriptorProvider`](../providers/audio.py) and [`compute_descriptor_matrix`](../attributes.py).

## Files

| File | Role |
|------|------|
| [`features.py`](features.py) | `compute_librosa`, `compute_timbral`, `compute_all` — main entry |
| [`timbral_models/`](timbral_models/README.md) | IRCAM timbral (Audio Commons) extractors |

## Available descriptor names

**Librosa** (native sample rate): `rms`, `zcr`, `f0`, `centroid`, `bandwidth`, `rolloff`, `flatness`

**Timbral** (upsampled to 44.1 kHz when needed): `booming`, `brightness`, `depth`, `hardness`, `reverb`, `roughness`, `sharpness`, `warmth`

Add a name to gin `continuous_attributes`, include it in `precompute_descriptors.py`, and re-run training stats.

Unknown names trigger a **warning** and produce a zero row (see `compute_descriptor_matrix` in [`attributes.py`](../attributes.py)).

## Pipeline

```
mono waveform (samples)
  → compute_all(..., resample=T_lat)
  → dict[name → (T_lat,)]
  → stack in gin attribute order → (D_cont, T_lat)
```

Default training list (see [`brave_fader.gin`](../../../../configs/brave_fader.gin)): `centroid`, `rms`, `bandwidth`, `sharpness`, `booming`.

**RMS** is frame-wise librosa RMS — decoder control + binned latent CE; optional `rms_gate` binning in precompute. This is separate from RAVE’s `reject_silent` dataset filter.

## Dependencies

- `librosa` — spectral + RMS + ZCR + YIN
- `torchaudio` — optional resampler inside timbral path
- Timbral code under [`timbral_models/`](timbral_models/) (vendored, modified for time series)

## Related scripts

- [`RAVE/scripts/precompute_descriptors.py`](../../../scripts/precompute_descriptors.py) — dataset min/max and quantile bins
- [`RAVE/scripts/eval_fader_attributes.py`](../../../scripts/eval_fader_attributes.py) — re-extracts after attribute swap
