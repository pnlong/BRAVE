# Preprocess packing + training RMS gate — manual checks

## Unit tests

From `BRAVE/RAVE` with `PYTHONPATH` including this directory:

```bash
python -m pytest tests/test_preprocess_plan.py tests/test_preprocess_denoise.py -v
```

## Preprocess integration (short files)

Create three tiny WAVs (requires `ffmpeg`):

```bash
mkdir -p /tmp/rave_pack_test
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=0.5" -ar 44100 /tmp/rave_pack_test/short_a.wav
ffmpeg -y -f lavfi -i "sine=frequency=880:duration=1.0" -ar 44100 /tmp/rave_pack_test/short_b.wav
ffmpeg -y -f lavfi -i "sine=frequency=220:duration=10.0" -ar 44100 /tmp/rave_pack_test/long.wav

python scripts/preprocess.py \
  --input_path=/tmp/rave_pack_test \
  --output_path=/tmp/rave_pack_test.lmdb \
  --num_signal=4096 \
  --workers=2
```

Expect `=== preprocess summary ===` with `short files >= 2`, `concat packs >= 1`, and `LMDB chunks written >= 1`.

Regression without packing:

```bash
python scripts/preprocess.py \
  --input_path=/tmp/rave_pack_test \
  --output_path=/tmp/rave_pack_test_noconcat.lmdb \
  --num_signal=4096 \
  --noconcat_short
```

Short files alone should contribute zero chunks unless padded.

## Optional denoise (`--denoise`)

Stationary spectral gating before LMDB write (useful for noisy environmental texture ablations):

```bash
python scripts/preprocess.py \
  --input_path=/path/to/wavs \
  --output_path=/path/to/preprocessed_denoised.lmdb \
  --denoise \
  --denoise_strength=0.75
```

`metadata.yaml` in the LMDB folder records `denoise`, `denoise_strength`, and `denoise_noise_sec`.
Set `--denoise_noise_sec=0.5` to estimate the noise floor from clip starts only.

## Training RMS gate

With a preprocessed dataset and gin config enabling `dataset.maybe_reject_silent`, training uses the gate on the **train** split only. Override from CLI:

```bash
python scripts/train.py --config=../../configs/brave.gin ... --noreject_silent
python scripts/train.py ... --reject_silent --reject_silent_rms_db=-35
```

`configs/brave.gin` sets `enabled = True`, `rms_db_threshold = -40.0`, and `max_tries = 16` by default.
