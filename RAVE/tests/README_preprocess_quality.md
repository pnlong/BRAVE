# Preprocess packing + training RMS gate — manual checks

## Unit tests

From `BRAVE/RAVE` with `PYTHONPATH` including this directory:

```bash
python -m pytest tests/test_preprocess_plan.py -v
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

## Training RMS gate

With a preprocessed dataset and gin config enabling `dataset.maybe_reject_silent`, training uses the gate on the **train** split only. Override from CLI:

```bash
python scripts/train.py --config=../../configs/brave.gin ... --noreject_silent
python scripts/train.py ... --reject_silent --reject_silent_rms_db=-45
```

`configs/brave.gin` sets `enabled = True` and `rms_db_threshold = -50.0` by default.
