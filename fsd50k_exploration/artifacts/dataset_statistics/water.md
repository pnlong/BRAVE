## FSD50K subset audio statistics

| Field | Value |
| --- | --- |
| Partition | `dev_train` |
| Manifest | `/deepfreeze/pnlong/hai_lab/BRAVE/FSD50K/FSD50K.ground_truth/dev.csv` |
| WAV root | `/deepfreeze/pnlong/hai_lab/BRAVE/fsd50k_brave/water/audio_subset` |
| Whitelist tags (count) | 18 |
| Clips matched manifest ∩ whitelist ∩ wav exists | 2641 |
| Clips measured (ffprobe ok) | 2641 |
| ffprobe failures / missing duration | 0 |
| Total measured duration | 32251.13 s (~8.9586 h) |

### Raw WAV length (seconds)
- min: `0.3038`
- max: `57.5712`
- mean: `12.2117`
- stdev: `8.4715`
- median: `10.9834`
- p10 / p90: `1.6235` / `24.8999`

### RAVE preprocessing + training window (non-lazy pipeline)

Stock **`rave preprocess --num_signal 131072`** reads each file as a stream of exactly **`131072`** PCM samples per LMDB row; **`2.972154` s** at **44100 Hz** (see `scripts/preprocess.py`: only full chunks are written; a partial tail **is discarded**).
Stock **`rave train --n_signal 131072`** asks the loader for **`131072`** samples; with **non-lazy** LMDB entries of that length, `RandomCrop` in `rave.dataset.get_dataset` picks offset `0` (no shortening). Longer LMDB buffers would be random-cropped unless you change preprocessing.
- **`brave.gin`**: `SAMPLING_RATE = 44100` Hz (match `--sample-rate` for chunk math); causal conv paddings (`cc.get_padding.mode = 'causal'`); `valid_signal_crop = False`.
- **`dataset.split_dataset.max_residual = 1000`**: caps the **validation** split size (train/val **example counts**, not waveform padding).

| Chop metric | Value |
| --- | --- |
| `num_signal` / `n_signal` (samples) | 131072 |
| Chunk duration at 44100 Hz | `2.972154` s |
| Clips shorter than one chunk (< `2.972154` s) | 476 |
| Sum of floor(duration×sr)//num_signal over clips | `9563` LMDB rows |
| Approx training hours if each row trained once | `~7.8952` h |

### Hours per whitelist tag (ontology token / pseudo-prompt)

Each clip contributes its **full** measured duration once **per intersecting whitelist tag** (multi-label rows add to several tags).

| Tag | Clip count | Hours (approx) |
| --- | ---: | ---: |
| `water` | 1201 | 4.2549 |
| `liquid` | 979 | 2.1225 |
| `thunderstorm` | 403 | 1.9373 |
| `rain` | 447 | 1.7740 |
| `water_tap_and_faucet` | 238 | 0.9611 |
| `stream` | 181 | 0.8705 |
| `ocean` | 209 | 0.8703 |
| `sink_(filling_or_washing)` | 247 | 0.8427 |
| `toilet_flush` | 175 | 0.6848 |
| `waves_and_surf` | 140 | 0.5728 |
| `drip` | 198 | 0.4614 |
| `pour` | 125 | 0.4588 |
| `trickle_and_dribble` | 125 | 0.4588 |
| `bathtub_(filling_or_washing)` | 125 | 0.4565 |
| `splash_and_splatter` | 316 | 0.3500 |
| `boiling` | 55 | 0.2251 |
| `fill_(with_liquid)` | 74 | 0.2213 |
| `raindrop` | 101 | 0.1093 |
