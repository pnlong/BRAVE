## FSD50K subset audio statistics

| Field | Value |
| --- | --- |
| Partition | `dev_train` |
| Manifest | `/deepfreeze/pnlong/hai_lab/BRAVE/FSD50K/FSD50K.ground_truth/dev.csv` |
| WAV root | `/deepfreeze/pnlong/hai_lab/BRAVE/fsd50k_brave/water_industrial_birds/audio_subset` |
| Whitelist tags (count) | 30 |
| Clips matched manifest ∩ whitelist ∩ wav exists | 6706 |
| Clips measured (ffprobe ok) | 6706 |
| ffprobe failures / missing duration | 0 |
| Total measured duration | 64923.31 s (~18.0343 h) |

### Raw WAV length (seconds)
- min: `0.3000`
- max: `57.5712`
- mean: `9.6814`
- stdev: `8.4714`
- median: `7.1947`
- p10 / p90: `0.8870` / `23.3061`

### RAVE preprocessing + training window (non-lazy pipeline)

Stock **`rave preprocess --num_signal 131072`** reads each file as a stream of exactly **`131072`** PCM samples per LMDB row; **`2.972154` s** at **44100 Hz** (see `scripts/preprocess.py`: only full chunks are written; a partial tail **is discarded**).
Stock **`rave train --n_signal 131072`** asks the loader for **`131072`** samples; with **non-lazy** LMDB entries of that length, `RandomCrop` in `rave.dataset.get_dataset` picks offset `0` (no shortening). Longer LMDB buffers would be random-cropped unless you change preprocessing.
- **`brave.gin`**: `SAMPLING_RATE = 44100` Hz (match `--sample-rate` for chunk math); causal conv paddings (`cc.get_padding.mode = 'causal'`); `valid_signal_crop = False`.
- **`dataset.split_dataset.max_residual = 1000`**: caps the **validation** split size (train/val **example counts**, not waveform padding).

| Chop metric | Value |
| --- | --- |
| `num_signal` / `n_signal` (samples) | 131072 |
| Chunk duration at 44100 Hz | `2.972154` s |
| Clips shorter than one chunk (< `2.972154` s) | 2044 |
| Sum of floor(duration×sr)//num_signal over clips | `18712` LMDB rows |
| Approx training hours if each row trained once | `~15.4486` h |

### Hours per whitelist tag (ontology token / pseudo-prompt)

Each clip contributes its **full** measured duration once **per intersecting whitelist tag** (multi-label rows add to several tags).

| Tag | Clip count | Hours (approx) |
| --- | ---: | ---: |
| `wild_animals` | 1516 | 4.5398 |
| `water` | 1201 | 4.2549 |
| `bird` | 1117 | 3.0684 |
| `mechanisms` | 960 | 2.4510 |
| `liquid` | 979 | 2.1225 |
| `thunderstorm` | 403 | 1.9373 |
| `thunder` | 389 | 1.8753 |
| `rain` | 447 | 1.7740 |
| `bird_vocalization_and_bird_call_and_bird_song` | 459 | 1.5952 |
| `glass` | 858 | 1.0358 |
| `water_tap_and_faucet` | 238 | 0.9611 |
| `stream` | 181 | 0.8705 |
| `ocean` | 209 | 0.8703 |
| `sink_(filling_or_washing)` | 247 | 0.8427 |
| `toilet_flush` | 175 | 0.6848 |
| `wind` | 248 | 0.6356 |
| `waves_and_surf` | 140 | 0.5728 |
| `drip` | 198 | 0.4614 |
| `pour` | 125 | 0.4588 |
| `trickle_and_dribble` | 125 | 0.4588 |
| `bathtub_(filling_or_washing)` | 125 | 0.4565 |
| `chirp_and_tweet` | 162 | 0.4186 |
| `boat_and_water_vehicle` | 89 | 0.3589 |
| `splash_and_splatter` | 316 | 0.3500 |
| `wood` | 265 | 0.2828 |
| `boiling` | 55 | 0.2251 |
| `fill_(with_liquid)` | 74 | 0.2213 |
| `gurgling` | 103 | 0.2072 |
| `raindrop` | 101 | 0.1093 |
| `whoosh_and_swoosh_and_swish` | 227 | 0.0902 |
