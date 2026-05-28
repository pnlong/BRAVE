## FSD50K subset — post-preprocess (LMDB) estimates

Subset statistics for clips matching your whitelist on disk. Sections below estimate **post-preprocess** yield (LMDB), not raw WAV totals alone.

#### Run setup (input subset)

This small table describes **which files were measured** before any preprocess simulation. *Raw input duration* is the sum of ffprobe lengths; everything else in the report applies preprocess rules to that pile of audio.

- **Partition** — FSD50K CSV split (`dev_train`, etc.).
- **WAV root** — Folder of `<clip_id>.wav` files analyzed.
- **Clips in report** — Manifest rows with a whitelist tag and an existing WAV.
- **ffprobe failures** — Files that could not be measured (excluded from simulation).
- **Raw input duration** — Total seconds of audio **before** preprocess (on-disk WAV).

| Field | Value |
| --- | --- |
| Partition | `dev_train` |
| WAV root | `/deepfreeze/pnlong/hai_lab/BRAVE/fsd50k_brave/water_industrial_birds/audio_subset` |
| Clips in report | 6706 |
| ffprobe failures | 0 |
| Raw input duration | `64923.31` s |

### How to read this report

This script measures your FSD50K WAVs with ffprobe, then **simulates** what `python RAVE/scripts/preprocess.py` would write into an LMDB database. It does **not** open a preprocessed LMDB; it predicts row counts and how much audio is kept vs thrown away.

RAVE preprocess (non-lazy, the usual BRAVE path) reads each file in fixed-size chunks. With `--num_signal 131072` and 44100 Hz, each LMDB entry holds **262144 samples (~5.94 s)**. Training later takes a shorter random crop (**131072 samples ~2.97 s**) from each entry via `RandomCrop` in `rave/dataset.py`.

You ran **`--compare-concat`**, so the report shows **two scenarios** for the same clips:

1. **Concat ON** — matches default preprocess (`--concat_short`): clips shorter than one LMDB row are shuffled (seed `--concat_seed`), concatenated in groups until the group is long enough, then chunked.
2. **Concat OFF** — matches `--noconcat_short`: every clip shorter than one row is dropped with no packing.

Compare **Stored in LMDB** and **Clips fully discarded** between the two sections to see how much packing recovers short FSD50K clips.

**Discard types** (used in the summary and per-tag discarded tables):

- **Full clip** — The entire WAV produces **zero** LMDB rows. Typical when concat is off and the clip is too short, or concat is on but the clip only appears in a **final pack** that never reached one row (unless `--pad_short_remainder`).
- **Long tail** — The clip is long enough to process alone. All full rows are saved; any samples left at the end of the file (less than one row) are trimmed off.
- **Pack remainder** — **Concat on only.** After grouping short clips, the last group still does not add up to one row, so the whole group is discarded.
- **Pack chunk tail** — **Concat on only.** A group was long enough to create at least one LMDB row, but after concatenating clips and cutting row-sized blocks, the **leftover audio at the end of that group** is still shorter than one row and is dropped (like a long-tail trim, but on glued-together shorts).

**Multi-label clips:** FSD50K clips can have several tags. Per-tag tables attribute the same clip's stored or discarded time to **each** whitelist tag on that clip, so tag hours can sum to more than the global total.

### Concat ON (default preprocess)

**Preprocess flags simulated:** `lazy=non-lazy, concat_short=on, concat_seed=42, pad_short_remainder=no`

This section answers: *if I preprocess this subset with these flags, how much audio lands in the LMDB, how much is lost, and why?* Use it as a budget before running `RAVE/scripts/preprocess.py` on disk.

#### Scenario summary (global totals)

One row per quantity below. **Stored** is what survives as full LMDB chunks; **Discarded** breaks down into the four mechanisms defined in the introduction.

- **`num_signal`** — `--num_signal` / `--n_signal` for training (here 131072). Preprocess non-lazy row size is still 2× this value.
- **LMDB row size (samples)** — Samples per LMDB entry (262144 = 2×num_signal in non-lazy mode).
- **LMDB row duration @ 44100 Hz** — Seconds of audio in each LMDB entry (5.9443 s).
- **Train crop duration** — Seconds drawn per training step after `RandomCrop` (2.9722 s).
- **Input audio (measured clips)** — Sum of ffprobe durations for all clips in this subset.
- **Stored in LMDB** — Total seconds that become complete LMDB rows (usable training material).
- **Discarded total** — Input minus stored; equals the four discard bullets below.
- **fully discarded clips** — Seconds from files/clips that produced **no** LMDB rows.
- **long-file tail (last partial chunk)** — Seconds trimmed off the end of **long** files.
- **short pack remainder (unpackable)** — Seconds in the final short pack that never reached one row.
- **pack concat chunk tail** — Seconds trimmed after chunking a **successful** short pack.
- **LMDB rows (estimated)** — Count of LMDB keys (`00000000`, …) written.
- **LMDB row-hours (stored window)** — `rows × row duration` — hours of waveform actually stored.
- **Train crop-hours (1 crop / row)** — `rows × train crop` — rough training exposure if each row is seen once.
- **Short files (< one row)** — Clips shorter than 5.9443 s by themselves (candidates for packing when concat is on).
- **Long files (≥ one row alone)** — Clips that already fill at least one LMDB row without packing.
- **Concat packs** — How many short-clip groups were concatenated before chunking (0 if concat off).
- **Clips fully discarded (0 stored)** — Number of clips with **zero** stored seconds.
- **Utilization (stored / input)** — Percentage of raw duration that becomes LMDB audio.

| Metric | Value |
| --- | --- |
| `num_signal` (train crop samples) | 131072 |
| LMDB row size (samples) | 262144 |
| LMDB row duration @ 44100 Hz | `5.944308` s |
| Train crop duration | `2.972154` s |
| Input audio (measured clips) | `64923.31` s (`18.0343` h) |
| **Stored in LMDB** | `53183.73` s (`14.7733` h) |
| **Discarded total** | `11739.52` s (`3.2610` h) |
| └ fully discarded clips | `0.00` s |
| └ long-file tail (last partial chunk) | `10194.43` s |
| └ short pack remainder (unpackable) | `0.00` s |
| └ pack concat chunk tail | `1545.09` s |
| LMDB rows (estimated) | `8947` |
| LMDB row-hours (stored window) | `~14.7733` h |
| Train crop-hours (1 crop / row) | `~7.3866` h |
| Short files (< one row) | 2987 |
| Long files (≥ one row alone) | 3719 |
| Concat packs | 902 |
| Clips fully discarded (0 stored) | 0 |
| Utilization (stored / input) | `81.9%` |

#### Per-tag: stored in LMDB (estimated)

This table shows **how much training material each ontology tag contributes** after preprocess. It helps compare tags (e.g. `water` vs `raindrop`) when building class-conditioned subsets. If a clip has multiple whitelist labels, its stored time is counted toward **each** of those tags.

- **Tag** — Label from your `--whitelist` file (FSD50K ontology token).
- **Clips w/ stored audio** — How many clips carrying this tag still have **some** audio in the LMDB (not fully discarded).
- **LMDB hours (rows×row)** — Estimated LMDB row-hours credited to this tag (pro-rata when clips were packed together).
- **Stored hours** — Wall-clock hours of waveform from this tag that survive preprocess (should match LMDB hours up to rounding).

| Tag | Clips w/ stored audio | LMDB hours (rows×row) | Stored hours |
| --- | ---: | ---: | ---: |
| `wild_animals` | 1516 | 3.7336 | 3.7336 |
| `water` | 1201 | 3.5448 | 3.5448 |
| `bird` | 1117 | 2.5197 | 2.5197 |
| `mechanisms` | 960 | 2.0103 | 2.0103 |
| `liquid` | 979 | 1.7022 | 1.7022 |
| `thunderstorm` | 403 | 1.6294 | 1.6294 |
| `thunder` | 389 | 1.5798 | 1.5798 |
| `rain` | 447 | 1.4890 | 1.4890 |
| `bird_vocalization_and_bird_call_and_bird_song` | 459 | 1.3293 | 1.3293 |
| `glass` | 858 | 0.8280 | 0.8280 |
| `water_tap_and_faucet` | 238 | 0.7819 | 0.7819 |
| `ocean` | 209 | 0.7272 | 0.7272 |
| `stream` | 181 | 0.7264 | 0.7264 |
| `sink_(filling_or_washing)` | 247 | 0.6829 | 0.6829 |
| `toilet_flush` | 175 | 0.5523 | 0.5523 |
| `wind` | 248 | 0.5022 | 0.5022 |
| `waves_and_surf` | 140 | 0.4790 | 0.4790 |
| `drip` | 198 | 0.3857 | 0.3857 |
| `pour` | 125 | 0.3811 | 0.3811 |
| `trickle_and_dribble` | 125 | 0.3811 | 0.3811 |
| `bathtub_(filling_or_washing)` | 125 | 0.3691 | 0.3691 |
| `chirp_and_tweet` | 162 | 0.3432 | 0.3432 |
| `boat_and_water_vehicle` | 89 | 0.3007 | 0.3007 |
| `splash_and_splatter` | 316 | 0.2794 | 0.2794 |
| `wood` | 265 | 0.2252 | 0.2252 |
| `boiling` | 55 | 0.1851 | 0.1851 |
| `fill_(with_liquid)` | 74 | 0.1747 | 0.1747 |
| `gurgling` | 103 | 0.1705 | 0.1705 |
| `raindrop` | 101 | 0.0870 | 0.0870 |
| `whoosh_and_swoosh_and_swish` | 227 | 0.0743 | 0.0743 |

#### Per-tag: discarded (estimated)

This table shows **where each tag loses audio**, using the same four discard types as the scenario summary. Use it to see whether a tag suffers mostly from short clips (Full clip / Pack remainder) or from end-trimming on longer recordings (Long tail / Pack chunk tail). All values are in **hours**.

- **Tag** — Whitelist label.
- **Full clip** — Hours lost because clips with this tag never produced any LMDB row.
- **Long tail** — Hours trimmed from the **end** of long clips tagged with this label.
- **Pack remainder** — Hours from this tag's clips that were only in a **final** short pack too small to store (concat on).
- **Pack chunk tail** — Hours trimmed after chunking a concat pack that included this tag.
- **Total discarded h** — Sum of the four discard columns above for this tag.

| Tag | Full clip | Long tail | Pack remainder | Pack chunk tail | Total discarded h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wild_animals` | 0.0000 | 0.7087 | 0.0000 | 0.0975 | 0.8062 |
| `water` | 0.0000 | 0.6518 | 0.0000 | 0.0583 | 0.7101 |
| `bird` | 0.0000 | 0.4711 | 0.0000 | 0.0775 | 0.5486 |
| `mechanisms` | 0.0000 | 0.3810 | 0.0000 | 0.0596 | 0.4407 |
| `liquid` | 0.0000 | 0.3450 | 0.0000 | 0.0754 | 0.4204 |
| `thunderstorm` | 0.0000 | 0.3021 | 0.0000 | 0.0059 | 0.3080 |
| `thunder` | 0.0000 | 0.2896 | 0.0000 | 0.0059 | 0.2955 |
| `rain` | 0.0000 | 0.2737 | 0.0000 | 0.0113 | 0.2850 |
| `bird_vocalization_and_bird_call_and_bird_song` | 0.0000 | 0.2453 | 0.0000 | 0.0206 | 0.2659 |
| `glass` | 0.0000 | 0.1227 | 0.0000 | 0.0852 | 0.2079 |
| `water_tap_and_faucet` | 0.0000 | 0.1733 | 0.0000 | 0.0060 | 0.1793 |
| `sink_(filling_or_washing)` | 0.0000 | 0.1468 | 0.0000 | 0.0129 | 0.1598 |
| `stream` | 0.0000 | 0.1410 | 0.0000 | 0.0030 | 0.1441 |
| `ocean` | 0.0000 | 0.1328 | 0.0000 | 0.0103 | 0.1432 |
| `wind` | 0.0000 | 0.1154 | 0.0000 | 0.0181 | 0.1334 |
| `toilet_flush` | 0.0000 | 0.1282 | 0.0000 | 0.0043 | 0.1325 |
| `waves_and_surf` | 0.0000 | 0.0845 | 0.0000 | 0.0092 | 0.0938 |
| `bathtub_(filling_or_washing)` | 0.0000 | 0.0807 | 0.0000 | 0.0068 | 0.0875 |
| `pour` | 0.0000 | 0.0734 | 0.0000 | 0.0043 | 0.0777 |
| `trickle_and_dribble` | 0.0000 | 0.0734 | 0.0000 | 0.0043 | 0.0777 |
| `drip` | 0.0000 | 0.0666 | 0.0000 | 0.0091 | 0.0758 |
| `chirp_and_tweet` | 0.0000 | 0.0675 | 0.0000 | 0.0078 | 0.0753 |
| `splash_and_splatter` | 0.0000 | 0.0408 | 0.0000 | 0.0298 | 0.0706 |
| `boat_and_water_vehicle` | 0.0000 | 0.0540 | 0.0000 | 0.0042 | 0.0582 |
| `wood` | 0.0000 | 0.0301 | 0.0000 | 0.0275 | 0.0576 |
| `fill_(with_liquid)` | 0.0000 | 0.0401 | 0.0000 | 0.0066 | 0.0466 |
| `boiling` | 0.0000 | 0.0389 | 0.0000 | 0.0010 | 0.0400 |
| `gurgling` | 0.0000 | 0.0301 | 0.0000 | 0.0066 | 0.0367 |
| `raindrop` | 0.0000 | 0.0157 | 0.0000 | 0.0067 | 0.0224 |
| `whoosh_and_swoosh_and_swish` | 0.0000 | 0.0040 | 0.0000 | 0.0118 | 0.0159 |

#### Per-tag: clips fully discarded (0 LMDB audio)

This table counts **clips** (not hours) that contribute **nothing** to the LMDB for each tag—every second of that clip is lost. With concat off, this is usually all clips shorter than one row; with concat on, it should be small unless packing fails for the final group.

- **Tag** — Whitelist label.
- **Clips fully discarded** — Number of clips with this tag that have **zero** stored audio after preprocess.

| Tag | Clips fully discarded |
| --- | ---: |
| *(none)* | 0 |

_Every clip in this scenario contributes at least some audio to the LMDB._

### Concat OFF (--noconcat_short)

**Preprocess flags simulated:** `lazy=non-lazy, concat_short=off, concat_seed=42, pad_short_remainder=no`

This section answers: *if I preprocess this subset with these flags, how much audio lands in the LMDB, how much is lost, and why?* Use it as a budget before running `RAVE/scripts/preprocess.py` on disk.

#### Scenario summary (global totals)

One row per quantity below. **Stored** is what survives as full LMDB chunks; **Discarded** breaks down into the four mechanisms defined in the introduction.

- **`num_signal`** — `--num_signal` / `--n_signal` for training (here 131072). Preprocess non-lazy row size is still 2× this value.
- **LMDB row size (samples)** — Samples per LMDB entry (262144 = 2×num_signal in non-lazy mode).
- **LMDB row duration @ 44100 Hz** — Seconds of audio in each LMDB entry (5.9443 s).
- **Train crop duration** — Seconds drawn per training step after `RandomCrop` (2.9722 s).
- **Input audio (measured clips)** — Sum of ffprobe durations for all clips in this subset.
- **Stored in LMDB** — Total seconds that become complete LMDB rows (usable training material).
- **Discarded total** — Input minus stored; equals the four discard bullets below.
- **fully discarded clips** — Seconds from files/clips that produced **no** LMDB rows.
- **long-file tail (last partial chunk)** — Seconds trimmed off the end of **long** files.
- **short pack remainder (unpackable)** — Seconds in the final short pack that never reached one row.
- **pack concat chunk tail** — Seconds trimmed after chunking a **successful** short pack.
- **LMDB rows (estimated)** — Count of LMDB keys (`00000000`, …) written.
- **LMDB row-hours (stored window)** — `rows × row duration` — hours of waveform actually stored.
- **Train crop-hours (1 crop / row)** — `rows × train crop` — rough training exposure if each row is seen once.
- **Short files (< one row)** — Clips shorter than 5.9443 s by themselves (candidates for packing when concat is on).
- **Long files (≥ one row alone)** — Clips that already fill at least one LMDB row without packing.
- **Concat packs** — How many short-clip groups were concatenated before chunking (0 if concat off).
- **Clips fully discarded (0 stored)** — Number of clips with **zero** stored seconds.
- **Utilization (stored / input)** — Percentage of raw duration that becomes LMDB audio.

| Metric | Value |
| --- | --- |
| `num_signal` (train crop samples) | 131072 |
| LMDB row size (samples) | 262144 |
| LMDB row duration @ 44100 Hz | `5.944308` s |
| Train crop duration | `2.972154` s |
| Input audio (measured clips) | `64923.31` s (`18.0343` h) |
| **Stored in LMDB** | `47821.96` s (`13.2839` h) |
| **Discarded total** | `17101.31` s (`4.7504` h) |
| └ fully discarded clips | `6906.89` s |
| └ long-file tail (last partial chunk) | `10194.43` s |
| └ short pack remainder (unpackable) | `0.00` s |
| └ pack concat chunk tail | `0.00` s |
| LMDB rows (estimated) | `8045` |
| LMDB row-hours (stored window) | `~13.2839` h |
| Train crop-hours (1 crop / row) | `~6.6419` h |
| Short files (< one row) | 2987 |
| Long files (≥ one row alone) | 3719 |
| Concat packs | 0 |
| Clips fully discarded (0 stored) | 2987 |
| Utilization (stored / input) | `73.7%` |

#### Per-tag: stored in LMDB (estimated)

This table shows **how much training material each ontology tag contributes** after preprocess. It helps compare tags (e.g. `water` vs `raindrop`) when building class-conditioned subsets. If a clip has multiple whitelist labels, its stored time is counted toward **each** of those tags.

- **Tag** — Label from your `--whitelist` file (FSD50K ontology token).
- **Clips w/ stored audio** — How many clips carrying this tag still have **some** audio in the LMDB (not fully discarded).
- **LMDB hours (rows×row)** — Estimated LMDB row-hours credited to this tag (pro-rata when clips were packed together).
- **Stored hours** — Wall-clock hours of waveform from this tag that survive preprocess (should match LMDB hours up to rounding).

| Tag | Clips w/ stored audio | LMDB hours (rows×row) | Stored hours |
| --- | ---: | ---: | ---: |
| `wild_animals` | 923 | 3.3965 | 3.3965 |
| `water` | 854 | 3.3635 | 3.3635 |
| `bird` | 631 | 2.2473 | 2.2473 |
| `mechanisms` | 516 | 1.7899 | 1.7899 |
| `thunderstorm` | 377 | 1.6083 | 1.6083 |
| `thunder` | 363 | 1.5587 | 1.5587 |
| `rain` | 350 | 1.4514 | 1.4514 |
| `liquid` | 460 | 1.4398 | 1.4398 |
| `bird_vocalization_and_bird_call_and_bird_song` | 306 | 1.2566 | 1.2566 |
| `water_tap_and_faucet` | 214 | 0.7612 | 0.7612 |
| `stream` | 171 | 0.7166 | 0.7166 |
| `ocean` | 172 | 0.7018 | 0.7018 |
| `sink_(filling_or_washing)` | 194 | 0.6440 | 0.6440 |
| `toilet_flush` | 163 | 0.5416 | 0.5416 |
| `glass` | 195 | 0.5284 | 0.5284 |
| `waves_and_surf` | 110 | 0.4574 | 0.4574 |
| `wind` | 162 | 0.4508 | 0.4508 |
| `pour` | 94 | 0.3649 | 0.3649 |
| `trickle_and_dribble` | 94 | 0.3649 | 0.3649 |
| `drip` | 89 | 0.3517 | 0.3517 |
| `bathtub_(filling_or_washing)` | 98 | 0.3484 | 0.3484 |
| `chirp_and_tweet` | 83 | 0.3104 | 0.3104 |
| `boat_and_water_vehicle` | 65 | 0.2857 | 0.2857 |
| `boiling` | 50 | 0.1816 | 0.1816 |
| `splash_and_splatter` | 58 | 0.1552 | 0.1552 |
| `fill_(with_liquid)` | 51 | 0.1536 | 0.1536 |
| `gurgling` | 49 | 0.1486 | 0.1486 |
| `wood` | 47 | 0.1222 | 0.1222 |
| `raindrop` | 20 | 0.0627 | 0.0627 |
| `whoosh_and_swoosh_and_swish` | 11 | 0.0215 | 0.0215 |

#### Per-tag: discarded (estimated)

This table shows **where each tag loses audio**, using the same four discard types as the scenario summary. Use it to see whether a tag suffers mostly from short clips (Full clip / Pack remainder) or from end-trimming on longer recordings (Long tail / Pack chunk tail). All values are in **hours**.

- **Tag** — Whitelist label.
- **Full clip** — Hours lost because clips with this tag never produced any LMDB row.
- **Long tail** — Hours trimmed from the **end** of long clips tagged with this label.
- **Pack remainder** — Hours from this tag's clips that were only in a **final** short pack too small to store (concat on).
- **Pack chunk tail** — Hours trimmed after chunking a concat pack that included this tag.
- **Total discarded h** — Sum of the four discard columns above for this tag.

| Tag | Full clip | Long tail | Pack remainder | Pack chunk tail | Total discarded h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wild_animals` | 0.4346 | 0.7087 | 0.0000 | 0.0000 | 1.1433 |
| `water` | 0.2396 | 0.6518 | 0.0000 | 0.0000 | 0.8914 |
| `bird` | 0.3500 | 0.4711 | 0.0000 | 0.0000 | 0.8211 |
| `liquid` | 0.3377 | 0.3450 | 0.0000 | 0.0000 | 0.6827 |
| `mechanisms` | 0.2801 | 0.3810 | 0.0000 | 0.0000 | 0.6611 |
| `glass` | 0.3848 | 0.1227 | 0.0000 | 0.0000 | 0.5074 |
| `bird_vocalization_and_bird_call_and_bird_song` | 0.0933 | 0.2453 | 0.0000 | 0.0000 | 0.3386 |
| `thunderstorm` | 0.0270 | 0.3021 | 0.0000 | 0.0000 | 0.3291 |
| `rain` | 0.0489 | 0.2737 | 0.0000 | 0.0000 | 0.3226 |
| `thunder` | 0.0270 | 0.2896 | 0.0000 | 0.0000 | 0.3166 |
| `water_tap_and_faucet` | 0.0267 | 0.1733 | 0.0000 | 0.0000 | 0.1999 |
| `sink_(filling_or_washing)` | 0.0519 | 0.1468 | 0.0000 | 0.0000 | 0.1987 |
| `splash_and_splatter` | 0.1540 | 0.0408 | 0.0000 | 0.0000 | 0.1948 |
| `wind` | 0.0695 | 0.1154 | 0.0000 | 0.0000 | 0.1849 |
| `ocean` | 0.0357 | 0.1328 | 0.0000 | 0.0000 | 0.1686 |
| `wood` | 0.1305 | 0.0301 | 0.0000 | 0.0000 | 0.1606 |
| `stream` | 0.0128 | 0.1410 | 0.0000 | 0.0000 | 0.1539 |
| `toilet_flush` | 0.0150 | 0.1282 | 0.0000 | 0.0000 | 0.1432 |
| `waves_and_surf` | 0.0309 | 0.0845 | 0.0000 | 0.0000 | 0.1154 |
| `drip` | 0.0431 | 0.0666 | 0.0000 | 0.0000 | 0.1097 |
| `chirp_and_tweet` | 0.0406 | 0.0675 | 0.0000 | 0.0000 | 0.1081 |
| `bathtub_(filling_or_washing)` | 0.0274 | 0.0807 | 0.0000 | 0.0000 | 0.1081 |
| `pour` | 0.0204 | 0.0734 | 0.0000 | 0.0000 | 0.0938 |
| `trickle_and_dribble` | 0.0204 | 0.0734 | 0.0000 | 0.0000 | 0.0938 |
| `boat_and_water_vehicle` | 0.0192 | 0.0540 | 0.0000 | 0.0000 | 0.0732 |
| `whoosh_and_swoosh_and_swish` | 0.0647 | 0.0040 | 0.0000 | 0.0000 | 0.0687 |
| `fill_(with_liquid)` | 0.0277 | 0.0401 | 0.0000 | 0.0000 | 0.0677 |
| `gurgling` | 0.0285 | 0.0301 | 0.0000 | 0.0000 | 0.0586 |
| `raindrop` | 0.0309 | 0.0157 | 0.0000 | 0.0000 | 0.0466 |
| `boiling` | 0.0045 | 0.0389 | 0.0000 | 0.0000 | 0.0435 |

#### Per-tag: clips fully discarded (0 LMDB audio)

This table counts **clips** (not hours) that contribute **nothing** to the LMDB for each tag—every second of that clip is lost. With concat off, this is usually all clips shorter than one row; with concat on, it should be small unless packing fails for the final group.

- **Tag** — Whitelist label.
- **Clips fully discarded** — Number of clips with this tag that have **zero** stored audio after preprocess.

| Tag | Clips fully discarded |
| --- | ---: |
| `glass` | 663 |
| `wild_animals` | 593 |
| `liquid` | 519 |
| `bird` | 486 |
| `mechanisms` | 444 |
| `water` | 347 |
| `splash_and_splatter` | 258 |
| `wood` | 218 |
| `whoosh_and_swoosh_and_swish` | 216 |
| `bird_vocalization_and_bird_call_and_bird_song` | 153 |
| `drip` | 109 |
| `rain` | 97 |
| `wind` | 86 |
| `raindrop` | 81 |
| `chirp_and_tweet` | 79 |
| `gurgling` | 54 |
| `sink_(filling_or_washing)` | 53 |
| `ocean` | 37 |
| `pour` | 31 |
| `trickle_and_dribble` | 31 |
| `waves_and_surf` | 30 |
| `bathtub_(filling_or_washing)` | 27 |
| `thunder` | 26 |
| `thunderstorm` | 26 |
| `boat_and_water_vehicle` | 24 |
| `water_tap_and_faucet` | 24 |
| `fill_(with_liquid)` | 23 |
| `toilet_flush` | 12 |
| `stream` | 10 |
| `boiling` | 5 |

