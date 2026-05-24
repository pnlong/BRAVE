# FSD50K exploration · BRAVE training

Utilities and notes for auditing FSD50K tags under the IRCAM graft tree, assembling a whitelist-based training subset from **train**, and preprocessing it into a RAVE-compatible database for [BRAVE](https://github.com/fcaspe/BRAVE).

## 1. Environment (micromamba)

```bash
cd BRAVE/fsd50k_exploration
micromamba env create -f environment.yaml
micromamba activate brave-fsd50k
```

Packages mirror the upstream BRAVE install (`pip install h5py acids-rave==2.3`; see [../README.md](../README.md)). `ffmpeg` is installed via conda-forge here.

### GPU tip

If `pip` installs a CPU-only PyTorch and you need CUDA on the cluster, install matching `pytorch` / `pytorch-cuda` from the [official PyTorch index](https://pytorch.org/) into this env first, then `pip check`.

## 2. Paths

### Read-only (graft) FSD partitions

Declared in [`paths.py`](paths.py):

| Partition | Path |
|-----------|------|
| `train` | `/graft1/datasets/kechen/fsd50k/fsd50k/train/mnt/audio_clip/processed_datasets/FSD50K/train` |
| `valid` | `/graft1/datasets/kechen/fsd50k/fsd50k/valid/mnt/audio_clip/processed_datasets/FSD50K/valid` |
| `test` | `/graft1/datasets/kechen/fsd50k/fsd50k/test/mnt/audio_clip/processed_datasets/FSD50K/test` |

For **training**, build the FLAC pool from **`train`** only unless you knowingly pool partitions (holding out `valid`/`test` preserves cleaner evaluation splits).

### Writable layout (Arrakis storage)

Suggested base (once per machine):

```bash
mkdir -p /mnt/arrakis_data/pnlong/fsd50k_brave/{train_audio_symlinks,preprocessed,artifacts}
```

| Path under `fsd50k_brave` | Purpose |
|---------------------------|---------|
| `train_audio_symlinks/` | Output of [`build_subset.py`](build_subset.py)—flat symlink farm into graft `*.flac` files |
| `preprocessed/` | Output of **`rave preprocess`** — this folder is **`--db_path`** for `rave train` |
| `artifacts/` | Optional—save `tag_frequencies.tsv`, whitelist copies, notes |

[`paths.py`](paths.py) exposes `DATA_ROOT` and helpers so defaults match the layout above.

## 3. Tag mining

Produce two columns per line (**tag**, then TAB, then **count**), lowercase tags, descending by count:

```bash
cd BRAVE/fsd50k_exploration
mkdir -p artifacts
python count_tags.py > artifacts/tag_frequencies.tsv 2> artifacts/count_tags.log

# Narrow to partitions:
python count_tags.py --partition train --partition valid

# Sanity cap per partition:
python count_tags.py --partition train --limit 500
```

Stdout prints TAB-separated **`tag`** and **`count`** columns sorted by descending count. Scan stats (`# partitions= …`) land on stderr—capture with `2>…` alongside the redirects above.

Tag extraction prefers `original_data["all_tags"]` and falls back to top-level `"all_tags"`, `"tag"`, or `"text"` lists (same logic used by subset building)—see [`tag_utils.py`](tag_utils.py).

## 4. Whitelist

Plain UTF-8 file: **one strip+lowercase tag token per non-empty line** (authors follow that convention; scripts still lowercase/strip lines on load).

A clip **`id`** is eligible when **any** normalized tag from its JSON intersects the whitelist (**exact equality**—no substring rules in v1).

## 5. Building the subset (symlinks)

Write symlinks pointing from `/mnt/.../train_audio_symlinks/<id>.flac` back to graft sources:

```bash
python build_subset.py --whitelist artifacts/my_whitelist.txt --overwrite

# Overrides (defaults: train graft + symlink pool dir from paths.DATA_ROOT):
python build_subset.py --whitelist artifacts/my_whitelist.txt \
    --source /graft1/datasets/kechen/fsd50k/fsd50k/train/mnt/audio_clip/processed_datasets/FSD50K/train \
    --output-dir /mnt/arrakis_data/pnlong/fsd50k_brave/train_audio_symlinks \
    --overwrite
```

See [`example_whitelist.txt`](example_whitelist.txt) for formatting.

If `rave preprocess` ever refuses symlinked FLACs on your stack, rerun with copies instead (not automated here).

## 6. Preprocess + train (`db_path`)

The preprocessed dataset path is chosen **only** on **`rave train --db_path`**; Gin never embeds filesystem paths for the database.

Use [`configs/brave.gin`](../configs/brave.gin) like any other BRAVE run (44.1 kHz). Duplicate `*_fsd50k*.gin` configs are unnecessary unless you want different architecture hyperparameters (capacity, warmup, …).

From the **BRAVE** repo root (`cd .../BRAVE`):

```bash
rave preprocess \
  --input_path /mnt/arrakis_data/pnlong/fsd50k_brave/train_audio_symlinks \
  --output_path /mnt/arrakis_data/pnlong/fsd50k_brave/preprocessed \
  --channels 1

rave train \
  --config ./configs/brave.gin \
  --name fsd50k_texture_run \
  --db_path /mnt/arrakis_data/pnlong/fsd50k_brave/preprocessed
```

Confirm preprocessing/resampling behaviour with your acids-rave build if source FLAC sample rates vary.

## 7. Export (Minifusion)

After checkpoints exist (typically long runs—defaults target ~1.5M-step schedules per upstream docs):

```bash
python ./scripts/export_brave_plugin.py --model path/to/epoch_XXXXX.ckpt --output_path ./exported_model.h5
```

Runs from the repo root documented in [BRAVE README](../README.md).

## 8. Note: sustain via reverb (Logic Pro · Minifusion)

This appendix is **signal routing for listening experiments only**—it does **not** change training datasets or ML code.

Logic **channel strip inserts run top → bottom**. To audition “more sustained” material before neural timbre shaping, instantiate **your reverb (Space Designer / ChromaVerb / third-party convolution)** **above** Minifusion so incoming audio hits reverb **before** the BRAVE realtime plugin receives it.

Considerations:

- Reverbs introduce **latency** and change **latency compensation / monitoring feel** versus a dry realtime chain; Eco/low-latency modes help playable monitoring.
- If you want **some notes dry and others soaked**, a lone serial FX chain cannot express that cleanly—use parallel **aux/bus sends**, **wet/dry blend**, automation, transient detection, split buses between percussive and sustained sources, etc.

---

### Optional loudness tooling

[Loud_tool.py](../evaluation/scripts/loud_tool.py) can scan FLAC/WAV aggregates before preprocessing if you normalize levels manually.
