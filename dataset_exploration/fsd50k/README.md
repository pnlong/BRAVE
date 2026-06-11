# FSD50K → BRAVE (Hai lab)

Turn **official [FSD50K](https://annotator.freesound.org/fsd/release/fsd50k/)** (development set with **`dev.csv`** **`train`**/**`val`** rows, plus **`eval.csv`**) into a whitelist subset, vendored **`RAVE/scripts/preprocess.py`** LMDB, and **[BRAVE](https://github.com/fcaspe/BRAVE)** training. Helpers live in **`paths.py`**, **`fsd50k_manifest.py`**, **`count_tags.py`**, **`build_subset.py`**, **`subset_audio_stats.py`**, **`sample_tag_audio.py`**.

**Cite:**

```bibtex
@article{fonseca2022FSD50K,
  title={{FSD50K}: an open dataset of human-labeled sound events},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30}, pages={829--852}, year={2022}, publisher={IEEE}
}
```

## Setup

```bash
cd BRAVE && micromamba env create -f environment.yaml && micromamba activate brave
```

Details: [BRAVE README](../../README.md). Dataset hub: [dataset_exploration](../README.md). For CUDA PyTorch inside this env, use the [PyTorch install picker](https://pytorch.org/) if needed.

**Data scale:** **`BRAVE_STORAGE`** points at large disks (default **`/deepfreeze/pnlong/hai_lab/BRAVE`**). Scripts read **`$BRAVE_STORAGE/FSD50K`** (override with **`export BRAVE_STORAGE=...`** or **`--dataset-root`**).

---

## Paths (quick reference)

Official tree (unpack the MTG release under **`$BRAVE_STORAGE/FSD50K/`**):

`FSD50K.dev_audio/`, **`FSD50K.ground_truth/`** (**`dev.csv`**, **`eval.csv`**), `FSD50K.eval_audio/`, metadata/doc.

**Writable project dir** **`$BRAVE_STORAGE/fsd50k_brave/`**:

| Dir | Role |
|-----|------|
| **`audio_subset/`** | `build_subset` output → preprocess **`--input_path`** (`paths.AUDIO_SUBSET_DIR`) |
| **`preprocessed/`** | LMDB → train **`--db_path`** |
| **`artifacts/`** | Logs, **`tag_frequencies.tsv`**, whitelists |

```bash
mkdir -p "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/"{audio_subset,preprocessed,artifacts}
```

**Partitions (`--partition`):** **`dev_train`**, **`dev_val`**, **`eval`** (synonyms **`train`**, **`valid`**, **`test`**). **`eval`** = held-out corpus—omit from train pools unless intentional. Tokens in **`labels`** CSV cells are ontology names; whitelist lines are **strip + lowercase** (see **`…/ground_truth/vocabulary.csv`**).

---

## Listen by tag (copy samples locally)

**`sample_tag_audio.py`** — sample **N** random clips with a given ontology tag and **symlink** them (default) under `artifacts/listen_samples/` (gitignored). Use `--no-symlink` to copy. A `manifest.tsv` lists clip ids and labels.

```bash
cd BRAVE/dataset_exploration/fsd50k

python3 sample_tag_audio.py --tag water -n 8 --seed 42
python3 sample_tag_audio.py --tag rain -n 12 --partition dev_val --overwrite

# Staged subset only:
python3 sample_tag_audio.py --tag water -n 8 \
  --wav-root "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset"

python3 sample_tag_audio.py --help
```

Default output: `artifacts/listen_samples/<tag>_<partition>_n<N>_seed<S>/` with files `001_<clip_id>.wav`, …

---

## 1. Mine tags

```bash
cd BRAVE/dataset_exploration/fsd50k
mkdir -p artifacts
python3 count_tags.py > artifacts/tag_frequencies.tsv 2> artifacts/count_tags.log
python3 count_tags.py --partition dev_train --partition dev_val
python3 count_tags.py --partition dev_train --limit 500 --no-progress
```

**`count_tags`** uses **`tqdm`** on stderr (**`--no-progress`** disables it).

---

## 2. Whitelist + stage WAVs

Whitelist: UTF-8, **one ontology token per line** (normalized like CSV: **`electric_guitar`**). Clip kept if **any** label token equals a line (**exact match**).

```text
electric_guitar
domestic_sounds_and_home_sounds
```

**`build_subset`** defaults **`--partition dev_train`** and **`audio_subset/`**. **`--method copy`** avoids symlinks on NAS (**`symlink`** is cheaper on POSIX disks). **`--workers`** defaults to **1**. Progress uses **`tqdm`** on stderr; **`--no-progress`** disables it.

```bash
python3 build_subset.py --whitelist artifacts/whitelists/my_whitelist.txt --overwrite
python3 build_subset.py --whitelist artifacts/whitelists/my_whitelist.txt --method copy --overwrite

python3 build_subset.py --whitelist artifacts/whitelists/my_whitelist.txt --partition eval \
  --output-dir "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/eval_audio_subset" \
  --method copy --overwrite
```

---

## 2b. Subset durations + training chop stats

**`subset_audio_stats.py`** scans the **same** manifest ∩ whitelist ∩ `*.wav` paths as **`build_subset`**, runs **`ffprobe`** on each file, and prints descriptive stats (distribution of raw durations, clips shorter than one RAVE **`num_signal`** window, estimated LMDB row count after non-lazy preprocess, total hours).

It also totals **hours per whitelist tag**: each ontology token is treated like a separate “prompt” / class—the **full clip duration is added once per overlapping tag**, which matches intuition for “how many hours do we get when conditioning on label *X*?” on multi-labelled FSD50K rows.

Training-time notes in the script output match vendored RAVE: causal conv padding (**`configs/brave.gin`** → **`cc.get_padding.mode = 'causal'`**), **`valid_signal_crop = False`**, **`RandomCrop`** on top of preprocess, default **`131072`** samples at **44100** Hz (**`≈ 2.97` s**) for both preprocess **`--num_signal`** and train **`--n_signal`**. Override **`--sample-rate`** / **`--n-signal`** if your run differs.

```bash
python3 subset_audio_stats.py --whitelist artifacts/whitelists/my_whitelist.txt
# Only staged copies (recommended after ``build_subset``):
python3 subset_audio_stats.py --whitelist artifacts/whitelists/my_whitelist.txt \
  --wav-root "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset"

python3 subset_audio_stats.py --help   # parallelism, eval split, optional TSV export
```

---

## 3. Preprocess + train

Run from **`BRAVE/`**. Config: **`configs/brave.gin`** (44.1 kHz mono aligns with official WAV). Log in to W&B once (`wandb login`) before training.

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/preprocess.py \
  --input_path="${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset" \
  --output_path="${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed" \
  --channels=1

python RAVE/scripts/train.py \
  --config=configs/brave.gin \
  --name=my_run \
  --db_path="${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed"
```

Preprocess decodes WAV → LMDB; add **`--lazy`** only for on-the-fly decode from originals (see [RAVE README](../RAVE/README.md)).

**Parallelism.** Preprocess supports **`--workers`** (FFmpeg pool; default uses all CPUs). Train has **`--workers`** (PyTorch **`DataLoader`**, default **8**) and **`--gpu`**. Metrics go to Weights & Biases (`--wandb_project`, default **`brave`**). **`cannot unpack … NoneType`** in preprocess ⇒ **`ffmpeg`/`ffprobe`** failed on some staged WAV/path.

### Fader + `texture_class` (texture LMDB)

After preprocess, wire a 10-class discrete **`texture_class`** sidecar (classes 0–9; music/vocal 10–11 omitted by default). From **`BRAVE/`**:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/build_lmdb_index_manifest.py \
  --input_path="${BRAVE_STORAGE}/fsd50k_brave/texture/audio_subset" \
  --db_path="${BRAVE_STORAGE}/fsd50k_brave/texture/preprocessed" \
  --num_signal 131072

python RAVE/scripts/build_attribute_sidecar.py \
  --db_path="${BRAVE_STORAGE}/fsd50k_brave/texture/preprocessed" \
  --scheme texture_class --partition dev_train

python RAVE/scripts/precompute_descriptors.py \
  --db_path=.../texture/preprocessed \
  --continuous_attributes=rms --continuous_attributes=flatness \
  --continuous_attributes=centroid --continuous_attributes=roughness \
  --continuous_attributes=brightness \
  --discrete_attributes=texture_class --train_only

python RAVE/scripts/train.py --config configs/brave_fader_texture.gin \
  --db_path=.../texture/preprocessed --name texture_fader
```

Build a texture-only audio subset first (exclude `music`, `musical_instrument`, `speech`, etc.) so LMDB indices align with the sidecar. Tag taxonomy: [`configs/fader_texture_class_tags.yaml`](configs/fader_texture_class_tags.yaml). Validate mapping: `python dataset_exploration/fsd50k/validate_texture_class_tags.py`.

### Fader + `water_scene` (optional)

After preprocess, wire a 3-class discrete **`water_scene`** sidecar (storm / coastal / other). From **`BRAVE/`**:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/build_lmdb_index_manifest.py \
  --input_path="${BRAVE_STORAGE}/fsd50k_brave/water/audio_subset" \
  --db_path="${BRAVE_STORAGE}/fsd50k_brave/water/preprocessed" \
  --num_signal 131072

python RAVE/scripts/build_attribute_sidecar.py \
  --db_path="${BRAVE_STORAGE}/fsd50k_brave/water/preprocessed" \
  --scheme water_scene --partition dev_train

python RAVE/scripts/precompute_descriptors.py \
  --db_path=.../water/preprocessed --discrete_attributes water_scene

# Copy configs/brave_fader_fsd50k_water.gin.example → brave_fader_fsd50k_water.gin
python RAVE/scripts/train.py --config configs/brave_fader_fsd50k_water.gin \
  --db_path=.../water/preprocessed --name water_fader
```

Tag sets: [`configs/fader_water_scene_tags.yaml`](configs/fader_water_scene_tags.yaml), [`configs/fader_texture_class_tags.yaml`](configs/fader_texture_class_tags.yaml). Details: [`scratchpaper/fader_future_work.md`](../../scratchpaper/fader_future_work.md).

---

## 4. Export plugin

From **`BRAVE/`**: [main README](../../README.md) → **`scripts/export_brave_plugin.py`** (Minifusion). Optional level scans: **`evaluation/scripts/loud_tool.py`**.
