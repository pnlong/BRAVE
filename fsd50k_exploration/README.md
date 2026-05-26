# FSD50K → BRAVE (Hai lab)

Turn **official [FSD50K](https://annotator.freesound.org/fsd/release/fsd50k/)** (development set with **`dev.csv`** **`train`**/**`val`** rows, plus **`eval.csv`**) into a whitelist subset, **`rave preprocess`** LMDB, and **[BRAVE](https://github.com/fcaspe/BRAVE)** training. Helpers live in **`paths.py`**, **`fsd50k_manifest.py`**, **`count_tags.py`**, **`build_subset.py`**, **`subset_audio_stats.py`**.

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

Details: [BRAVE README](../README.md). For CUDA PyTorch inside this env, use the [PyTorch install picker](https://pytorch.org/) if needed.

**Data scale:** **`BRAVE_STORAGE`** points at large disks (default **`/deepfreeze/pnlong/hai_lab/BRAVE`**). Scripts read **`$BRAVE_STORAGE/FSD50K`** (override with **`export BRAVE_STORAGE=...`** or **`--dataset-root`**).

---

## Paths (quick reference)

Official tree (unpack the MTG release under **`$BRAVE_STORAGE/FSD50K/`**):

`FSD50K.dev_audio/`, **`FSD50K.ground_truth/`** (**`dev.csv`**, **`eval.csv`**), `FSD50K.eval_audio/`, metadata/doc.

**Writable project dir** **`$BRAVE_STORAGE/fsd50k_brave/`**:

| Dir | Role |
|-----|------|
| **`audio_subset/`** | `build_subset` output → **`rave preprocess --input_path`** (`paths.AUDIO_SUBSET_DIR`) |
| **`preprocessed/`** | LMDB → **`rave train --db_path`** |
| **`artifacts/`** | Logs, **`tag_frequencies.tsv`**, whitelists |

```bash
mkdir -p "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/"{audio_subset,preprocessed,artifacts}
```

**Partitions (`--partition`):** **`dev_train`**, **`dev_val`**, **`eval`** (synonyms **`train`**, **`valid`**, **`test`**). **`eval`** = held-out corpus—omit from train pools unless intentional. Tokens in **`labels`** CSV cells are ontology names; whitelist lines are **strip + lowercase** (see **`…/ground_truth/vocabulary.csv`**).

---

## 1. Mine tags

```bash
cd BRAVE/fsd50k_exploration
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

Training-time notes in the script output are distilled from **`acids-rave`**: causal conv padding (**`configs/brave.gin`** → **`cc.get_padding.mode = 'causal'`**), **`valid_signal_crop = False`**, **`RandomCrop`** on top of preprocess, default **`131072`** samples at **44100** Hz (**`≈ 2.97` s**) for both **`rave preprocess --num_signal`** and **`rave train --n_signal`**. Override **`--sample-rate`** / **`--n-signal`** if your run differs.

```bash
python3 subset_audio_stats.py --whitelist artifacts/whitelists/my_whitelist.txt
# Only staged copies (recommended after ``build_subset``):
python3 subset_audio_stats.py --whitelist artifacts/whitelists/my_whitelist.txt \
  --wav-root "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset"

python3 subset_audio_stats.py --help   # parallelism, eval split, optional TSV export
```

---

## 3. Preprocess + train

Run from **`BRAVE/`**. Config: **`configs/brave.gin`** (44.1 kHz mono aligns with official WAV).

```bash
rave preprocess \
  --input_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset" \
  --output_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed" \
  --channels 1

rave train \
  --config ./configs/brave.gin \
  --name my_run \
  --db_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed"
```

**`rave preprocess`** decodes WAV → LMDB; add **`--lazy`** only if you want RAVE-style on-the-fly decode from originals (see [RAVE README](https://github.com/acids-ircam/RAVE?tab=readme-ov-file#dataset-preparation)).

**Parallelism.** Stock **`acids-rave`** forks about **logical CPU count** subprocesses for FFmpeg during preprocess unless limited. **`brave`** patches **`~/micromamba/envs/brave/lib/python3.11/site-packages/scripts/preprocess.py`** to add **`--workers`** (re‑apply whenever you reinstall/upgrade **`acids-rave`**; resolve with **`python -c 'import scripts.preprocess as m; print(m.__file__)'`**). **`rave train`** has its own **`--workers`** (PyTorch **`DataLoader` parallelism**, default **8**) plus **`--gpu`**. **`cannot unpack … NoneType`** in preprocess ⇒ **`ffmpeg`/`ffprobe`** failed on some staged WAV/path.

---

## 4. Export plugin

From **`BRAVE/`**: [main README](../README.md) → **`scripts/export_brave_plugin.py`** (Minifusion). Optional level scans: **`evaluation/scripts/loud_tool.py`**.
