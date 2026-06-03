# Tabla ISMIR 2021 → BRAVE (Hai lab)

Turn the **[4-way Tabla stroke dataset](https://zenodo.org/records/7110248)** (ISMIR 2021) into a RAVE LMDB and train **[BRAVE](https://github.com/fcaspe/BRAVE)**. Helpers: **`paths.py`**, **`sample_audio.py`**, **`audio_stats.py`**.

**Cite:**

```bibtex
@inproceedings{rohit2021fourway,
  title={Four-way Classification of Tabla Strokes with Models Adapted from Automatic Drum Transcription},
  author={Rohit, M. A. and Bhattacharjee, Amitrajit and Rao, Preeti},
  booktitle={Proc. ISMIR},
  year={2021}
}
```

License on Zenodo: **CC-BY 4.0**.

## Setup

```bash
cd BRAVE && micromamba activate brave
export BRAVE_STORAGE="${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}"
```

Details: [BRAVE README](../../README.md). Hub: [dataset_exploration](../README.md).

---

## Paths (quick reference)

| Location | Role |
|----------|------|
| `$BRAVE_STORAGE/tabla_ismir21/raw/` | Zenodo download + zip |
| `$BRAVE_STORAGE/tabla_ismir21/4way-tabla-ismir21-dataset/` | Unpacked tree (`train/`, `test/`) |
| `$BRAVE_STORAGE/tabla_ismir21/preprocessed/` | LMDB → `train.py --db_path` |
| `artifacts/listen_samples/` | Local listening samples (gitignored) |
| `artifacts/dataset_statistics/` | Saved `audio_stats` reports (optional) |

```bash
mkdir -p "${BRAVE_STORAGE}/tabla_ismir21/"{raw,preprocessed}
```

**Splits:** **`train`** (~1.25 h, solo + theka) for LMDB; **`test`** (~20 min, isolated accompaniment) for held-out listening/eval — do not mix test into preprocess unless intentional.

Stroke **classes** are subfolders under each split; paired **`.onsets`** files are ignored by BRAVE preprocess (full-file training).

---

## 1. Download (Zenodo)

```bash
cd "${BRAVE_STORAGE}/tabla_ismir21/raw"
zenodo_get 7110248 -o .    # or: zenodo_get 10.5281/zenodo.7110248 -o .
unzip -q 4way-tabla-ismir21-dataset.zip -d ..
```

See [zenodo_get](https://github.com/dvolgyes/zenodo_get) if your CLI differs (`-r`, `-g "*.zip"`, etc.).

---

## 2. Listen / inspect

```bash
cd BRAVE/dataset_exploration/tabla_ismir21

# Random clips (all classes or one folder)
python3 sample_audio.py --split train -n 8 --seed 42
python3 sample_audio.py --stroke-class <folder_name> --split train -n 12

# Durations + LMDB row estimate (matches brave.gin 44.1 kHz, n_signal=131072)
python3 audio_stats.py --split train
python3 audio_stats.py --split train --compare-concat
```

---

## 3. Preprocess + train

From **`BRAVE/`**:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/preprocess.py \
  --input_path="${BRAVE_STORAGE}/tabla_ismir21/4way-tabla-ismir21-dataset/train" \
  --output_path="${BRAVE_STORAGE}/tabla_ismir21/preprocessed" \
  --channels=1

wandb login   # once

python RAVE/scripts/train.py \
  --config=configs/brave.gin \
  --name=tabla_ismir21_brave \
  --db_path="${BRAVE_STORAGE}/tabla_ismir21/preprocessed"
```

**Notes:**

- Small corpus (~1.25 h train): consider lower `--max_steps`, RAVE `--augment` configs, and watch **`train/loss`** vs **`val/loss`** ([wandb guide](../../RAVE/docs/wandb_guide.md)).
- Training uses a **98/2 LMDB row split** (seed 42), not the dataset’s train/test folders.
- Adjust `--input_path` if your unzip layout differs (`paths.unpacked_root()`).

---

## 4. Export

From **`BRAVE/`**: [main README](../../README.md) → `scripts/export_brave_plugin.py` or `RAVE/scripts/export.py`.
