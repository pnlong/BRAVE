# FSD50K exploration · BRAVE training

Utilities for auditing **[FSD50K](https://annotator.freesound.org/fsd/release/fsd50k/)** class labels via the official **development / evaluation** release, assembling a whitelist-based WAV pool, and preprocessing it into a RAVE-compatible database for [BRAVE](https://github.com/fcaspe/BRAVE).

FSD50K is a clip-level weak-label corpus: **development** clips *may* omit some true labels but are accurate when present; the **evaluation** split is labelled exhaustively for the ontology subset used here. Canonical train/validation rows live in ``dev.csv`` (`split`), under the **development** audio tree; evaluation clips ship in separate folders (see §2).

If you cite the dataset:

```bibtex
@article{fonseca2022FSD50K,
  title={{FSD50K}: an open dataset of human-labeled sound events},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={829--852},
  year={2022},
  publisher={IEEE}
}
```

## 1. Environment (conda / micromamba)

Create the **`brave`** env from the repo root (shared with the rest of [BRAVE](../README.md)):

```bash
cd BRAVE
micromamba env create -f environment.yaml   # or: conda env create -f environment.yaml
micromamba activate brave                   # or: conda activate brave
```

Packages mirror the upstream BRAVE install (`pip install h5py acids-rave==2.3`; see [../README.md](../README.md)). `ffmpeg` is installed via conda-forge via [`environment.yaml`](../environment.yaml).

### GPU tip

If `pip` installs a CPU-only PyTorch and you need CUDA on the cluster, install matching `pytorch` / `pytorch-cuda` from the [official PyTorch index](https://pytorch.org/) into this env first, then `pip check`.

## 2. Storage layout · dataset root

**Lab convention:** persistent large files live under **`BRAVE_STORAGE`**, defaults to **`/deepfreeze/pnlong/hai_lab/BRAVE`**.

Unpack the official archive so the tree looks like:

```text
{BRAVE_STORAGE}/FSD50K/
├── FSD50K.dev_audio/      # WAV for all development clips (~40 966 stems)
├── FSD50K.eval_audio/     # WAV for evaluation clips (~10 231 stems)
├── FSD50K.ground_truth/
│   ├── dev.csv            # fname, labels, mids, split  (train | val columns)
│   └── eval.csv           # fname, labels, mids
├── FSD50K.metadata/
└── FSD50K.doc/
```

To point scripts at another mount, export:

```bash
export BRAVE_STORAGE=/path/to/parent/of/FSD50K   # omit trailing slash quirks
```

or pass `--dataset-root /path/to/FSD50K` to `count_tags.py` / `build_subset.py`.

**Writable artefacts** default to **`{BRAVE_STORAGE}/fsd50k_brave/`** (`paths.DATA_ROOT`): WAV staging (**symlinks or copies**, see §5), `rave preprocess` outputs, local notes.

Suggested layout:

```bash
mkdir -p "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/"{audio_subset,preprocessed,artifacts}
```

| Path | Purpose |
|------|---------|
| `fsd50k_brave/audio_subset/` | **`build_subset`** default output (`paths.AUDIO_SUBSET_DIR`): flat **`*.wav` farm** (symlinks or copies) consumed by **`rave preprocess`** |
| `fsd50k_brave/preprocessed/` | Output of **`rave preprocess`** — pass as **`--db_path`** |
| `fsd50k_brave/artifacts/` | Optional — `tag_frequencies.tsv`, whitelists, logs |

Partitions are defined in [`paths.py`](paths.py):

| Canonical | Official role | Audio dir | Rows |
|-----------|---------------|-----------|------|
| `dev_train` | Development · train rows | `FSD50K.dev_audio` | `dev.csv`, `split=train` |
| `dev_val` | Development · validation rows | `FSD50K.dev_audio` | `dev.csv`, `split=val` |
| `eval` | Evaluation set | `FSD50K.eval_audio` | all of `eval.csv` |

**Splits still exist.** The official **development** release is partitioned into **`train`** and **`val`** rows in **`dev.csv`** (clips live under **`FSD50K.dev_audio`**). The **evaluation** clips are a separate corpus: **`eval.csv`** + **`FSD50K.eval_audio`**.

The canonical flags **`dev_train`**, **`dev_val`**, **`eval`** mirror that layout. Scripts also accept the synonyms **`train`**, **`valid`**, **`test`** (same mapping as in [`paths.py`](paths.py)). In writing, **`eval`** is clearer than **`test`** for the evaluation corpus—there is only one official held-out set.

When **training** BRAVE subsets, omit **`eval`** from your preprocessor pool unless you deliberately include it (evaluation is normally held out for benchmarking).

Tokens in `labels` CSV cells are **ontology class strings** (`Electric_guitar,…`). Whitelists match the **strip+lowercase** form (e.g. `electric_guitar`). Inspect `{BRAVE_STORAGE}/FSD50K/FSD50K.ground_truth/vocabulary.csv` on disk for authoritative spellings (not shipped in-tree).

## 3. Tag mining

Aggregate **per-label occurrences** among clips matching the manifest (each comma-separated token in `labels` counts once per clip):

```bash
cd BRAVE/fsd50k_exploration
mkdir -p artifacts
python3 count_tags.py > artifacts/tag_frequencies.tsv 2> artifacts/count_tags.log

# Narrow to splits:
python3 count_tags.py --partition dev_train --partition dev_val

# Sanity cap **per canonical partition**
python3 count_tags.py --partition dev_train --limit 500
```

Stdout is TAB-separated **`tag` \t `count`**, descending by count. Diagnostics (`# partitions=`, `# clips_seen=` …) land on stderr.

While scans run, **`tqdm`** renders a clip progress bar on **stderr** (stdout stays clean for `>` redirection). Use **`--no-progress`** when you capture stderr (`2>`) without bar noise.

Parsing lives in [`fsd50k_manifest.py`](fsd50k_manifest.py).

## 4. Whitelist

Plain UTF-8: **one strip+lower ontology token per non-empty line** (comments are **not** supported — any non-empty line is parsed as a label token).

A clip **`id`** is eligible when **any** normalized CSV label token intersects the whitelist (**exact equality**).

Sample whitelist file (`artifacts/my_whitelist.txt`):

```text
electric_guitar
coin_(dropping)
domestic_sounds_and_home_sounds
```

## 5. Building the subset (staging WAVs)

By default the subset pulls **development audio only** from **`FSD50K.dev_audio`**: **`--partition` defaults to `dev_train`**, so WAVs staged into `fsd50k_brave/audio_subset/` (`paths.AUDIO_SUBSET_DIR`) come from the **training** rows of **`dev.csv`**. Use **`dev_val`** for validation rows—still entirely under **`FSD50K.dev_audio`**.

Staging uses **symlinks** by default (**`--method symlink`**) so little extra disk is used on POSIX-friendly storage. Many **SMB/NAS mounts (including some deepfreeze setups)** disallow or mishandle symlink creation—in that case use **`--method copy`** to **`shutil.copy2`** each matching **`*.wav`** into the output tree (**much more disk**; slower first pass; **`rave preprocess`** then reads ordinary files).

**Evaluation WAVs (`FSD50K.eval_audio`) are never mixed in unless you set `--partition eval`** (synonym **`test`**).

Staging overlaps filesystem work when **`--workers` > 1** (each process symlink/copies WAVs concurrently). **`--workers`** defaults to **1**, i.e. no `ProcessPoolExecutor` (**single‑process streaming**, easier on quotas / flaky NAS). Raise workers if your storage tolerates concurrent writes.

Subset creation shows **`tqdm`**: manifest scan stats, then (when **`--workers`** > 1) per-file staging progress—use **`--no-progress`** to silence bars.

```bash
python3 build_subset.py --whitelist artifacts/my_whitelist.txt --overwrite

python3 build_subset.py --whitelist artifacts/my_whitelist.txt --method copy --overwrite

# Override split / locations (still dev_audio unless you switch partition to eval):
python3 build_subset.py --whitelist artifacts/my_whitelist.txt \
  --partition dev_train \
  --dataset-root "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/FSD50K" \
  --output-dir "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset" \
  --method copy \
  --overwrite

# Optionally include evaluation audio instead (or rerun with a dedicated output-dir):
python3 build_subset.py --whitelist artifacts/my_whitelist.txt \
  --partition eval \
  --output-dir "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/eval_audio_subset" \
  --method copy \
  --overwrite
```

## 6. Preprocess + train (`db_path`)

The preprocessed database path is chosen **only** on **`rave train --db_path`**; Gin configs do not bake filesystem paths.

Use [`configs/brave.gin`](../configs/brave.gin) (44.1 kHz, mono matches official releases).

### Regular vs lazy (`rave preprocess`)

As in [RAVE dataset preparation](https://github.com/acids-ircam/RAVE?tab=readme-ov-file#dataset-preparation), you can prepare a database two ways:

- **Regular (default)** — Omit **`--lazy`**. Audio is decoded and written into the preprocessed LMDB bundle up front (**`rave preprocess`**). Training then reads material that is already unpacked in the dataset—this repo’s **`build_subset`** WAV farm (**symlinks or copied files**) matches this pattern.
- **Lazy** — Add **`--lazy`**. **RAVE** trains **directly on the originals** (**mp3**, **ogg**, and other **`--ext`** formats) without converting the whole corpus to PCM on disk first. That saves space when a large library would **not fit uncompressed**, but **`RAVE`** warns **lazy dataset loading sharply increases CPU use during training**, especially **on Windows**.

```bash
rave preprocess --input_path /audio/folder --output_path /dataset/path
rave preprocess --input_path /audio/folder --output_path /dataset/path --lazy
```

### CPUs / workers (`rave preprocess`)

Upstream **`acids-rave`** ships **`scripts/preprocess.py`** inside your Python env. Stock **2.3** builds **`multiprocessing.Pool()`** with **`processes=None`**, so Python uses about **`multiprocessing.cpu_count()`** workers unless you edit the file.

**Local patch (this workspace / lab):** **`rave preprocess`** in the **`brave`** micromamba env was extended with an AbSL **`--workers`** flag and a matching **`Pool(processes=…)`** line so you can cap parallelism from the CLI (**`0`** = keep the default “use all logical CPUs” behaviour). The edited file lives at:

**`~/micromamba/envs/brave/lib/python3.11/site-packages/scripts/preprocess.py`**

(Verify with **`python -c "import scripts.preprocess as m; print(m.__file__)"`** in the **`brave`** env.) **Re-apply or re-diff this edit after reinstalling / upgrading `acids-rave`** — the package overwrites **`site-packages`** on upgrade.

Each worker **`Popen`**s **`ffmpeg`** / **`ffprobe`**, and **`flatmap(...)`** spins up a **`multiprocessing.Manager()`** queue as well, so the process tree stays busy. **`ps aux | grep 'rave preprocess' | wc -l`** often over-counts (**`grep`**, **`ffmpeg`** rows, multiprocessing helpers, etc.). Prefer **`pstree`** around the preprocess PID.

Important:

- **`export OMP_NUM_THREADS=1`** (and **`MKL_NUM_THREADS`**, **`OPENBLAS_NUM_THREADS`**, …) only caps **threads used by BLAS/OpenMP-linked code inside each process**. It **does not** replace **`--workers`** for limiting **pool process count**.
- **`taskset -c`** can pin **which CPUs** the lineage uses; without changing **`Pool(processes)`** you may still fork **many** workers that contend on fewer cores.

**If you lose the patch** (fresh env / upgrade), you can again edit **`preprocess.py`** as above, or hard-code **`Pool(processes=8)`** (or **`max(1, int(cpu_count() / 8))`**) in that same file.

Keeping the **`OMP_*` exports** remains useful so each worker stays **thin** once **`--workers`** is sensible.

### If you see `TypeError('cannot unpack non-iterable NoneType object')` or tqdm `0it`

That message is printed when a worker hits a file **`ffprobe`/`ffmpeg` cannot read**. In preprocessing, **`get_audio_channels` returns `None`** and the unpack in **`load_audio_chunk`** fails—a **broken symlink** (when inputs were symlink-staged), **missing copy**, or corrupt header typically causes this. Spot-check (**`realpath`** / **`ffprobe -hide_banner`** on random stems). Fixing or removing bad entries and re-running preprocess resolves it.

**Ctrl+C while the pool runs** often prints noisy **`KeyboardInterrupt`** / **`ImportError: sys.meta_path is None`** during multiprocessing teardown; that reflects interrupted worker shutdown, not your dataset layout.

From the **BRAVE** repo root (`cd …/BRAVE`):

```bash
# Optional: keep math libs from spawning many threads inside each multiprocessing worker
# (does NOT reduce how many Pool workers rave preprocess forks).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

rave preprocess \
  --input_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/audio_subset" \
  --output_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed" \
  --channels 1
# Omit `--lazy` for regular WAV → LMDB decoding (usual for this FSD50K flow). Append `--lazy` to train directly from originals (see §6 “Regular vs lazy”).

rave train \
  --config ./configs/brave.gin \
  --name fsd50k_texture_run \
  --db_path "${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}/fsd50k_brave/preprocessed"
```

## 7. Export (Minifusion)

After checkpoints exist:

```bash
python ./scripts/export_brave_plugin.py --model path/to/epoch_XXXXX.ckpt --output_path ./exported_model.h5
```

Runs from the repo root documented in [BRAVE README](../README.md).

## 8. Note: sustain via reverb (Logic Pro · Minifusion)

Listening-only appendix — unrelated to datasets or ML.

Logic **channel strip inserts run top → bottom**. Audition sustained material upstream of Minifusion by placing **reverb inserts above** Minifusion on the strip.

Considerations:

- Reverbs introduce **latency** vs a dry realtime chain.
- Uneven wet amounts need **parallel sends / blending / automation**.

---

### Optional loudness tooling

[`loud_tool.py`](../evaluation/scripts/loud_tool.py) can aggregate loudness scans before preprocessing if you normalize levels manually.
