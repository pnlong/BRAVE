# JSUT → BRAVE (Hai lab)

Stage the **[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)** Japanese speech corpus into `$BRAVE_STORAGE/jsut/` and preprocess to a RAVE LMDB.

**Shared unpack (already on disk):** `/data/hai-res/shared/p1long/JSUT/jsut_ver1.1`  
(~10 h, 48 kHz mono; preprocess resamples to 44.1 kHz.)

Audio terms: academic / non-commercial research (see `LICENCE.txt` in the release). Do not redistribute the full corpus.

## Stage + preprocess

```bash
cd BRAVE
source scripts/load_env.sh

# Symlink all subsets → $BRAVE_STORAGE/jsut/audio_subset
bash dataset_exploration/jsut/stage_subset.sh

# Or only basic5000:
# SUBSETS=basic5000 bash dataset_exploration/jsut/stage_subset.sh

INPUT_PATH="${BRAVE_STORAGE}/jsut/audio_subset" \
OUTPUT_PATH="${BRAVE_STORAGE}/jsut/preprocessed" \
  sbatch scripts/preprocess.sbatch

# One shot:
# SUBMIT_PREPROCESS=1 bash dataset_exploration/jsut/stage_subset.sh
```

Override source tree: `JSUT_ROOT=/path/to/jsut_ver1.1`.

## Train

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
# After preprocess finishes — use your usual train.sbatch / foundation-Y recipe with:
#   DB_PATH=$BRAVE_STORAGE/jsut/preprocessed
#   OUT_PATH=$BRAVE_STORAGE/jsut/runs
#   NAME=jsut_foundation_y
```
