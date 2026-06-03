# Dataset exploration (Hai lab)

Helpers for turning external audio corpora into BRAVE / RAVE LMDB training pools. Each dataset lives in its own subfolder; large binaries and LMDBs stay under **`$BRAVE_STORAGE`** (see per-dataset `paths.py`).

| Dataset | Folder | Zenodo / source |
|--------|--------|-----------------|
| [FSD50K](https://annotator.freesound.org/fsd/release/fsd50k/) | [`fsd50k/`](fsd50k/) | MTG release (not on Zenodo) |
| [4-way Tabla ISMIR 2021](https://zenodo.org/records/7110248) | [`tabla_ismir21/`](tabla_ismir21/) | [10.5281/zenodo.7110248](https://doi.org/10.5281/zenodo.7110248) |

**Training** always uses vendored RAVE from the BRAVE root:

```bash
cd BRAVE
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
export BRAVE_STORAGE="${BRAVE_STORAGE:-/deepfreeze/pnlong/hai_lab/BRAVE}"
```

Shared gitignored artifacts: `*/artifacts/listen_samples/`, `*/artifacts/exported_models/` (see [`.gitignore`](../.gitignore)).
