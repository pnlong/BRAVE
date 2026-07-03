# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BRAVE is a PyTorch implementation of a low-latency audio variational autoencoder for instrumental performance. It extends the [RAVE](https://github.com/acids-ircam/RAVE) framework with causal convolutions and a smaller, faster architecture optimized for real-time inference (< 10 ms latency). The model is configured entirely via Gin config files.

Vendored RAVE lives in `RAVE/`. Run its scripts with `PYTHONPATH` set to that directory (scripts also insert the RAVE root on `sys.path`). Training logs to Weights & Biases.

## Setup

```bash
conda env create -f environment.yaml
conda activate brave
# Install PyTorch/torchaudio for your CUDA stack first if needed
wandb login
```

For evaluation only:
```bash
pip install frechet_audio_distance
pip install git+https://github.com/jorshi/neural-latency-eval
```

## Key Commands

From the BRAVE repo root:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
```

**Preprocess audio dataset:**
```bash
python RAVE/scripts/preprocess.py --input_path=/audio/folder --output_path=/path/to/dataset/ --channels=1
```

**Train a model** (specify a `.gin` config from `configs/`):
```bash
python RAVE/scripts/train.py --config=configs/brave.gin --name=my_run --db_path=/path/to/dataset/
```
Training runs for 1.5M steps; the checkpoint used for evaluation is `epoch_1500000.ckpt`.

**Export for realtime (Max nn~ bundle — recommended):**
```bash
python scripts/export_model.py \
  --model runs/my_run --db_path /path/to/lmdb \
  --output_dir exports/my_run
```
Writes `model.ts`, sidecars, and pre-wired `play.maxpat`. See [`RAVE/rave/fader/export/README.md`](RAVE/rave/fader/export/README.md#max-9--nn-bundles).

**Export for Minifusion plugin** (requires `.ckpt`, not `.ts`):
```bash
python scripts/export_model.py --model path/to/model.ckpt --host h5 --output_dir exports/my_run
# or: python ./scripts/export_brave_plugin.py --model path/to/model.ckpt --output_path ./exported_model.h5
```

## Fader Networks training

Fader training uses standalone `configs/brave_fader.gin` (includes base `brave.gin`). Providers live in `RAVE/rave/fader/providers/`. See [`scratchpaper/fader_future_work.md`](scratchpaper/fader_future_work.md) and [`docs/fader_host_controls.md`](docs/fader_host_controls.md).

**Precompute attribute stats (train split only by default):**
```bash
python RAVE/scripts/precompute_descriptors.py \
  --db_path /path/to/lmdb --n_signal 131072 --train_only
```

**Train FaderRAVE:**
```bash
python RAVE/scripts/train.py --name brave_fader_run \
  --config configs/brave_fader.gin \
  --db_path /path/to/lmdb --batch 8 --gpu -1
```

Writes `attribute_stats.yaml` beside the LMDB. Swap continuous/discrete attributes via gin lists + `attribute_sidecar.yaml` without editing dataset code.

**Fader inference (latent exploration):**
```bash
python latent_exploration/mask_reconstruct.py \
  --model runs/brave_fader_run --input clip.wav \
  --db-path /path/to/lmdb --attr-mode extract
```

**Eval attribute swap:**
```bash
python RAVE/scripts/eval_fader_attributes.py \
  --model runs/brave_fader_run --db_path /path/to/lmdb
```

**Prior on 128-D content z (after Fader training):**
```bash
python RAVE/scripts/train_prior.py --model runs/brave_fader_run --fader \
  --db_path /path/to/lmdb --name fader_prior
```

**Dataset exploration:** `dataset_exploration/fsd50k/`, `dataset_exploration/tabla_ismir21/` (see each README).

**FSD50k water + `water_scene` sidecar:**
```bash
python RAVE/scripts/build_lmdb_index_manifest.py --input_path .../audio_subset --db_path .../preprocessed
python RAVE/scripts/build_attribute_sidecar.py --db_path .../preprocessed --scheme water_scene
```
Use `configs/brave_fader_fsd50k_water.gin.example` (copy to `.gin`; includes `brave_fader.gin`).

**Export Fader for Max/nn~ (attribute knobs + play.maxpat):**
```bash
python scripts/export_model.py \
  --model runs/brave_fader_run --db_path /path/to/lmdb \
  --output_dir exports/brave_fader_run
```
Lower-level: `RAVE/scripts/export_fader_nn.py` (canonicalizer: `--canonicalizer auto`).

**Export Fader plain TorchScript (128+D concat, Python demos):**
```bash
python scripts/export_model.py --model runs/brave_fader_run --host ts \
  --db_path /path/to/lmdb --output_dir exports/brave_fader_run
```
Also writes `model_host_controls.json`. See [`docs/fader_host_controls.md`](docs/fader_host_controls.md).

**Subjective swap listening assets:**
```bash
python RAVE/scripts/generate_attribute_swap_pairs.py \
  --model runs/brave_fader_run --db_path /path/to/lmdb --output_dir listening/swap_pairs
```
See [`docs/fader_listening_protocol.md`](docs/fader_listening_protocol.md).

**Export to TorchScript** (for nn~, SuperCollider, RAVE VST):
```bash
python RAVE/scripts/export.py --run=path/to/model.ckpt
```

**Run resynthesis / timbre transfer:**
```bash
python RAVE/scripts/generate.py --model=path/to/model.ckpt --input=./experiments/test_audios/drumset
```

## Evaluation

All evaluation scripts live in `evaluation/scripts/` and assume you `cd evaluation` first.

**Fréchet Audio Distance:**
```bash
python ./scripts/fad.py ./experiments/fad/drumset.json
```
Edit the JSON to set `background_path` (train set) and `resynth_paths` (resynthesized output dirs).

**Latency measurement** (requires GPU):
```bash
python ./scripts/latency.py --model /path/to/checkpoint.ckpt --gpu 0 --name EXPERIMENT_ID
```

**Timbre transfer evaluation** (MMD via `nas_eval`):
```bash
nas-eval timbre ./experiments/test_audios/ ./reconstructions/ --matrix beatbox-drumset
```

**Fetch datasets** (run from `evaluation/`):
```bash
source scripts/generate_drumset.sh   # or generate_beatbox_test_files.sh, etc.
```

## Architecture

Model code is in the vendored `RAVE/rave/` package (`rave`, `cached_conv`). BRAVE configures models via `.gin` files in `configs/`.

All configs share the same structure:
- **PQMF filterbank** (`CachedPQMF`): splits audio into `N_BAND=16` subbands; `attenuation` controls filter steepness (40 dB for BRAVE, 100 dB for heavier models)
- **Variational Encoder** (`VariationalEncoder` wrapping `Encoder`): encodes subbands to a 128-dim latent
- **Generator** (`blocks.Generator`): decodes latent back to subbands via transposed convolutions with `ResidualStack` dilations
- **MultiScaleDiscriminator**: GAN training component, not exported to the plugin
- `cc.get_padding.mode = 'causal'` is set globally, making all convolutions causal (zero look-ahead)

**BRAVE vs other configs:** BRAVE uses `CAPACITY=32` (4.9M params) vs `CAPACITY=64` (15–18M params) for all other models. The ratios `[2,2,2,1]` give a receptive field of ~517 ms at 44.1 kHz. Lower PQMF attenuation (40 dB) reduces filter memory, further cutting latency.

**Export for Minifusion plugin** (`export_brave_plugin.py`): strips discriminator, audio distance, and padding/cache buffers from the state dict, then saves weights as HDF5. TorchScript exports are not supported for this plugin path.

## Important Notes

- Models must be trained and run at the **same sample rate as the training data** (default 44.1 kHz) for best results.
- The Filosax dataset requires manual download permission from Zenodo before evaluation scripts will work.
- `generate.py` writes output to `--out_path` (default `generations`).
