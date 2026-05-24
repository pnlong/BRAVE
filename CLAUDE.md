# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BRAVE is a PyTorch implementation of a low-latency audio variational autoencoder for instrumental performance. It extends the [RAVE](https://github.com/acids-ircam/RAVE) framework with causal convolutions and a smaller, faster architecture optimized for real-time inference (< 10 ms latency). The model is configured entirely via Gin config files.

## Setup

```bash
pip install h5py acids-rave==2.3
conda install ffmpeg
```

For evaluation only:
```bash
pip install frechet_audio_distance
pip install git+https://github.com/jorshi/neural-latency-eval
```

## Key Commands

**Preprocess audio dataset:**
```bash
rave preprocess --input_path /audio/folder --output_path path/to/dataset/ --channels X
```

**Train a model** (specify a `.gin` config from `configs/`):
```bash
rave train --config ./configs/brave.gin --name my_run --db_path path/to/dataset/
```
Training runs for 1.5M steps; the checkpoint used for evaluation is `epoch_1500000.ckpt`.

**Export for Minifusion plugin** (requires `.ckpt`, not `.ts`):
```bash
python ./scripts/export_brave_plugin.py --model path/to/model.ckpt --output_path ./exported_model.h5
```

**Export to TorchScript** (for nn~, SuperCollider, RAVE VST):
```bash
rave export --run path/to/model.ckpt
```

**Run resynthesis / timbre transfer:**
```bash
rave generate --model path/to/model.ckpt --input ./experiments/test_audios/drumset
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

The codebase contains no custom Python source — it relies entirely on the `acids-rave` package (`rave`, `cached_conv`) and configures models via `.gin` files in `configs/`.

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
- `rave generate` writes output next to the input directory by default; check RAVE docs for `--output` flag behavior.
