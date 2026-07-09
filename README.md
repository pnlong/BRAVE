<h1 align="center">Designing Neural Synthesizers for Low-Latency Interaction</h1>
<div align="center">
<h3>
    <a href="https://arxiv.org/abs/2503.11562" target="_blank">paper</a> - <a href="https://fcaspe.github.io/brave" target="_blank">audio examples</a> - <a href="https://github.com/jorshi/nas-eval" target="_blank">NAS evaluation package</a> - <a href="https://minifusion.live" target="_blank">low-latency plugin</a>
</h3>

</div>


This repo contains the official Pytorch implementaiton of **BRAVE** a low-latency audio variational autoencoder for instrumental performance. It also implements all of the [other models tested on the paper](https://github.com/fcaspe/BRAVE/tree/main/configs).

Check the [`evaluation`](./evaluation) directory for info on replicating the results of the paper.

Training and preprocessing use a **vendored copy of RAVE** in [`RAVE/`](./RAVE/). Run its scripts directly (no `acids-rave` PyPI package). Metrics are logged to **[Weights & Biases](https://wandb.ai)**.


## Install

```bash
git clone https://github.com/fcaspe/BRAVE
cd BRAVE
```

Install **PyTorch** and **torchaudio** for your CUDA stack first ([PyTorch install picker](https://pytorch.org/)), then create the conda env.

### Option A: Conda env from `environment.yaml` (recommended)

[`environment.yaml`](./environment.yaml) provides Python 3.11, **ffmpeg** (conda-forge), `h5py`, `tqdm`, and pip dependencies from [`RAVE/requirements.txt`](./RAVE/requirements.txt) (including `wandb`).

```bash
conda env create -f environment.yaml
conda activate brave
```

To refresh an existing env after the file changes: `conda env update -n brave -f environment.yaml --prune`.

### Option B: Manual install

```bash
conda install ffmpeg
pip install h5py tqdm
pip install -r RAVE/requirements.txt
```

### Weights & Biases

One-time login (or set `WANDB_API_KEY`):

```bash
wandb login
```

Training logs go to the project named by `--wandb_project` (default: `brave`). See [RAVE/docs/wandb_guide.md](./RAVE/docs/wandb_guide.md) for metric interpretation.


## Preparing Dataset

Preprocess audio with the vendored RAVE script. RAVE LMDB datasets work with BRAVE models. See also [RAVE dataset preparation](https://github.com/acids-ircam/RAVE?tab=readme-ov-file#dataset-preparation).

From the **BRAVE repo root**:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/preprocess.py \
  --input_path=/audio/folder \
  --output_path=/path/to/preprocessed/dataset/ \
  --channels=1
```

Use `--workers=N` to limit FFmpeg parallelism during preprocess.

Dataset-specific download, staging, and stats helpers live under [`dataset_exploration/`](./dataset_exploration/) (e.g. [FSD50K](./dataset_exploration/fsd50k/), [Tabla ISMIR 2021](./dataset_exploration/tabla_ismir21/)).


## Training

Train with a `.gin` config from [`configs/`](./configs/). Example — BRAVE:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/train.py \
  --config=configs/brave.gin \
  --name=my_brave_run \
  --db_path=/path/to/preprocessed/dataset/
```

Optional W&B flags: `--wandb_project`, `--wandb_entity`, `--wandb_offline`.

Checkpoints and run metadata are written under `--out_path` (default `runs/`). View scalars and validation audio on [wandb.ai](https://wandb.ai).


## Exporting BRAVE for Real-Time Inference

### Quick start — Max 9 (recommended)

One command exports a **bundle folder** with the nn~ model and a pre-wired `play.maxpat`:

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python scripts/export_model.py \
  --model runs/YOUR_RUN \
  --db_path /path/to/lmdb \
  --output_dir exports/YOUR_RUN
```

Copy `exports/YOUR_RUN/` to your Mac, then open `play.maxpat` in Max 9. See [`RAVE/rave/fader/export/README.md`](./RAVE/rave/fader/export/README.md#max-9--nn-bundles) for the one-time nn~ install and load checklist.

The export prints an `scp` one-liner when it finishes.

### Which export path?

| You trained… | Canonicalizer? | Where it runs | Command / output |
|--------------|----------------|---------------|------------------|
| BRAVE (no Fader) | — | Max nn~ | `export_model.py` → `model.ts` + `play.maxpat` |
| FaderRAVE | no | Max nn~ | same (auto-detected) |
| FaderRAVE | yes (`*canonicalizer.ckpt` in run dir) | Max nn~ | same (`--canonicalizer auto`, default) |
| FaderRAVE | yes | Python demo | `export_model.py --host ts` |
| BRAVE or Fader | — | Minifusion plugin | `export_model.py --host h5` → `model.h5` |

Fader attribute knobs are documented in [`docs/fader_host_controls.md`](./docs/fader_host_controls.md) (not required to get sound from `play.maxpat`).

Stage-1 canonicalizers (waveform vs latent) are documented in [`docs/canonicalizer/`](./docs/canonicalizer/README.md).

### Max load checklist

1. Copy the bundle folder to `~/Documents/Max 9/Packages/nn_tilde/models/`
2. Open `play.maxpat` from that folder
3. Enable audio at **44100 Hz** and turn on **ezdac~**

### Low-latency BRAVE Plugin (Minifusion)

The [Minifusion plugin](https://minifusion.live) can run BRAVE models at < 10 ms latency and low jitter (~3 ms).

```bash
python scripts/export_model.py \
  --model runs/YOUR_RUN \
  --host h5 \
  --output_dir exports/YOUR_RUN
```

Or directly:

```bash
python ./scripts/export_brave_plugin.py --model path/to/model_checkpoint.ckpt --output_path ./exported_model.h5
```

**NOTE:** BRAVE works better when run at its original sampling rate. For best results, make sure that you run the plugin **at the same sample rate as the data used to train it**.

### Advanced export scripts

[`scripts/export_model.py`](./scripts/export_model.py) is the default entry point. Lower-level scripts (called internally or for power users):

| Script | Model | Target |
|--------|-------|--------|
| [`RAVE/scripts/export.py`](./RAVE/scripts/export.py) | Vanilla 128-D RAVE | Max nn~ (no Fader attributes) |
| [`RAVE/scripts/export_fader_nn.py`](./RAVE/scripts/export_fader_nn.py) | FaderRAVE | Max nn~ with attribute knobs |
| [`RAVE/scripts/export_fader_ts.py`](./RAVE/scripts/export_fader_ts.py) | FaderRAVE | Plain TorchScript (`forward(x, attr)`) |
| [`scripts/export_brave_plugin.py`](./scripts/export_brave_plugin.py) | BRAVE / Fader | Minifusion HDF5 |

BRAVE is also compatible with [SuperCollider](https://github.com/victor-shepardson/rave-supercollider), IRCAM's [RAVE VST](https://forum.ircam.fr/projects/detail/rave-vst/), and other RAVE TorchScript hosts via [`RAVE/scripts/export.py`](./RAVE/scripts/export.py). These may show **higher latency** than the BRAVE plugin due to a different audio buffering strategy.

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/export.py --run=path/to/model_checkpoint.ckpt --streaming
```

**Do not use `export.py` for Fader models** — use `export_model.py` or `export_fader_nn.py` instead.


## Cite Us

If you find this work useful please consider citing our paper:

```bibtex
@article{caspe2025designing,
    title={{Designing Neural Synthesizers for Low-Latency Interaction}},
    author={Caspe, Franco and Shier, Jordie and Sandler, Mark and Saitis, Charis and McPherson, Andrew},
    journal={Journal of the Audio Engineering Society},
    year={2025}
}
```
