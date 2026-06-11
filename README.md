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

### Low-latency BRAVE Plugin

The [Minifusion plugin](https://minifusion.live) can run BRAVE models at < 10 ms latency and low jitter (~3 ms).

Use the `export_brave_plugin.py` utility to export a trained model. This requires a BRAVE checkpoint (`.ckpt`) from training above.  
It does not work with models exported to TorchScript (`.ts`).

```bash
python ./scripts/export_brave_plugin.py --model path/to/model_checkpoint.ckpt --output_path ./exported_model.h5
```

**NOTE:** BRAVE works better when run at its original sampling rate. For best results, make sure that you run the plugin **at the same sample rate as the data used to train it**.

### Standard RAVE Export Method

BRAVE is compatible with many creative coding tools and plugins that use RAVE models. You can export a BRAVE model to work with some great tools created by the community, such as:

 - [nn~](https://github.com/acids-ircam/nn_tilde) for Max-MSP & PureData
 - [SuperCollider](https://github.com/victor-shepardson/rave-supercollider) UGen
 - IRCAM's [RAVE VST](https://forum.ircam.fr/projects/detail/rave-vst/)
 - And probably some more

Please note these might show **higher latency** than the BRAVE Plugin due to a different audio buffering strategy.

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"

python RAVE/scripts/export.py --run=path/to/model_checkpoint.ckpt
```

This stores a TorchScript `(.ts)` model next to the checkpoint file which you can load in your selected application. Use `--streaming` for realtime-safe cached convolutions (see [RAVE README](./RAVE/README.md)).

**Fader models (128+D with attribute knobs):** use [`RAVE/scripts/export_fader_nn.py`](./RAVE/scripts/export_fader_nn.py), not `export.py`. See [`docs/fader_host_controls.md`](./docs/fader_host_controls.md).


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
