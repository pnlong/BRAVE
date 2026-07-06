# Latent exploration

Encode audio through a trained RAVE/BRAVE checkpoint, visualize latents, and experiment with latent masking before decode.

Run from the **BRAVE repo root** with the **`brave` conda/micromamba env activated**:

```bash
micromamba activate brave
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
```

On hai-res login nodes where AFS home is unreadable, set `HOME=/data/hai-res/$USER` before `micromamba activate` (see `scripts/micromamba_env.sh`).

## Reconstruction test (offline)

**Use this first** when Max/nn~ sounds wrong (buzzing, silence, level issues) and you need to know whether the checkpoint or the realtime patch is at fault.

**Process:** encode the input clip → decode (no mask) → write WAVs. All reconstruction tests go through **`mask_reconstruct.py`** with **`--mask-style none`** (or the thin wrapper **`visualize_latents.py`**, which calls the same code path).

| What you hear | Likely cause |
|---------------|--------------|
| Buzzing / bad audio in `{stem}_reconstructed.wav` | Model, checkpoint, or training data mismatch |
| Clean offline recon, bad audio only in Max | nn~ export, block streaming, patch wiring, or sample rate |

**Unconditional BRAVE** (no `--db-path`, no `--attr-mode`):

```bash
python latent_exploration/mask_reconstruct.py \
  --model /data/scratch-fast/p1long/BRAVE/yt_birdsong/runs/yt_birdsong_run_d8e2ae9d65/best.ckpt \
  --input /data/scratch-fast/p1long/BRAVE/tap_samples/audio_subset/0.wav \
  --output-dir /data/scratch-fast/p1long/BRAVE/yt_birdsong/recon_test_tap0 \
  --mask-style none \
  --latent-mode mean \
  --gpu
```

**Outputs** (in `--output-dir`, or `latent_exploration/artifacts/reconstructions/<stem>/` by default):

- `{stem}_original.wav` — resampled input at model rate (44.1 kHz)
- `{stem}_reconstructed.wav` — full-clip encode/decode (compare this to Max)
- `{stem}_latents.png`, `{stem}_latent_hist.png` — optional diagnostics (omit with `--no-plot`)

Listen to `_reconstructed.wav` in any player; no Max required. Use `--no-wavs` for plot-only runs.

Equivalent one-liner:

```bash
python latent_exploration/visualize_latents.py \
  --model /path/to/run_or_ckpt \
  --input /path/to/audio.wav \
  --output-dir /path/to/output \
  --gpu
```

For **Fader** checkpoints, add `--db-path /path/to/lmdb` (or `--stats-path`) and `--attr-mode extract` — see [FaderRAVE](#faderrave-attribute-controlled-decode) below.

Lower-level alternative (resynth only, no side-by-side original): `RAVE/scripts/generate.py --model=... --input=... --gpu=0`.

## Environment / troubleshooting

**Activate `brave` first** — the scripts need RAVE deps (`cached_conv`, `torch`, etc.) from that env:

```bash
micromamba activate brave
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
```

**`CXXABI_1.3.15` not found:** conda-built wheels (matplotlib, scipy, …) need the env's newer `libstdc++`, but Linux often loads the older system one first. One-time setup (writes hooks into the active env):

```bash
micromamba activate brave
bash latent_exploration/setup_ld_library_path.sh
micromamba deactivate && micromamba activate brave
```

Or set manually each session before running:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

**`numpy.dtype size changed`:** usually a broken mix of pip + conda numpy/scipy (e.g. after `mamba install matplotlib seaborn` upgraded numpy while scipy stayed on an old build). Reinstall pinned versions from conda-forge:

```bash
micromamba install -n brave numpy=1.26.4 scipy=1.10.0 seaborn matplotlib -c conda-forge
```

**`TorchCodec is required` (torchaudio load/save):** torchaudio 2.9+ delegates I/O to torchcodec. The latent exploration scripts use **soundfile** instead (already in the `brave` env via librosa). No torchcodec install needed for these scripts.

To use a GPU, set `CUDA_VISIBLE_DEVICES` and pass `--gpu` (PyTorch always uses the first visible device, `cuda:0`):

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Latent layout (BRAVE)

For `configs/brave.gin`: `LATENT_SIZE=128`, `RATIOS=[2,2,2,1]` → compression ratio **8**.

After encode + reparametrize, latents are **`[batch, latent_dim, time]` = `[B, 128, T_audio // 8]`**.

Plot convention:

- **x-axis** = temporal latent frames (columns)
- **y-axis** = latent dimensions (rows)
- **Temporal mask** → zero vertical strips (columns)
- **Latent-dim mask** → zero horizontal strips (rows)

## PCA fidelity — which of the 128 dims matter?

During validation (before phase-1 warmup ends), RAVE collects sampled latents from the validation set and treats every time frame as a point in **128-dimensional** VAE space.

1. **Center:** compute mean vector \(\mu \in \mathbb{R}^{128}\) over all validation frames.
2. **PCA:** on centered points \(z' = z - \mu\), find orthonormal directions (rows of `latent_pca`) that maximize variance. Component \(k=0\) captures the most variance, \(k=1\) the next most (orthogonal to prior), etc.
3. **Explained variance:** for each component \(k\), RAVE stores cumulative fraction
   \[
   \text{fidelity}[k] = \frac{\sum_{i=0}^{k} \lambda_i}{\sum_{j=0}^{127} \lambda_j}
   \]
   where \(\lambda_i\) are PCA eigenvalues (sklearn `explained_variance_`).

**`--pca-fidelity`** (default `0.95`) picks the smallest integer \(N\) such that \(\text{fidelity}[N-1] \ge 0.95\). The PCA plot shows only rows `0 … N-1` — the leading directions that together explain ≥95% of validation latent variance. The remaining \(128-N\) rows are not discarded from the model; they carry the leftover ~5% variance and are what export can truncate (with noise fill on decode).

This is the same criterion as `RAVE/scripts/export.py --fidelity` (export additionally rounds \(N\) up to a power of two for realtime latent width).

## Scripts

Beyond the [reconstruction test](#reconstruction-test-offline) workflow above, all encode/decode paths go through **`mask_reconstruct.py`** (`run_reconstruction`). By default each run saves:

- `{stem}_original.wav`
- `{stem}_reconstructed.wav`
- `{stem}_latents.png` (no mask) or `{stem}_mask_plot.png` (with mask)

Use `--no-wavs` or `--no-plot` to skip either output.

Each run prints **latent value distributions** to stdout (min/max/mean/std and percentiles for VAE and PCA tensors used for decode/plot).

With **`--plot`** (default on `visualize_latents` / `mask_reconstruct`), also saves:

- `{stem}_latents.png` or `{stem}_mask_plot.png` — mel + latent heatmaps
- `{stem}_latent_hist.png` — VAE and PCA histograms (x-axis uses `--clip-percentile`, default p2–p98)

Use **`--clip-outliers`** to set heatmap color limits from percentiles instead of min/max (default range `--clip-percentile 2 98`), so extreme values don't wash out the plot. Values outside the range still render at the colorbar ends.

### Visualize latents

Thin wrapper for **reconstruction test** (`--mask-style none`); same as `mask_reconstruct.py` above. Default output under `artifacts/plots/<stem>/`.

```bash
python latent_exploration/visualize_latents.py \
  --model /path/to/run_or_ckpt \
  --input tap_samples/0.wav \
  --latent-mode mean \
  --pca-fidelity 0.95 \
  --gpu
```

Pass `--no-wavs` for plot-only. Same as:

```bash
python latent_exploration/mask_reconstruct.py \
  --model ... --input ... --mask-style none \
  --output-dir latent_exploration/artifacts/plots/<stem>
```

### Mask and reconstruct

```bash
python latent_exploration/mask_reconstruct.py \
  --model /path/to/run_or_ckpt \
  --input tap_samples/0.wav \
  --mask-style temporal \
  --mask-start 20 --mask-width 30 \
  --latent-mode mean \
  --pca-fidelity 0.95 \
  --gpu
```

Mask styles:

| `--mask-style` | Effect |
|----------------|--------|
| `none` (default) | all ones — identity pass-through |
| `temporal` | zero columns `[start : start+width)` (time frames) |
| `latent` | zero rows `[start : start+width)` — VAE dims or PCA components (see `--mask-space`) |

### Mask space (`--mask-space`)

| Value | Mask applied | `latent` style zeros | Affects WAV |
|-------|----------------|----------------------|-------------|
| `vae` (default) | before decode, in encoder latent space | VAE channels 0–127 | yes |
| `pca` | rotate → mask → inverse PCA → decode | PCA components (row 0 = most variance) | yes |

Example — zero PCA components 80–99, then reconstruct:

```bash
python latent_exploration/mask_reconstruct.py \
  --model /path/to/checkpoint \
  --input tap_samples/0.wav \
  --mask-space pca \
  --mask-style latent \
  --mask-start 80 --mask-width 20 \
  --latent-mode mean
```

## Latent mode

- **`mean`** (default): deterministic VAE mean — reproducible plots and reconstructions.
- **`sample`**: stochastic reparametrize sample — matches default `model.forward()`.

## Dependencies

Requires `seaborn` (see `environment.yaml`). Mel spectrograms use `torchaudio`; plots use seaborn heatmaps; WAV I/O uses soundfile.

## RAVE model hook

`RAVE/rave/model.py` exposes optional latent masking **after reparametrize**, before decode:

- `encode_to_latent(x, use_mean=False)`
- `set_latent_mask(mask)` / `clear_latent_mask()`
- `reconstruct_from_latent(z)`
- `forward_with_mask(x, mask=None)`

Training paths (`forward`, `training_step`) are unchanged; default behavior is identity (no mask set).

## FaderRAVE (attribute-controlled decode)

When the checkpoint was trained with `configs/brave_fader.gin`, use `load_model()` (auto-detects FaderRAVE) and pass attribute stats:

```bash
python latent_exploration/mask_reconstruct.py \
  --model runs/brave_fader_run \
  --input tap_samples/0.wav \
  --db-path /path/to/lmdb \
  --attr-mode extract \
  --gpu
```

| `--attr-mode` | Behavior |
|---------------|----------|
| `extract` (default) | timbral attributes from input audio |
| `zeros` | zero control tensor (ablation) |
| `swap` | content `z` from `--input`, attributes from `--swap-input` |
| `constant` | fixed values via `--attr-constant rms=0.5,centroid=0.2` |

Override stats location with `--stats-path /path/to/attribute_stats.yaml`.
