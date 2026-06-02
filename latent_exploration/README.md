# Latent exploration

Encode audio through a trained RAVE/BRAVE checkpoint, visualize latents, and experiment with latent masking before decode.

Run from the **BRAVE repo root** with the **`brave` conda/micromamba env activated**:

```bash
micromamba activate brave
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
```

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

All paths go through **`mask_reconstruct.py`** (`run_reconstruction`). By default each run saves:

- `{stem}_original.wav`
- `{stem}_reconstructed.wav`
- `{stem}_latents.png` (no mask) or `{stem}_mask_plot.png` (with mask)

Use `--no-wavs` or `--no-plot` to skip either output.

### Visualize latents

Thin wrapper: `mask_reconstruct` with `--mask-style none`, output under `artifacts/plots/<stem>/`.

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
