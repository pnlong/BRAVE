# Production training augmentations (notes)

## Random phase mangle (train-only)

RAVE’s `get_dataset` applies `random_phase_mangle` with probability 0.8 on each crop: a random all-pass-style IIR that warps **phase** in a mid-band range (~20–2000 Hz) while leaving the magnitude spectrum mostly intact. It is **data augmentation for training/val dataloaders**, not something Max/nn~/export applies at inference.

You do **not** need to “turn it off in production” at runtime — it is already off for live / TorchScript / demo encode–decode. Val loaders skip it (prefix crop, no gin augs). For **production training**, keep or drop it as an ablation: keeping it can reduce phase overfitting; dropping it makes train crops closer to raw recordings (and to our presentation demos). Do not confuse “turn off for production” with inference.

## Loudness robustness (preferred over hard peak-norm)

Today we often **peak-normalize** (LMDB `normalize: true`, and/or demo peak-norm to ~0.95). That trains models in a narrow loudness regime. Live contact-mic / host levels vary a lot, so timbre-transfer and CycleGAN can latch onto absolute gain — models may only “wake up” when host input gain is cranked high.

**Default:** relative **`RandomGain`** in [`configs/brave.gin`](../configs/brave.gin), the same settings as the keep-silence gain runs (`gain_range = (-12, 6)`, `prob = 1`, `limit = True`). Each crop is shifted by that many dB, so a naturally loud clip stays louder than a quiet one. The −12 dB end is a modest quieter copy. If the shift would exceed ±1, `limit` peak-normalizes that crop. Train-only (val loaders skip gin augs).

**`RandomTargetLoudness`** (`target − current_rms`, previously `loudness_range = (-60, -3)`) puts loud and quiet clips in one absolute band. That range spent most draws near silence and is not the training default. It remains available if a run needs an absolute-level ablation.

Pair with **non-peak-norm LMDBs** (do not pass `--normalize` at preprocess). Keep a light **inference safety** (soft limiter / clip guard), and keep train and live paths consistent — do not silently peak-norm every Max input if the model was trained this way.

Quiet crops / mic gaps: `dataset.maybe_reject_silent.enabled = False` (same gin). Re-enable with `--reject_silent` only if needed.

Related demos: `scripts/presentation_phase_demos.py` currently peak-norms for listening fairness; that is not a claim that peak-norm is the right production train recipe.
