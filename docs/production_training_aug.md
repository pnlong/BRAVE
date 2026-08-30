# Production training augmentations (notes)

## Random phase mangle (train-only)

RAVE’s `get_dataset` applies `random_phase_mangle` with probability 0.8 on each crop: a random all-pass-style IIR that warps **phase** in a mid-band range (~20–2000 Hz) while leaving the magnitude spectrum mostly intact. It is **data augmentation for training/val dataloaders**, not something Max/nn~/export applies at inference.

You do **not** need to “turn it off in production” at runtime — it is already off for live / TorchScript / demo encode–decode. Val loaders skip it (prefix crop, no gin augs). For **production training**, keep or drop it as an ablation: keeping it can reduce phase overfitting; dropping it makes train crops closer to raw recordings (and to our presentation demos). Do not confuse “turn off for production” with inference.

## Gain robustness (preferred over hard peak-norm)

Today we often **peak-normalize** (LMDB `normalize: true`, and/or demo peak-norm to ~0.95). That trains models in a narrow loudness regime. Live contact-mic / host levels vary a lot, so timbre-transfer and CycleGAN can latch onto absolute gain.

**Preferred production direction:** random **gain augmentation** during BRAVE / CycleGAN / canonicalizer training (e.g. uniform dB scale per crop in a sane range), so adapters see many levels of the same content. Keep a light **inference safety** (soft limiter / clip guard), and keep train and live paths consistent — do not silently peak-norm every Max input if the model was trained with random gain.

Related demos: `scripts/presentation_phase_demos.py` currently peak-norms for listening fairness; that is not a claim that peak-norm is the right production train recipe.
