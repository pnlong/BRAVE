# Weights & Biases guide

Training logs scalars, validation audio, and gin/model summaries to [Weights & Biases](https://wandb.ai). Default project: `brave` (`--wandb_project` on `train.py` / `train_prior.py`).

Scalars are logged **per training step only** (no duplicate epoch-averaged curves).

## Latent space size estimation

During training, RAVE regularly estimates the **size** of the latent space given a specific dataset for a specific *fidelity*. The fidelity parameter is a percentage that defines how well the model should be able to reconstruct an input audio sample.

Usually values around 80% yield correct yet not accurate reconstructions. Values around 95% are most of the time sufficient to have both a compact latent space and correct reconstructions.

We log the estimated size of the latent space for several values of fidelity (80, 90, 95 and 99%) as `val/fidelity_0.8`, `val/fidelity_0.9`, etc.

![log_fidelity](log_fidelity.png)

## Reconstruction error

The values you should look at for tracking the reconstruction error of the model are **`train/loss_recon`** (and per-distance components in `train/*`) paired with **`val/loss`** (fullband reconstruction sum at validation).

**Headline scalars (generator steps only):** **`train/loss`** matches the generator objective used in `backward()`. **`train/loss_recon`** is the reconstruction sum. **`train/loss_latent`** is the Fader phase-1 latent adversarial term (0 for base BRAVE).

**Checkpointing:** `ModelCheckpoint` monitors **`val/loss`** (validation reconstruction sum).

![log_distance.png](log_distance.png)

When the 2 phase kicks in, those values increase — **that's usually normal**.

## Adversarial losses (phase 2)

After warmup, training alternates between **discriminator steps** and **generator steps** (`update_discriminator_every` in gin). Metrics are logged only on the step where they are optimized, so curves line up with what each optimizer is actually minimizing.

| Metric | Logged on | What it measures |
|--------|-----------|------------------|
| `train/loss_dis` | Discriminator steps | Hinge (or LS) GAN loss on the **final** discriminator score per scale — how well D separates real vs fake |
| `train/pred_real`, `train/pred_fake` | Discriminator steps | Mean discriminator output on real / fake audio |
| `train/adversarial` | Generator steps | Generator's adversarial term (weighted); pushes fake scores toward “real” |
| `train/feature_matching` | Generator steps | L1 mean distance between **intermediate** discriminator features (real vs fake), weighted by `feature_matching` in gin |

**`train/feature_matching` is not part of `train/loss_dis`.** Feature matching is a *generator* regularizer (Salimans et al.): it penalizes fake intermediate activations drifting from real ones. `loss_dis` only uses the final per-scale logits via hinge/LS GAN. They can move independently — e.g. intermediate features can diverge (feature matching rises) while the final score layer still separates real/fake easily (low `loss_dis`).

The `train/loss_dis`, `train/pred_real`, `train/pred_fake` losses only appear during the second phase. They are usually harder to read, as most GAN losses are, but we include here an example of what *normal* logs should look like.

![log_gan.png](log_gan.png)

## Fader attribute metrics

Fader logs **aggregate** latent-discriminator metrics (not per-attribute curves):

- **`train/fader/latent_ce`**, **`train/fader/latent_acc`** — phase 1, when training the latent discriminator
- **`train/fader/latent_gen_ce`**, **`train/fader/latent_gen_acc`** — phase 1, generator-side view of the same heads
- **`train/fader/latent_dis_loss_dis`** — CE loss from the latent-discriminator optimizer sub-step
- **`train/fader/lambda_factor`** — ramped weight on the latent adversarial term (0 until `LAMBDA_DELAY`, full by `2 * LAMBDA_DELAY`; must be **< `phase_1_duration`** or the encoder adversary never runs in phase 1)

## Validation audio

On validation epochs, `audio_val` (main RAVE model) and `generation` (prior) are logged as W&B Audio panels. Gin config and model architecture strings are stored in the run summary as `config` and `model`.
