# Conditional CycleGAN — design plan

Living design note for BRAVE × CycleGAN × Fader. Paper-facing Method math: [`conditional.md`](conditional.md). Ops for unconditional CycleGAN stay in [`README.md`](README.md).

Formulas use plain text (Cursor plan preview often breaks LaTeX).

---

## Locked v1 decisions

- **Geometry / critics:** `cycle_domain="waveform"` + **audio** MSD (`gan_domain=audio`) — same as current separate runs.
- **Warps:** content-only `LatentCanonicalizer`; **no FiLM**. `c` enters at **decode** only.
- **`C_D`:** frozen Fader `LatentDiscriminator`; probe for `L_inv` only — never retrained in CycleGAN.
- **Classic `lambda_identity`:** **0** to match separate Approach 2 (not a new disable; joint uses 0.5 for a different geometry).
- **Case C strip / local-attr `L_id`:** **not implemented**. Default cycle reuses `c_true` including Case C on return; `L_cyc` still hard (content must ride in `z`).
- **First recipe:** tap (plain BRAVE, empty attrs) ↔ water (Fader, e.g. `water_scene`) = Case B on X→Y; empty `c_X` on Y→X.
- **Paper doc required:** `docs/cyclegan/conditional.md`.

---

## The three axes

| Axis | What | Role | Trainable at CycleGAN stage? |
|------|------|------|------------------------------|
| **BRAVE** | Causal VAE Enc/Dec (optional Fader widens Dec to 128+D) | Domain codecs; content in `z` | **Frozen** |
| **CycleGAN** | Dual warps `W_xy`, `W_yx` + dual audio D | Content transport across domains | **Hot:** warps + D |
| **Fader** | `Dec(cat(z,c))` + classifier `C_D` on `z` | Inject `c` at decode; frozen probe for `L_inv`; attrs/sidecars/priors | Codecs + `C_D` **frozen** |

Empty attribute list ⇒ unconditional CycleGAN (today’s path). Non-empty ⇒ that domain must be a **FaderRAVE training ckpt** (widened Dec + `latent_discriminator` still in ckpt; export `.ts` is not enough for `C_D`).

```text
x → Enc_X → W_xy → cat(z, c_Y) → Dec_Y → y_fake
                 ↘ L_inv via C_Y
                              y_fake → D_Y (realness + class heads)
```

---

## Generators

```text
G_xy(x, c_Y) = Dec_Y( W_xy(Enc_X(x)), c_Y )
G_yx(y, c_X) = Dec_X( W_yx(Enc_Y(y)), c_X )
```

Empty `c` ⇒ omit concat (unconditional). **No FiLM on W.**

Hot W is still co-trained with whatever `c` was used at decode (gradients depend on `c`), but at inference `W(Enc(x))` is a function of `x` only — changing class only changes `c` at Dec. That is **not** implicit FiLM (FiLM would be `W(z,c)`).

---

## Cases A / B / C (per attribute)

Organize **per attribute**, not “is domain X conditional.”

| Case | Signal | Forward (X→Y) `c_Y[a]` | Return `c_X_return[a]` |
|------|--------|------------------------|-------------------------|
| **A shared** | in both lists | Pass-through `c_true_X`; optional `swap_prob` (default 0) | Same value used forward |
| **B target-only** | only in Y | Sample `prior_Y` (uniform/marginal) | N/A on `c_X`; Y→X→Y uses `c_true_Y` on return |
| **C source-local** | only in X | **Absent from `c_Y`** | **Reuse `c_true_X` including `a`** |

### Case C

Case C means absent from the **target** conditioning set, not dropped from the round-trip:

```text
c_Y        := assemble for Y   # a not included
y_fake     := G_xy(x, c_Y)
c_X_return := c_true_X(x)      # INCLUDES a
x_hat      := G_yx(y_fake, c_X_return)
L_cyc^x    := L_recon(x, x_hat)
```

Passing `a` on the return does **not** make `L_cyc` easy. Under Fader, `c` selects controls; **content still rides in `z`**. Full-waveform recon still requires correct content through Enc→W→…→Dec.

**Section 8 local-attr `L_id` (strip content, then reattach):** out of v1. Weak motivation once you accept default `L_cyc` is already a hard content constraint.

---

## Losses (waveform cycle + audio D)

```text
L_G = λ_cyc * L_cyc
    + γ_gan * (λ_GAN * L_GAN + λ_FM * L_FM + λ_adv_cls * L_adv_cls)
    + γ_inv * λ_inv * L_inv
    + λ_id * L_id          # classic identity; λ=0 for separate
    + γ_spread * λ_spread * L_spread

L_D = hinge real/fake + L_D_cls on reals only
```

| Term | Role |
|------|------|
| `L_cyc` | STFT+RMS round-trip; return-leg `c` is **bookkept**, never re-inferred from the fake |
| `L_GAN` / `L_FM` | Existing hinge + feature matching on waveforms |
| `L_adv_cls` / `L_D_cls` | **AC-GAN** class heads on audio D (not Stage-1 projection-cGAN) |
| `L_inv` | `CE(C_target(z_post_warp), Uniform)`; warp-only grads; frozen `C_D` |
| Classic `L_id` | Wrong-encoder identity in trainer; separate default λ=0; joint uses 0.5 |

**FM:** prefer same-class reals in-batch when `c` present; log miss rate on fallback.

**Schedule:**

- Assemble/pass `c` from step 0 (warmup included) so Dec width is always correct.
- `γ_gan`: existing warmup + ramp; gates GAN, FM, **and** `L_adv_cls`.
- `γ_inv`: starts only after `γ_gan=1`, plus `inv_delay` / `inv_ramp` — **not** lockstep with spread.

**Diagnostics:** log `L_cyc` stratified by Case B sampled class.

---

## Why waveform D for v1

- `L_inv` wants post-warp `z` **class-free**.
- `L_adv_cls` needs class evidence on the **output**.
- Latent D with class heads conflicts with `L_inv`.

Future (document in `conditional.md`, not v1): latent cycle + audio AC-GAN; or latent D real/fake-only.

---

## Classic identity vs Case C local `L_id`

| Name | Meaning | v1 |
|------|---------|-----|
| Classic CycleGAN `lambda_identity` | Separate: `G_xy(y)` via Enc_X (awkward). Joint: meaningful within-space prior | **0** (match separate runs) |
| Case C local-attr `L_id` | Optional strip/reattach diagnostic | **Not implemented** |

Current separate log example: `lambda_identity=0.0`. Joint gin: `0.5`.

---

## FiLM on warps — not strictly better

Conditionality without FiLM:

1. Explicit decode injection (frozen Fader Dec).
2. `L_adv_cls` steers audio toward used `c`.
3. `L_inv` keeps `z` class-uninformative.
4. Hot W co-trained with the `c`-family (compatibility), but `W(z)` has no `c` input.

FiLM = second injection site `W(z,c)`. Can help if class needs different Y-latent geometry; fights Fader/`L_inv` story; harder ablations. **v1: no FiLM**; ablation only if decode-side steering fails.

---

## Out of v1

1. FiLM on warps  
2. Latent `cycle_domain` / latent D (esp. with class heads)  
3. Case C strip / local-attr `L_id`  
4. Retrain `C_D`  

**Joint-embedding conditional** is in scope: joint Fader foundation (schema \(x\cup y\) + availability masks) + [`brave_cyclegan_conditional_joint.gin`](../../configs/brave_cyclegan_conditional_joint.gin). See [`conditional.md`](conditional.md) §10.

---

## Paper-facing documentation (primary deliverable)

**File:** [`docs/cyclegan/conditional.md`](conditional.md) (to write; LaTeX-ready Method text).

Required sections:

1. Setup and three axes  
2. Generators + empty-`c` degeneration  
3. Fader pretraining recall  
4. Cases A/B/C (Case C return)  
5. Cycle consistency with bookkept `c`  
6. Warp invariance `L_inv`  
7. Audio D AC-GAN heads  
8. Full objective + schedule  
9. Special cases (unconditional empty lists; tap↔water)  
10. Relation to unconditional CycleGAN / Fader / domain-adaptation arc  
11. Future work (latent+audio class; FiLM ablation)

Cross-link from [`README.md`](README.md) and [`../domain_adaptation_arc.md`](../domain_adaptation_arc.md). Keep doc in sync when code changes.

---

## Implementation map

### Docs

1. Write `conditional.md` per outline above.  
2. Short pointer in CycleGAN README + domain adaptation arc.

### Reuse

- Fader load: `RAVE/scripts/train_canonicalizer.py` (`FaderRAVE` + stats).  
- Datasets: `TaggedFaderDataset`, sidecars, `attribute_marginals.py`.  
- Core: `RAVE/rave/canonicalizer/cycle_trainer.py`, `train_cyclegan.py`.  
- Frozen `C_D`: `RAVE/rave/fader/latent_discriminator.py`.

### Add / extend

1. `RAVE/rave/canonicalizer/cycle_attributes.py` — A/B/C resolve, assemble `c` (return includes Case C), CE-to-uniform.  
2. `train_cyclegan.py` — per-domain RAVE vs Fader; mixed collate that keeps Y attrs when X is `None`.  
3. `CycleGANTrainer` — thread `c`; `cat(z, attr_norm)` before `_decode` (need multiband for STFT; don’t call `FaderRAVE.decode()`).  
4. Audio D — AC-GAN heads; `L_D_cls` / `L_adv_cls`; class-matched FM.  
5. `CycleGANRampCallback` — independent `inv_factor`.  
6. Manifest + `cycle_inference.py` `attr_y`; gin `configs/brave_cyclegan_conditional.gin`.  
7. Tests: resolver, empty-list = unconditional, cycle bookkeeping, aux heads; keep existing CycleGAN tests green.

---

## Implementation todos

- [ ] Write `docs/cyclegan/conditional.md` + cross-links  
- [ ] `cycle_attributes.py` + unit tests  
- [ ] Fader/RAVE load + mixed collate + conditional gin  
- [ ] Thread `c` through trainer forward/cycle  
- [ ] AC-GAN heads + class-matched FM  
- [ ] Frozen `C_D` `L_inv` + `inv_factor` ramp + stratified Case B logs  
- [ ] Inference `attr_y`, manifest, README ops; unconditional path green  

---

## Mental model for debugging

| Symptom | Likely cause | Knob |
|---------|--------------|------|
| Domain OK, wrong water class | Weak `L_adv_cls` | Raise λ_adv_cls; class-matched FM |
| Class follows, content destroyed | Cycle too weak | Raise λ_cyc; longer warmup |
| Stratified cycle varies by Case B class | Class leaking into `z` | Start γ_inv earlier / raise λ_inv |
| Decoder shape / silence | Wrong backbone width | Fader Y ckpt + stats; concat = `DECODER_LATENT_SIZE` |
| Unconditional regresses | Empty-list path broken | New losses must no-op on empty attrs |
