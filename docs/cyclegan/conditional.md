# Conditional CycleGAN (Method)

Paper-facing math for BRAVE × CycleGAN × Fader. Ops / train recipes: [`README.md`](README.md). Design checklist: [`conditional_cyclegan_plan.md`](conditional_cyclegan_plan.md). Arc: [`../domain_adaptation_arc.md`](../domain_adaptation_arc.md).

## 1. Setup and three axes

Domains \(X,Y\) (e.g. tap, water) with unpaired data \(x\sim p_X\), \(y\sim p_Y\).

| Axis | Objects | Trainable at CycleGAN stage |
|------|---------|-----------------------------|
| BRAVE codecs | \(\mathrm{Enc}_D,\mathrm{Dec}_D\) (optional Fader-widened decoder) | Frozen |
| CycleGAN | Warps \(W_{xy},W_{yx}\); audio discriminators \(D_X,D_Y\) | Hot |
| Fader | Explicit controls \(c_D\); classifier \(C_D\) on content \(z\) | \(C_D\) frozen (probe) |

Empty conditioning list on a domain ⇒ unconditional decode (no \(c\), no \(C_D\), no class heads). Non-empty ⇒ that backbone must be a FaderRAVE **training** checkpoint (widened \(\mathrm{Dec}\) and `latent_discriminator` present).

## 2. Generators

\[
G_{xy}(x,c_Y)=\mathrm{Dec}_Y\big(W_{xy}(\mathrm{Enc}_X(x)),\,c_Y\big),\qquad
G_{yx}(y,c_X)=\mathrm{Dec}_X\big(W_{yx}(\mathrm{Enc}_Y(y)),\,c_X\big).
\]

Fader decode is channel-concat: \(\mathrm{Dec}_D(z,c)=\mathrm{Gen}_D([z;c])\) with \(c\in[-1,1]^{K_D\times T}\). If \(c_Y=\emptyset\), omit concat.

**No FiLM on warps in v1.** \(W\) is content-only. Hot \(W\) is co-trained with the \(c\)-family (gradients depend on the \(c\) used at decode) but \(W(z)\) has no \(c\) input at inference.

## 3. Fader pretraining (recall)

Phase 1 trains \(C_D(z)\) with \(+\mathrm{CE}\) vs true labels and the encoder with \(-\mathrm{CE}\), while \(\mathrm{Dec}_D(z,c)\) reconstructs with explicit \(c\). Attribute information is driven out of content \(z\) into \(c\). At CycleGAN time \(C_D\) is frozen and reused as an invariance probe.

## 4. Attribute cases A / B / C

Resolved **per attribute** from the two domains’ name lists:

| Case | Config | Forward \(c_Y[a]\) (X→Y) | Return \(c_X^{\mathrm{return}}[a]\) |
|------|--------|---------------------------|-------------------------------------|
| A shared | in both | Pass-through \(c_{\mathrm{true},X}\); optional swap | Same value used forward |
| B target-only | only in Y | Sample \(\mathrm{prior}_Y[a]\) | N/A on \(c_X\); Y→X→Y uses \(c_{\mathrm{true},Y}\) on return |
| C source-local | only in X | Absent from \(c_Y\) | Reuse \(c_{\mathrm{true},X}\) including \(a\) |

Case C means absent from the **target** list, not from the round-trip. Default cycle reuses full source-true \(c\) on return so \(\mathrm{Dec}\) sees the controls it was trained with. Passing local \(a\) on return does **not** make cycle easy: content must still survive in \(z\).

## 5. Cycle consistency with bookkept \(c\)

Return-leg conditioning is **never** re-inferred from the intermediate fake:

\[
\begin{aligned}
c_Y &\leftarrow \text{assemble (A/B)},\\
y_{\mathrm{fake}} &= G_{xy}(x,c_Y),\\
c_X^{\mathrm{return}} &\leftarrow c_{\mathrm{true},X}(x)\quad\text{(incl. Case C)},\\
\hat x &= G_{yx}(y_{\mathrm{fake}},\,c_X^{\mathrm{return}}),\\
\mathcal{L}_{\mathrm{cyc}}^x &= \mathcal{L}_{\mathrm{recon}}(x,\hat x).
\end{aligned}
\]

Symmetric for \(Y\to X\to Y\). v1 recon is waveform STFT+RMS (`cycle_domain="waveform"`).

## 6. Warp invariance

\[
\mathcal{L}_{\mathrm{inv}}^{xy}
=\sum_{k\in\mathrm{attrs}(Y)}
\mathrm{CE}\big(C_Y(W_{xy}(\mathrm{Enc}_X(x)))_k,\;\mathrm{Uniform}(K_k)\big).
\]

Gradients into the warp only; \(C_Y\) frozen. Empty target list ⇒ no term.

## 7. Audio discriminator with AC-GAN heads

\[
D_Y(x)\to\big(\text{realness},\;\{\text{class logits}_k\}\big).
\]

D-step (reals): hinge + \(\mathcal{L}_{D,\mathrm{cls}}=\sum_k\mathrm{CE}(D_Y^{\mathrm{cls},k}(y),c_{\mathrm{true}}(y)_k)\).

G-step: hinge + \(\mathcal{L}_{\mathrm{adv\_cls}}=\sum_k\mathrm{CE}(D_Y^{\mathrm{cls},k}(G_{xy}(x,c_Y)),c_Y[k])\).

v1 uses **audio** MSD (not latent D) so class evidence lives on the waveform while \(\mathcal{L}_{\mathrm{inv}}\) keeps \(z\) class-free.

## 8. Full objective and schedule

\[
\begin{aligned}
\mathcal{L}_G
&= \lambda_{\mathrm{cyc}}\mathcal{L}_{\mathrm{cyc}}
+ \gamma_{\mathrm{gan}}\big(
\lambda_{\mathrm{GAN}}\mathcal{L}_{\mathrm{GAN}}
+\lambda_{\mathrm{FM}}\mathcal{L}_{\mathrm{FM}}
+\lambda_{\mathrm{adv\_cls}}\mathcal{L}_{\mathrm{adv\_cls}}
\big) \\
&\quad+ \gamma_{\mathrm{inv}}\lambda_{\mathrm{inv}}\mathcal{L}_{\mathrm{inv}}
+ \lambda_{\mathrm{id}}\mathcal{L}_{\mathrm{id}}
+ \gamma_{\mathrm{spread}}\lambda_{\mathrm{spread}}\mathcal{L}_{\mathrm{spread}}.
\end{aligned}
\]

- Assemble \(c\) from step 0 (warmup included).
- \(\gamma_{\mathrm{gan}}\): existing cycle warmup then GAN ramp; gates GAN, FM, and \(\mathcal{L}_{\mathrm{adv\_cls}}\).
- \(\gamma_{\mathrm{inv}}\): starts only after \(\gamma_{\mathrm{gan}}=1\), plus delay/ramp (independent of spread).
- Classic \(\lambda_{\mathrm{id}}=0\) for separate Approach 2 (matches current unconditional separate runs).

## 9. Special cases

- **Unconditional:** both attribute lists empty ⇒ today’s CycleGAN.
- **First recipe (tap↔water):** \(X=\emptyset\), \(Y=\{\texttt{water\_scene}\}\) ⇒ Case B on X→Y; empty \(c_X\) on Y→X.

## 10. Full matrix: separate vs joint foundations

Conditional CycleGAN is complete once both foundation geometries exist; the CycleGAN objective does not change—only codec count / warp init / how \(c\) is sized for decode.

| Foundation | What it is | CycleGAN |
|------------|------------|----------|
| **Separate conditional** | FaderRAVE on \(X\) alone (\(c\in x\)) and/or \(Y\) alone (\(c\in y\)); plain BRAVE = empty \(c\) | Approach 2: [`brave_cyclegan_conditional.gin`](../../configs/brave_cyclegan_conditional.gin) |
| **Joint conditional** | One FaderRAVE on \(X\cup Y\) with schema \(c\in x\cup y\) + per-sample availability masks | Approach 3: [`brave_cyclegan_conditional_joint.gin`](../../configs/brave_cyclegan_conditional_joint.gin) (`shared_backbone=True`) |

Unconditional separate/joint are the empty-\(c\) degeneration of the same paths.

### Joint Fader foundation

**Schema** \(= x\cup y\): one widened \(\mathrm{Dec}\) (\(128+|x\cup y|\)) and one \(C\) with a head per union attribute.

**Availability is not a union.** Domain-local labels are missing on the other domain’s samples:

| Sample | Channel | \(\mathrm{attr\_norm}\) into Dec | CE on \(C\) |
|--------|---------|----------------------------------|-------------|
| Native attr present | that attr | true value | on |
| Domain-local missing | other domain’s attr | **neutral** (0 in \([-1,1]\)) | **masked off** |
| Shared continuous (audio descriptors) | always | from audio | on |

Unmasked CE on neutral missing labels would teach spurious maps (e.g. tap \(\Rightarrow\) `water_scene`=mid).

**Native vs schema at CycleGAN time.** The frozen joint backbone exposes the full schema for decode width. Cases A/B/C and \(\mathcal{L}_{\mathrm{inv}}\) use **native** lists per domain (`native_x` / `native_y`), not the full schema twice (that would mark every attr Case A). Assemble fills non-native schema slots with neutral so \(\mathrm{Dec}\) always sees \(|x\cup y|\) channels. \(\mathcal{L}_{\mathrm{inv}}\) runs CE-to-uniform only on **target-native** heads of the shared \(C\).

### Ops (joint)

```bash
# 1) Precompute descriptors on each LMDB; merge stats for union schema
python RAVE/scripts/precompute_descriptors.py --db_path /path/tap_lmdb \
  --config configs/brave_fader_joint_tap_water.gin --train_only
python RAVE/scripts/precompute_descriptors.py --db_path /path/water_lmdb \
  --config configs/brave_fader_joint_tap_water.gin --train_only
python RAVE/scripts/merge_attribute_stats.py \
  --db_path_x /path/tap_lmdb --db_path_y /path/water_lmdb \
  --output /path/joint_stats/attribute_stats.yaml

# 2) Joint Fader foundation (stratified dual LMDB)
python RAVE/scripts/train.py --config configs/brave_fader_joint_tap_water.gin \
  --db_path_x /path/tap_lmdb --db_path_y /path/water_lmdb \
  --name joint_fader_tap_water --batch 8
# or: bash scripts/submit_joint_fader_train.sh

# 3) Conditional joint CycleGAN
python RAVE/scripts/train_cyclegan.py \
  --config configs/brave_cyclegan_conditional_joint.gin \
  --backbone_x_config configs/brave_fader_joint_tap_water.gin \
  --ckpt_x /path/joint_fader.ckpt \
  --db_path_x /path/tap_lmdb --db_path_y /path/water_lmdb \
  --native_attrs_x "" --native_attrs_y water_scene \
  --canonicalizer_type latent --name tap_water_cond_joint
```

## 11. Out of v1 / future work

- FiLM on warps (ablation if decode-side steering fails).
- Latent cycle + audio AC-GAN; latent D real/fake-only (not class heads with strong \(\mathcal{L}_{\mathrm{inv}}\)).
- Case C strip/reattach local identity loss.
- Retraining \(C_D\).
