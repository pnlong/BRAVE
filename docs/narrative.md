# Domain adaptation narrative (write-up seed)

Internal story spine for the project page and a later paper methods section.
The public page currently shows phases **1–3** only.

Running example: **body tap (X)** → **babbling brook / water (Y)**. The same
arc applies to other Y domains (fire, rain, birds, …).

## 1. Zero-shot BRAVE (status quo)

Train an unconditional BRAVE on **Y only**. At inference, feed **X** (tap)
straight into that codec:

\[
x \mapsto \mathrm{Enc}_Y(x) \mapsto \mathrm{Dec}_Y(\cdot)
\]

**Why it fails:** tap is out of distribution for \(\mathrm{Enc}_Y\). Latents
leave the Y manifold; reconstructions are wrong-timbre, ringy, or lifeless.

Pointer: [`domain_adaptation_arc.md`](domain_adaptation_arc.md) §1,
[`configs/brave.gin`](../configs/brave.gin).

## 2. Manual domain adaptation (EQ / reverb)

Keep the frozen Y BRAVE. Before encode, apply hand-tuned **EQ + causal
reverb** so tap acoustics closer match in-domain Y:

\[
x \mapsto \mathrm{DSP}(x) \mapsto \mathrm{Enc}_Y \mapsto \mathrm{Dec}_Y
\]

**Takeaway:** spectrum / room cues matter, but the recipe is brittle and does
not learn a content–timbre tradeoff.

Pointer: phase-2 presets in
[`scripts/presentation_phase_demos.py`](../scripts/presentation_phase_demos.py).

## 3. Separate CycleGAN (Approach 2)

Train **two** plain BRAVE codecs (\(\mathrm{Enc}_X/\mathrm{Dec}_X\),
\(\mathrm{Enc}_Y/\mathrm{Dec}_Y\)). Freeze both. Learn bidirectional latent
warps \(W_{xy}\), \(W_{yx}\):

| Intent | Path |
|--------|------|
| X → Y | \(\mathrm{Enc}_X(x) \to W_{xy} \to \mathrm{Dec}_Y\) |
| Y → X | \(\mathrm{Enc}_Y(y) \to W_{yx} \to \mathrm{Dec}_X\) |

Cross-space warps, random init, cycle + GAN. This is what the project-page
“CycleGAN” demos use (`tap_fx_sc_*` / separate geometry).

**Loudness sidechain:** page demos write both raw and `*_sc` stems for phases
1–3 (`CausalLoudnessSidechain`: tap envelope ducks Y). The site toggle
defaults on (BraveRT / nn~ default).

**Write-up TODO — math to expand:**

- Cycle objectives in waveform vs latent domain (`cycle_domain`)
- Latent cycle modes (`ae_aware` vs `direct`)
- Discriminators (audio / latent) and hybrid / polish schedules

Pointers: [`cyclegan/README.md`](cyclegan/README.md),
[`configs/brave_cyclegan_separate.gin`](../configs/brave_cyclegan_separate.gin).

## 4. Joint embedding + CycleGAN (Approach 3)

Train **one** BRAVE on \(X \cup Y\) (stratified dual-LMDB) so domains share a
latent \(Z\). Then CycleGAN warps are **within-space** remaps on that shared
codec (identity residual init; identity loss is meaningful):

| Intent | Path |
|--------|------|
| X → Y | \(\mathrm{Enc}(x) \to W_{xy} \to \mathrm{Dec}\) |
| Y → X | \(\mathrm{Enc}(y) \to W_{yx} \to \mathrm{Dec}\) |

Not on the public page yet — generate demos once joint water (or chosen Y)
CycleGAN runs are ready.

Pointers: [`configs/brave_cyclegan_joint.gin`](../configs/brave_cyclegan_joint.gin),
[`scratchpaper/joint_embedding_cyclegan.md`](../scratchpaper/joint_embedding_cyclegan.md),
[`scripts/submit_joint_train.sh`](../scripts/submit_joint_train.sh).

## 5. Conditional / Fader controls (contribution track)

Fader attributes \(c\) at decode, availability masks on joint schema
\(c \in x \cup y\), then **conditional CycleGAN** (bookkept return \(c\),
\(L_\mathrm{inv}\), audio AC-GAN). Separate vs joint foundation matrix:

| Foundation | CycleGAN gin |
|------------|--------------|
| Separate Faders on X and/or Y | `brave_cyclegan_conditional.gin` |
| Joint Fader on \(X\cup Y\) | `brave_cyclegan_conditional_joint.gin` |

**Write-up TODO — math to expand:**

- Attribute extractors / sidecar schema
- How \(c\) is passed through warps and cycle consistency
- AC-GAN class heads vs reconstruction attribute losses
- Availability masks for attributes native to only one domain

Pointers: [`cyclegan/conditional.md`](cyclegan/conditional.md),
[`cyclegan/conditional_cyclegan_plan.md`](cyclegan/conditional_cyclegan_plan.md),
[`configs/brave_fader_joint_tap_water.gin`](../configs/brave_fader_joint_tap_water.gin).

## Demo regeneration (project page)

```bash
export PYTHONPATH="${PWD}/RAVE:${PYTHONPATH}"
export BRAVE_STORAGE=/data/hai-res/$USER/BRAVE-data

python scripts/presentation_phase_demos.py \
  --domains babbling_brook fire_crackling birds_chirping rain_sounds \
  --tap-names sm57-0 moanin Shotgun-tapping_11 \
  --gpu --segment-sec 3 --max-sec 12 --skip-pca --sidechain \
  --out-dir "$BRAVE_STORAGE/presentation_demos_v2"

python scripts/package_project_page_audio.py \
  --src "$BRAVE_STORAGE/presentation_demos_v2" \
  --out docs/assets/audio \
  --clips sm57-0 Shotgun-tapping_11 Contact_Mic_L-moanin
```


GitHub Pages serves [`index.html`](index.html) from this `docs/` folder.
