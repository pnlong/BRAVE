# Foundation Y + CycleGAN status

Shotgun+contact **X** is done (`tap_foundation_x_shotgun_contact_70d386ede1` @ 1.5M).

CycleGAN recipe: `configs/brave_cyclegan_best.gin` → run name `tap_fx_sc_<domain>_wf_cyc_mlp2` @ 400k steps.

Update this table when jobs finish or new domains start.

| Domain | Foundation Y | Y run / notes | CycleGAN (`tap_fx_sc_*_wf_cyc_mlp2`) | Next action |
|--------|--------------|---------------|--------------------------------------|-------------|
| babbling_brook | **done** | `70d` + `42be` @ 1.5M | **done** 400k (both hashes) | ok / export as needed |
| birds_chirping | **done** | both @ 1.5M | **done** 400k (both hashes) | ok |
| fire_crackling | **done** | both @ 1.5M | **done** 400k (both hashes) | ok |
| night_ambience | **done** | both @ 1.5M | **done** 400k | ok |
| rain_sounds | **done** | both @ 1.5M | **done** 400k | ok |
| waves_crashing | **done** | both @ 1.5M | **done** 400k | ok |
| handpan_taps | **done** | `42be` @ 1.5M | **done** 400k | ok / export |
| sand_shifting | **done** | `42be` @ 1.5M | **not started** | **start CycleGAN** |
| grass_swaying | **done** | `42be` @ 1.5M | **not started** | **start CycleGAN** |
| gravel_crunching | **done** | `42be` @ 1.5M | **not started** | **start CycleGAN** |
| mouth_sounds | **done** | `42be` @ 1.5M | **not started** | **start CycleGAN** |
| asmr | **in progress** (~1.0M) | `42be` — latest `epoch-epoch=5791.ckpt`, wandb `fuq7b3kv` | blocked on Y | **resume Y** → then CycleGAN |
| jsut | **done** | `42be` @ 1.5M | not started | optional CycleGAN |
| mechanical | **done** | `42be` @ 1.5M | not started | optional CycleGAN |
| synth_abstract | **done** | `42be` @ 1.5M | not started | optional CycleGAN |

**Legend:** Y **done** = `epoch_1500000.ckpt`. CycleGAN **done** = `last.ckpt` @ ≥400k.

Last reviewed: 2026-10-09.
