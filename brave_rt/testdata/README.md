# BraveRT testdata

## Bundles

| Path | Kind |
|------|------|
| `bundles/tap_foundation_x_shotgun_contact_70d/` | Vanilla BRAVE AE |
| `bundles/babbling_brook_foundation_y_70d/` | Vanilla Y foundation (Mac demo) |
| `bundles/tap_babbling_brook_cyc/` | CycleGAN X→Y (`cyclegan_xy`) + matching `model.ts` |

CycleGAN from `tap_babbling_brook_wf_cyc_mlp2_48072c314a/cyclegan_latent.ckpt`.

## Goldens

- `goldens/tap_foundation_x_shotgun_contact_70d/` — vanilla mean-encode layer refs
- `goldens/cyclegan_brook/` — CycleGAN stream refs vs matching `.ts` (SC on/off) + sidechain module refs

Regenerate:

```bash
cd BRAVE
export PYTHONPATH="${PWD}/RAVE:${PWD}/brave_rt/python:${PYTHONPATH}"
PY=…/micromamba/envs/brave/bin/python

# Vanilla
$PY brave_rt/python/export_brave_rt.py --model=$CKPT --output_dir=brave_rt/testdata/bundles/…

# CycleGAN
$PY scripts/export_model.py --model=$CYCLE_CKPT --host brave_rt \
  --output_dir=brave_rt/testdata/bundles/tap_babbling_brook_cyc
$PY brave_rt/python/cyclegan_ts_ref.py --ts=…/model.ts --input=…/00_tap.wav \
  --out_dir=brave_rt/testdata/goldens/cyclegan_brook --sidechain 0 --block 128
```
