# BraveRT

Open, preallocated C++ inference for BRAVE (and CycleGAN) at **128-sample** ticks,
hosted as a **JUCE VST3/AU** plugin. Goal: RTF &lt; 1 @ 128 on consumer CPU — the
latency path Minifusion provides for vanilla BRAVE, but open and extensible.

## Locked decisions

| Item | Choice |
|------|--------|
| Location | This directory (`BRAVE/brave_rt/`) |
| Host | JUCE (VST3 + AU, macOS first) |
| Engine | JUCE-free C++ library under `cpp/brave_rt/` |
| Tick / Cr | **128** samples @ 44.1 kHz (`N_BAND=16 × ratios [2,2,2,1]`) |
| Encode | Deterministic **mean** of VAE output; full 128-D; **no** PCA |
| Roadmap S0–S6 | **Done** — vanilla + JUCE + unconditional CycleGAN + sidechain + polish |
| Live conditional (later) | **Knob-only** is enough; attribute extract optional — Stage 7 |

## Spoonfeed an implementer agent

1. Read [`docs/PLAN.md`](docs/PLAN.md) for the full spec (shapes, weights, bundle format).
2. Implement **one stage at a time** using the matching brief:

| Stage | Brief | Summary |
|------:|-------|---------|
| 0 | [`docs/STAGE_00_goldens.md`](docs/STAGE_00_goldens.md) | Mean-encode goldens, schema, export stub |
| 1 | [`docs/STAGE_01_pqmf.md`](docs/STAGE_01_pqmf.md) | Streaming CachedPQMF in C++ |
| 2 | [`docs/STAGE_02_engine.md`](docs/STAGE_02_engine.md) | Vanilla BraveEngine + RTF CLI |
| 3 | [`docs/STAGE_03_juce.md`](docs/STAGE_03_juce.md) | Thin JUCE VST3/AU shell |
| 4 | [`docs/STAGE_04_cyclegan.md`](docs/STAGE_04_cyclegan.md) | CycleGAN Enc→W→Dec (sidechain off) |
| 5 | [`docs/STAGE_05_sidechain.md`](docs/STAGE_05_sidechain.md) | CausalLoudnessSidechain |
| 6 | [`docs/STAGE_06_polish.md`](docs/STAGE_06_polish.md) | `--host brave_rt`, CI, docs |
| 7 | [`docs/STAGE_07_knobs.md`](docs/STAGE_07_knobs.md) | Conditional knobs (`attr_mode=1`) |
| 8 | [`docs/STAGE_08_extract.md`](docs/STAGE_08_extract.md) | Optional attribute extract |

Do **not** start Stage N+1 until Stage N exit criteria pass.

## Why this exists

CycleGAN TorchScript in Max/nn~ (LibTorch) underruns below buffer 512. Offline
block reconstruction at 128/256/512 is numerically identical — the cliff is RTF,
not model audio. See diagnosis artifacts under
`evaluation/artifacts/babbling_brook_buffer_diag/` and
[`scratchpaper/realtime_timbre_transfer_hosts.md`](../scratchpaper/realtime_timbre_transfer_hosts.md).

## Layout

```
brave_rt/
  docs/          # PLAN + per-stage briefs (start here)
  python/        # export + parity helpers (Stage 0+)
  schemas/       # brave_rt.json schema
  testdata/      # golden WAVs / .npy refs
  cpp/brave_rt/  # JUCE-free engine
  cpp/plugin/    # JUCE wrapper
  third_party/   # RTNeural submodule (Stage 2+)
```

## Companion docs in BRAVE

- [`scratchpaper/open_brave_plugin_plan.md`](../scratchpaper/open_brave_plugin_plan.md) — earlier sketch; **living spec is `docs/PLAN.md`**
- [`configs/brave.gin`](../configs/brave.gin) — architecture constants
