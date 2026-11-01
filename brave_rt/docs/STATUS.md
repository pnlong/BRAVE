# BraveRT implementation status

Updated as stages land in-tree. Last update: **2026-10-31** (Stage 8 extract landed; Stages 4–7 closed on Linux Xeon).

| Stage | Status | Notes |
|------:|--------|-------|
| Docs bootstrap | **done** | `README.md`, `docs/PLAN.md`, `STAGE_*.md` |
| 0 Goldens | **done** | Bundle + mean-encode `.npy`/WAVs under `testdata/` |
| 1 PQMF | **done** | `CachedPqmf` + `pqmf_roundtrip` **PASS** |
| 2 Engine | **done** | Enc/Dec streaming; E2E parity; soak; RTF@128 mean &lt; 1 on Xeon |
| 3 JUCE | **done (Logic AU verified)** | Mac AU works in Logic (**copy** install, not symlink). Ellipsis button label fixed. |
| 4 CycleGAN | **done** | `CycleGanEngine` Enc_X→W_xy→Dec_Y; export `--host brave_rt`; parity vs matching `.ts` SC-off; RTF@128 ≪ 1 |
| 5 Sidechain | **done** | `CausalLoudnessSidechain` wired post-iPQMF; JUCE `sidechain` default on; parity SC-on; `sidechain_smoke` / `sidechain_parity_cli` |
| 6 Polish | **done** | `export_model.py --host brave_rt` vanilla+CycleGAN; CI smoke; CLAUDE + hosts table + Mac README |
| 7 Knobs | **done** | Fader export + widened Dec; data-driven editor; UI polish; empty-state message |
| 8 Extract | **done (opt-in)** | `DescriptorExtract` (rms/centroid/flatness/zcr) + `BraveEngine::set_attr_mode`; plugin `attr_mode` choice |

## Stage 7 verification (2026-10-31)

**Bundle:** `brave_rt/testdata/bundles/water_fader_scene` (FaderRAVE, D=6, Dec project 134-D)  
from `BRAVE-data/fsd50k_brave/water/runs/water_fader_scene_96d85428b0` + `attribute_stats.yaml`.

| Check | Result |
|-------|--------|
| Export `--host brave_rt` Fader | `model_type=fader`, `host_controls` (5 knobs + `water_scene` menu), `attribute_stats.yaml` copied |
| Stream @128 `attr_mode=1` vs Torch mean-encode oracle | PASS aligned max-abs **9.54e-7**, SNR **118.1 dB** |
| RTF @128 (`rtf_bench`, water fader) | mean **0.865** PASS (&lt; 1) |
| Unconditional CycleGAN editor | empty strip → “No conditioning features in this bundle” |
| JUCE params | fixed `attr_0`…`attr_15` [0,1] slots + `attr_mode` choice; automation/state restore |
| UI polish | graphite + copper panel; BraveRT wordmark; transfer title secondary; rotary/menus |

**Conditional CycleGAN:** no Fader-backed `cyclegan_latent.ckpt` in BRAVE-data yet. Export fills `host_controls` when Dec_Y is widened; C++ concat-before-Dec_Y is ready; FiLM/warp cond not implemented.

Goldens: `brave_rt/testdata/goldens/water_fader_scene/`.

## Stage 8 verification (2026-10-31)

**Bundle:** `water_fader_scene` (D=6: centroid, rms extractable; bandwidth/sharpness/booming/water_scene stay manual). Engine `build_s4`, g++ 13.3 `-O3 -march=native`.

| Check | Result |
|-------|--------|
| `attr_mode=1` vs Stage 7 golden `stream128_cpp.npy` | **bit-identical** (max-abs 0); vs oracle 9.5e-7 |
| Vanilla `parity_cli e2e stream 128` | PASS max-abs 3.14e-6, SNR 108.9 dB |
| `extract_smoke` | PASS (sine rms/centroid/zcr, noise flatness, chunk-invariance, mode 0/1/2 semantics, pre-ready == manual) |
| Extractor alone (`push(128)`, core 0) | mean **21 µs** per block ≈ **0.007 RTF** (one 2048-pt FFT frame per 4 blocks) |

**RTF @128** (`taskset -c 0 rtf_bench <bundle> 128 N 2 1 <mode>`; extract run warms until the 3 s history is full). The shared Xeon was heavily contended during this session (same bursty `max ≈ 3` outliers in *both* modes), so absolute numbers are well above the quiet-box Stage 4 figures:

| Run | mode 1 (manual) mean / median | mode 2 (extract) mean / median |
|-----|-------------------------------|--------------------------------|
| quietest pair (core 0, 3000 it) | **0.275** / 0.264 | **0.384** / 0.330 |
| contended, core 0 (7 pairs, 3000–4000 it) | 0.69–0.85 / ≈0.64–0.70 | 0.73–0.99 / ≈0.60–0.87 |
| core 20 / 40 / 60 | 0.63 / 0.78 / 0.72 | 0.83 / 0.88 / 0.76 |

Mean RTF stayed **< 1 in every run for both modes**; extract is within run-to-run noise of manual on the contended box (microbenchmark says ≈ +0.007), so no helper thread / one-hop delay was needed. Extract stays **opt-in** (default `attr_mode=1`) because it needs ~3 s of input before it takes over and differs from training-style trajectories.

## Stage 4–5 verification (2026-10-30)

**Machine:** Intel Xeon w7-3565X, g++ 13.3 `-O3 -march=native -DNDEBUG`

**Bundle:** `brave_rt/testdata/bundles/tap_babbling_brook_cyc`  
from `BRAVE-data/.../tap_babbling_brook_wf_cyc_mlp2_48072c314a/cyclegan_latent.ckpt`  
(`init_mode=random`, `geometry=separate`, `latent_n_layers=2`)

**Oracle:** streaming `ScriptedCycleGANXY` / matching `model.ts` exported from the **same** ckpt  
(not `BRAVE-data/exports/cyclegan_ts_done/babbling_brook_10122026.ts` — that file does **not** match this checkpoint; max-abs ≈ 1.7 vs ckpt).

| Check | Result |
|-------|--------|
| CycleGAN stream @128 SC **off** vs matching `.ts` | PASS max-abs **1.15e-5**, SNR **92.0 dB** |
| CycleGAN stream @128 SC **on** vs matching `.ts` | PASS max-abs **7.87e-6**, SNR **93.5 dB** |
| `CausalLoudnessSidechain` vs Python | PASS max-abs **0** |
| RTF @128 (SC on), `taskset -c 0` | mean **0.246**, median **0.242** PASS |
| RTF @256 (same) | mean **0.203**, median **0.166** PASS |
| Vanilla AE @128 (same pin) | mean **0.281** PASS (CycleGAN ≈ AE cost) |
| RTF @128 (shared box busy / unpinned) | can spike mean &gt; 1 — same Stage 2 caveat |

Goldens: `brave_rt/testdata/goldens/cyclegan_brook/` (`stream128_sc{0,1}_y.npy`, sidechain refs).

## Stage 2 verification (re-run 2026-10-30 evening)

| Check | Result |
|-------|--------|
| PQMF roundtrip | PASS (max-abs 0) |
| E2E stream host 128 | PASS max-abs **3.14e-6**, SNR **108.9 dB** |
| RTF @128 | mean **0.77**, median **0.64** PASS |

## Stage 3 notes

- Formats: **VST3** (Linux/macOS); **AU** when `APPLE`
- Mac install: **copy** AU into `~/Library/Audio/Plug-Ins/Components/` (see `dist/README_MAC_LOGIC.md`)
- Plugin loads `model_type=brave` / `fader` / `cyclegan_xy` / `cyclegan_yx`; CycleGAN exposes **Sidechain**; Fader exposes conditioning strip + **Attr mode**

## Build / smoke

```bash
# Engine + parity (Linux)
bash brave_rt/ci/run_engine_smoke.sh

# Or manual:
export PATH="/data/hai-res/p1long/micromamba/envs/brave/bin:$PATH"
cmake -S brave_rt/cpp/brave_rt -B brave_rt/cpp/brave_rt/build_s4 \
  -DCMAKE_BUILD_TYPE=Release -DBRAVE_RT_NATIVE=ON
cmake --build brave_rt/cpp/brave_rt/build_s4 -j
```

## Export

```bash
export PYTHONPATH="${PWD}/RAVE:${PWD}/brave_rt/python:${PYTHONPATH}"

# Vanilla or unconditional CycleGAN
python scripts/export_model.py --model path/to/vanilla_or_cyclegan --host brave_rt \
  --output_dir brave_rt/testdata/bundles/my_run

# Fader (needs attribute_stats.yaml via --stats_path or --db_path)
python scripts/export_model.py --model path/to/fader_run --host brave_rt \
  --stats_path /path/to/attribute_stats.yaml \
  --output_dir brave_rt/testdata/bundles/my_fader
```

## Next focus

1. Optional: train/export conditional CycleGAN + FiLM warp; per-attr override/scale; f0; nn~ extract calibration.
2. Manual DAW check: Fader knobs + CycleGAN tap→brook @ buffer 128 (Mac AU **copy**-install).
3. Out of roadmap: Max external, Windows.
