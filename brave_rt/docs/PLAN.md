# BraveRT JUCE plugin — full implementation plan (for implementer agents)

**Goal:** Open, preallocated C++ inference for BRAVE (then CycleGAN) at **128-sample** ticks with RTF &lt; 1, hosted as a **JUCE VST3/AU** plugin. Solves the nn~/LibTorch underrun cliff diagnosed on CycleGAN `.ts` (offline audio identical at 128/256/512; CPU RTF crossed &gt;1 below 512).

**Out of this roadmap (do not implement in S0–S6):** Max external, waveform canonicalizer, Minifusion h5 binary compatibility, PCA latent truncation path used by vanilla nn~ `VariationalScriptedRAVE`, and streaming **attribute extraction**.

**Conditional / Fader live control:** **Knob-only** (`attr_mode=1` — host sliders/menus → normalized attrs → decode/warp) is enough for live conditional use. Attribute extract is optional later for hands-free / automatic attrs, not a blocker. This roadmap ships **unconditional** CycleGAN first (Enc→W→Dec + sidechain) for the latency path; conditional + knobs can follow as a thin Stage 7 without extract.

**Host:** JUCE. **Engine:** JUCE-free library `brave_rt`. **Inference style:** RTNeural-class (prealloc, streaming caches) — paper §IV-F / Table V.

**Encode policy (locked):** deterministic **mean** of variational output (`z_raw.chunk(2,1)[0]`), full **128-D**, **no** `latent_mean` / `latent_pca`. Matches CycleGAN export and Minifusion-like hosts; differs from stock RAVE nn~ VariationalScriptedRAVE.

```mermaid
flowchart LR
  S0[S0 goldens] --&gt; S1[S1 PQMF]
  S1 --&gt; S2[S2 vanilla engine]
  S2 --&gt; S3[S3 JUCE shell]
  S3 --&gt; S4[S4 CycleGAN]
  S4 --&gt; S5[S5 sidechain]
  S5 --&gt; S6[S6 polish]
```

Companion notes: [`scratchpaper/open_brave_plugin_plan.md`](../../scratchpaper/open_brave_plugin_plan.md), [`scratchpaper/realtime_timbre_transfer_hosts.md`](../../scratchpaper/realtime_timbre_transfer_hosts.md).

---

## Repo layout (locked: in-tree [`BRAVE/brave_rt/`](brave_rt/))

Not a sibling repo. All engine, plugin, Python export helpers, schemas, and engine unit tests live under the BRAVE git root:

```
BRAVE/brave_rt/
  cpp/
    brave_rt/                 # JUCE-free engine
      include/brave_rt/
        types.hpp             # float32, shapes, Cr=128
        pqmf.hpp
        conv1d_cached.hpp     # causal pad rings + stride delay
        conv_transpose_cached.hpp
        encoder.hpp
        decoder.hpp
        residual_stack.hpp
        warp.hpp              # CycleGAN LatentCanonicalizer
        sidechain.hpp         # Stage 5
        brave_engine.hpp      # vanilla Enc-Dec
        cyclegan_engine.hpp   # Stage 4
        bundle_loader.hpp
      src/…
      tests/                  # Catch2 or googletest vs golden bins
      tools/
        rtf_bench.cpp
        parity_cli.cpp        # load bundle + wav, write out, print SNR
        pqmf_roundtrip.cpp
    plugin/                   # JUCE
      BraveRT.jucer or CMake + JUCE
      PluginProcessor.{h,cpp}
      PluginEditor.{h,cpp}
  python/
    export_brave_rt.py        # --host brave_rt entry (or extend export_model.py)
    dump_layer_refs.py        # write per-layer .npy goldens from Torch
    parity_check.py           # compare C++ wav vs Torch wav
  schemas/
    brave_rt.schema.json      # filled in Stage 0
  docs/
    README.md                 # how to spoonfeed agents; stage index
    PLAN.md                   # full implementer plan (this document, checked in)
    STAGE_00_goldens.md       # per-stage brief for agents
    STAGE_01_pqmf.md
    STAGE_02_engine.md
    STAGE_03_juce.md
    STAGE_04_cyclegan.md
    STAGE_05_sidechain.md
    STAGE_06_polish.md
    STAGE_07_knobs.md
    STAGE_08_extract.md
  third_party/
    RTNeural/                 # submodule later (Stage 2)
```

### Docs location

Living spec is this file (`brave_rt/docs/PLAN.md`). Per-stage agent briefs are `STAGE_0N_*.md` beside it. Tree under `brave_rt/` is seeded; implement stages in order.

---

## Locked architecture constants (`configs/brave.gin`)

| Constant | Value |
|----------|-------|
| `sr` | 44100 |
| `CAPACITY` | 32 |
| `N_BAND` | 16 |
| `LATENT_SIZE` | 128 |
| `RATIOS` | `[2,2,2,1]` |
| `Cr` | **128** (=16×8) |
| PQMF atten | 40 dB |
| Causal | yes (`cc.get_padding.mode='causal'`) |
| Conv bias (AE) | **False** |
| Decoder noise | **off** (`use_noise=False`) |
| Residual dilations | `[[3,1],[9,1],[27,1],[36,1]]`, kernel 3 |
| Norm | weight_norm on decoder (fuse at export); encoder uses BN |

### Tick tensor pipeline (vanilla)

```
x:       (1, 1, 128)
PQMF↓:   (1, 16, 8)
Enc raw: (1, 256, 1)   # n_out=2
z mean:  (1, 128, 1)
Dec mb:  (1, 16, 8)
PQMF↑:   (1, 1, 128)
```

### Encoder map (CAPACITY=32) — [`RAVE/rave/blocks.py`](../../RAVE/rave/blocks.py)

| Stage | In→Out | k / stride |
|-------|--------|------------|
| stem Conv | 16→32 | 7 / 1 |
| BN + LReLU(0.2) | 32 | |
| ds0 | 32→64 | 5 / 2 |
| ds1 | 64→128 | 5 / 2 |
| ds2 | 128→256 | 5 / 2 |
| ds3 | 256→512 | 3 / 1 |
| head (groups=2) | 512→256 | 5 / 1 |

BN eval: use `running_mean` / `running_var` / `weight` / `bias`.

### Decoder map

| Stage | In→Out | notes |
|-------|--------|-------|
| project | 128→512 | WN Conv k=7 |
| up0 + Res@256 | 512→256 | ConvT k=4 s=2 |
| up1 + Res@128 | 256→128 | |
| up2 + Res@64 | 128→64 | |
| up3 + Res@32 | 64→32 | Conv k=3 s=1 (ratio 1) |
| wave | 32→16 | tanh |
| loud | 32→1 | `mod_sigmoid`: `2*sigmoid(x)^2.3 + 1e-7` ([`RAVE/rave/core.py`](../../RAVE/rave/core.py)) |
| out | wave * loud | multiband |

Each ResidualStack: 4 ResidualLayers, dil pairs `[3,1]…[36,1]`; residual path needs **AlignBranches** delay match (critical for buzz-free streaming).

### PQMF streaming — [`RAVE/rave/pqmf.py`](../../RAVE/rave/pqmf.py) `CachedPQMF`

- `forward_conv.weight`: **(16, 1, 129)**, stride **16**
- `inverse_conv.weight`: **(16, 16, 9)**
- Also store `hk (16,128)`, `h` prototype for debug
- Match `reverse_half`, inverse `* n_band`, channel flip / polyphase interleave exactly
- Causal left-pad rings on both convs (cached_conv semantics)

### cached_conv semantics to port

From site-packages `cached_conv/convs.py` (v2.5.0):

- Causal pad: all pad on left; ring buffer updated each call
- `CachedConv1d`: pad + conv + optional downsampling delay crop
- `CachedConvTranspose1d`: overlap-add cache across blocks
- **Zero all pads/caches at load** (match `_zero_cached_conv_state` after export warmup)

---

## Bundle format (`brave_rt`)

```
my_model/
  brave_rt.json
  weights.h5
  # CycleGAN only:
  cyclegan_latent.manifest.json   # copy from training
```

### `brave_rt.json` (minimum)

```json
{
  "format_version": 1,
  "model_type": "brave",
  "sample_rate": 44100,
  "n_channels": 1,
  "n_band": 16,
  "latent_size": 128,
  "capacity": 32,
  "ratios": [2, 2, 2, 1],
  "compression_ratio": 128,
  "pqmf_attenuation_db": 40,
  "deterministic_encode": true,
  "use_noise": false,
  "dilations_list": [[3,1],[9,1],[27,1],[36,1]],
  "residual_kernel_sizes": [3]
}
```

CycleGAN extras (`model_type": "cyclegan_xy"`):

```json
{
  "geometry": "separate",
  "shared_backbone": false,
  "init_mode": "random",
  "latent_n_layers": 2,
  "latent_hidden_size": 128,
  "encode_hop": 128,
  "sidechain": {
    "present": true,
    "default_enabled": true,
    "knee_db": -40.0,
    "smooth_frames": 8
  },
  "weight_prefixes": {
    "pqmf_x": "pqmf_x.",
    "encoder_x": "encoder_x.",
    "warp_xy": "warp_xy.",
    "decoder_y": "decoder_y.",
    "pqmf_y": "pqmf_y."
  }
}
```

### Weight export rules (Python)

Extend [`scripts/export_model.py`](../../scripts/export_model.py) with `--host brave_rt` (do **not** reuse CycleGAN→h5; today CycleGAN is nn-only).

1. `cc.use_cached_conv(False)` while dumping weights (C++ owns state).
2. `remove_weight_norm` so h5 has fused `*.weight`.
3. Strip discriminator / audio distance / `pad` / `cache` buffers (same idea as [`scripts/export_brave_plugin.py`](../../scripts/export_brave_plugin.py)).
4. Write tensors under **stable prefixed names** listed in manifest (prefer remapping over raw `encoder.encoder.net.0.weight` chaos — but keep a key map JSON for debugging).
5. Include encoder BN running stats.
6. Omit unused `latent_pca` / `latent_mean` from runtime load (may leave in file unused).

Vanilla key map reference (post-strip h5 style):

- PQMF: `pqmf.forward_conv.weight`, `pqmf.inverse_conv.weight`
- Enc: `encoder.encoder.net.{0,3,6,9,12,14}.weight` + BN `.{1,4,7,10}.*`
- Dec: `decoder.net.*`, `decoder.synth.branches.{0,1}.weight`

Warp (n_layers=2, random init): `warp_xy.conv1.weight/bias`, `warp_xy.conv2.weight/bias`, `warp_xy.alpha` (alpha unused in random forward — still load).

**Warp forward (locked by `init_mode`):**

- `random` (separate CycleGAN): `z_out = MLP(z)` only — [`latent_canonicalizer.py`](../../RAVE/rave/canonicalizer/latent_canonicalizer.py)
- `identity` (joint): gated residual `z + sigmoid(alpha)*(warped-z)` with n_layers=2 inner `z+h`

---

## Stage 0 — Goldens and schema

**Do first. No C++.**

### Tasks

1. Choose vanilla BRAVE ckpt (e.g. tap foundation `epoch_1500000.ckpt` from CycleGAN manifest X backbone).
2. Export TorchScript streaming + run [`evaluation/scripts/nn_block_reconstruct.py`](../../evaluation/scripts/nn_block_reconstruct.py) at **128** and **512** on a fixed mono tap WAV (e.g. presentation demo tap).
3. Also dump **mean-encode** full-context reference via a small Python script that mirrors C++ policy (PQMF→Enc mean→Dec→iPQMF) — do **not** use VariationalScriptedRAVE sampling/PCA as the parity oracle.
4. Write `schemas/brave_rt.schema.json` + example `brave_rt.json`.
5. `python/dump_layer_refs.py`: for one 128-sample block (and a longer clip), save `.npy` for: PQMF bands, z_mean, decoder multiband, waveform.
6. Document SNR / max-abs thresholds (start with max abs &lt; 1e-4 on PQMF RT; relax to 1e-3 E2E if needed; record final gates in `docs/IMPLEMENTATION.md`).

### Exit

- Golden WAVs + `.npy` layer refs checked into `brave_rt/testdata/` (or BRAVE-data path documented).
- Schema frozen.

---

## Stage 1 — Streaming PQMF (C++)

### Tasks

1. Implement `PqmfAnalysis` / `PqmfSynthesis` with preallocated rings; load weights from h5.
2. Match PyTorch `CachedPQMF.forward` / `inverse` bit-closely on random and impulse inputs.
3. CLI `pqmf_roundtrip`.
4. Unit test vs Stage 0 `.npy`.

### Pitfalls

- `make_odd` / 129-tap analysis; inverse 9-tap polyphase; `reverse_half`; `*16` on inverse.
- State continuity across successive 128-sample calls (not only one-shot).

### Exit

Round-trip SNR within Stage 0 PQMF gate on ≥1 s audio, streaming in 128-sample blocks.

---

## Stage 2 — Vanilla `BraveEngine`

### Tasks

1. Implement cached Conv1d / ConvTranspose1d / ResidualStack / AlignBranches delays.
2. Encoder + Decoder as tables above; LReLU 0.2; BN eval; groups=2 head.
3. Fuse or load already-fused WN weights.
4. `BraveEngine::load(bundle)`; `resetState()` zeros caches; `process(float* in, float* out, int n)`:
   - If `n == 128`: one tick.
   - If `n` multiple of 128: loop ticks.
   - If DAW gives other sizes: internal FIFO accumulate/flush (document; Stage 3 needs this).
5. Layer-wise parity CLIs: `--stage pqmf|encode|decode|e2e`.
6. `rtf_bench`: 1100 iters, discard 100 warmup; report mean/std RTF @ 128/256/512 (paper method).

### RTNeural strategy

- Submodule RTNeural; **expect to extend** for strided + transposed + dilation caches (paper did this).
- Fallback acceptable for prototype: hand-written Eigen/plain loops for one layer type if RTNeural blocked — but keep API identical.

### Exit

- E2E parity vs mean-encode Torch goldens @ block 128.
- **RTF &lt; 1.0 @ 128** on target Mac (M-series preferred; record machine).
- Long soak 10+ min no NaNs / growth.

---

## Stage 3 — Thin JUCE plugin (vanilla)

### Tasks

1. JUCE 7/8 CMake project; formats **VST3 + AU**; macOS first.
2. `PluginProcessor` owns `BraveEngine`; prepareToPlay allocates nothing per callback beyond FIFO.
3. Editor: load bundle (folder picker), bypass, output gain, status (sr, RTF estimate optional).
4. Warn if sample rate ≠ 44100.
5. `setLatencySamples(...)` once PQMF+buffer policy known (document formula in UI tooltip).
6. Manual test: Reaper or Logic, buffer 128/256/512.

### Exit

Playable vanilla BRAVE in DAW at 128 without underruns; audible match to offline recon.

---

## Stage 4 — CycleGAN engine + plugin path

### Graph (match [`cyclegan_nn.py`](../../RAVE/rave/canonicalizer/export/cyclegan_nn.py))

```
x → pqmf_x → Enc_X → mean → warp_xy → Dec_Y → pqmf_y.inverse
  → trim length to x
  → (sidechain deferred to Stage 5; Stage 4 ships with sidechain hard-off)
```

### Tasks

1. Python `export_brave_rt` for CycleGAN: dual prefixes; read [`cyclegan_latent.manifest.json`](../../docs/cyclegan/README.md) for `init_mode`, `geometry`, `latent_n_layers`.
2. `CycleGanEngine`: two PQMF states even if filter weights identical (separate geometry).
3. Warp n_layers=2, hidden 128; honor `init_mode`.
4. Parity vs `babbling_brook_10122026.ts` block-128 offline (same tap as prior diagnosis) **with sidechain disabled** on both sides (`set_sidechain 0` / attribute).
5. JUCE: detect `model_type=cyclegan_xy` and swap engine; no new UI except name in title.

### Exit

CycleGAN bundle RTF &lt; 1 @ 128; parity with sidechain off; DAW demo of tap→babbling brook at 128.

---

## Stage 5 — Loudness sidechain (later, as requested)

Port [`CausalLoudnessSidechain`](../../RAVE/rave/canonicalizer/silence_gate.py) — **not** the offline boxcar helper.

| Param | Value |
|-------|-------|
| `hop` | 128 (`encode_hop`) |
| `knee_db` | −40 → linear knee `10^(-40/20)=0.01` |
| `smooth_frames` | 8 → `α = 1/8` |
| state | scalar `env` across blocks |

Per hop: mono mean → RMS → `env = (1-α)env + α*r` → `g = env/(env+knee)` → expand by hop (tail = last g).

### Tasks

1. `Sidechain` class in engine; apply after iPQMF trim.
2. JUCE parameter / button `sidechain` default **on** (match nn~).
3. Parity: enable sidechain on TorchScript + C++; compare on tap with silence gaps.
4. Tests mirroring [`RAVE/tests/test_cyclegan_silence_gate.py`](../../RAVE/tests/test_cyclegan_silence_gate.py).

### Exit

Sidechain on/off matches nn~; quiet input ducks output without hard mute clicks.

---

## Stage 6 — Polish and BRAVE repo integration

1. Wire `python scripts/export_model.py --host brave_rt` (vanilla + cyclegan).
2. CI: build engine tests + parity smoke (Linux CI OK; AU may be mac-only).
3. Update [`CLAUDE.md`](../../CLAUDE.md) Key Commands; mark hosts table in scratchpaper.
4. Example bundles README: install VST3/AU, set DAW buffer 128, 44.1 kHz.
5. Optional: Y→X engine using same class with swapped weights (`export_cyclegan_yx` pattern).

### Backlog (not S0–S6)

- Waveform canonicalizer
- Max external wrapping `brave_rt`
- Windows until macOS green
### Stage 7 (future, optional) — Conditional / Fader with **knobs only**

**Why:** Live conditional CycleGAN / FaderRAVE can be driven by host sliders/menus (`attr_mode=1`): normalize with `attribute_stats.yaml`, broadcast to `T_lat`, concat or condition as in training. No extract required.

**Sketch:** widened Dec or conditional warp inputs; JUCE one control per attribute; parity vs nn~ with `set_attr_mode(1)`.

### Stage 8 (future, optional) — Attribute extract

Hands-free descriptors (`TorchDescriptorExtract` / gin providers) for automatic attrs. Nice-to-have after knobs work; profile RTF @ 128 carefully. See [`docs/fader_host_controls.md`](../../docs/fader_host_controls.md).

---

## Testing matrix (every stage that touches audio)

| Test | When |
|------|------|
| Layer `.npy` SNR | S1–S2 |
| Block-128 E2E vs mean-Torch | S2, S4 |
| Block-128 E2E vs `.ts` sidechain off | S4 |
| Block-128 E2E vs `.ts` sidechain on | S5 |
| RTF bench 128/256/512 | S2, S4 |
| DAW manual 128 buffer | S3+ |
| 10 min soak | S2, S4 |

---

## Risk register

| Risk | Mitigation |
|------|------------|
| RTNeural missing strided/transpose | Budget S2; hand-roll with same API |
| Residual delay mismatch | Layer-wise parity before E2E |
| Wrong encode (sample/PCA) | Locked mean-128; goldens from mean path |
| Dual PQMF state bug | Separate rings; test separate geometry |
| `init_mode` wrong | Read manifest; unit test warp alone |
| Sidechain boxcar vs IIR | Port module only (S5) |
| JUCE friction | Engine CLI ships before plugin (S2 before S3) |

---

## Suggested agent execution order

1. Agent A: Stage 0 + Python `export_brave_rt` stub + schema.
2. Agent B: Stage 1 PQMF (can start after schema + one h5 dump).
3. Agent C: Stage 2 engine (depends on S1).
4. Agent D: Stage 3 JUCE (depends on S2 RTF+parity).
5. Agent E: Stage 4 CycleGAN (depends on S2; can parallelize export with S3).
6. Agent F: Stage 5 sidechain (depends on S4).
7. Agent G: Stage 6 docs/CI.

**Docs bootstrap is done** in this tree. Spoonfeed agents one `STAGE_0N_*.md` at a time; start with [`STAGE_00_goldens.md`](STAGE_00_goldens.md).
