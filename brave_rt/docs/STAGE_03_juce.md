# Stage 3 — Thin JUCE plugin (vanilla)

**Goal:** Playable vanilla BRAVE in a DAW at 128-sample buffer without underruns.

**Depends on:** Stage 2 RTF + parity.  
**Blocks:** CycleGAN UI wiring (Stage 4 extends this shell).

## In scope

1. JUCE 7/8 CMake (or Projucer) under `cpp/plugin/`; formats **VST3 + AU**; macOS first.
2. `PluginProcessor` owns `BraveEngine`; no malloc in `processBlock` beyond prepared FIFO.
3. Editor: bundle folder picker, bypass, output gain, **block size choice** (128 default / 256 / 512; match DAW I/O), sr warning if ≠ 44100.
4. `setLatencySamples` with documented formula (PQMF delay + buffering policy).
5. Manual checklist: Reaper/Logic @ 128/256/512.

## Out of scope

CycleGAN, sidechain toggle, Fader knobs, Windows.

## Exit criteria

- [x] Plugin sources + CMake build produce VST3 (Linux verified: `build_rel3/.../BraveRT.vst3`)
- [ ] Plugin loads Stage 0/2 vanilla bundle in a DAW (manual)
- [ ] Audible match to offline recon at 128 buffer (manual)
- [ ] No underruns at 128 on target Mac (manual; AU build on macOS)

## Suggested agent prompt

> Implement only Stage 3 per `brave_rt/docs/STAGE_03_juce.md`. Thin shell around existing BraveEngine.
