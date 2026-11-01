# Stage 8 — Attribute extract (optional)

**Goal:** Hands-free descriptors for automatic attrs (`TorchDescriptorExtract` / gin providers).

**Depends on:** Stage 7 knobs working.  
**Not a blocker** for live conditional use (Stage 7 is enough).

## In scope

- Wire extract path into BraveRT / plugin (or slower helper thread)
- Match training providers where feasible
- Parity vs nn~ extract modes where applicable
- Profile **RTF @ 128** carefully — may need one-hop delay / non-audio thread

## Out of scope

New attribute taxonomies; Max-only wiring (see [`docs/fader_host_controls.md`](../../docs/fader_host_controls.md)).

## Explicitly deferred (whole roadmap)

- Waveform canonicalizer
- Max external wrapping `brave_rt`
- Windows (until macOS green)

## Implementation (landed 2026-10-31)

- `include/brave_rt/descriptor_extract.hpp` + `src/descriptor_extract.cpp` — `DescriptorExtract`: 131072-sample rolling history, n_fft=2048 / hop=512 Hann STFT (own radix-2 FFT), all scratch preallocated. Computes **rms, centroid, flatness, zcr** in raw training units; **f0 skipped** (stays manual/default). One frame per 512 new samples (trailing window, no look-ahead); output = mean of last 8 frames (`set_smoothing_frames`). `ready()` after 131072 samples since `reset()` (= nn~ `_extract_history_ready`).
- `BraveEngine::set_attr_mode(int)` / `attr_mode()` — **1 manual (default, Stage 7 path untouched)**, 2 extract-only, 0 extract + per-attr override (`set_attr_override`; scale knobs not implemented). Manual values (`set_attr_raw`) are never overwritten; the decoder sees `attr_effective()`. Extracted values are clamped to `[min,max]` like nn~ `_clamp_raw`. Discrete + non-extractable (timbral, f0) rows always use the manual/default value. Reset on `reset_state()` / load; entering extract from manual restarts the history.
- Runs on the audio thread (cost ≈ 21 µs mean per 128-sample block, ≈ 0.007 RTF; one FFT frame per 4 blocks) — no helper thread needed.
- `CycleGanEngine`: not wired (shipped CycleGAN bundles have D=0).
- Plugin: `attr_mode` `AudioParameterChoice` (Manual / Extract / Extract+manual → engine 1 / 2 / 0), persisted in state XML (`attr_mode_index`), synced every block, ComboBox next to the Conditioning title, hidden when the bundle has no attributes (or CycleGAN).
- Tools: `extract_smoke` (analytic signals + mode semantics), `rtf_bench … [attr_mode]`, `fader_parity_cli --attr-mode`.

### Known differences vs nn~

- nn~ recomputes a centred STFT over the whole 3 s history each block and resamples to `T_lat`; BraveRT uses incremental trailing frames and one scalar per block (smoother, causal, much cheaper).
- nn~'s torch module emits an all-zero row for `zcr`/`f0` (only rms/centroid/flatness are implemented there); BraveRT computes real zcr, skips f0.
- nn~ `_row_scale` calibration vs librosa is not applied (scale = 1; `DescriptorExtract::set_scale` available).

## Exit criteria

- [x] Extract usable without underruns at documented buffer / machine (`rtf_bench` @128, `taskset -c 0`, water_fader_scene: mean RTF < 1 in both modes; see `STATUS.md`)
- [x] Docs note RTF impact vs knob-only (`STATUS.md` Stage 8 section)
- [x] Stage 7 parity unchanged (`attr_mode=1` output bit-identical to `stream128_cpp.npy`)

## Suggested agent prompt

> Implement only Stage 8 per `brave_rt/docs/STAGE_08_extract.md` after Stage 7 is green.
