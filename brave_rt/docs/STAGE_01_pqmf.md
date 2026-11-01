# Stage 1 — Streaming PQMF (C++)

**Goal:** Causal polyphase `CachedPQMF` analysis/synthesis in C++ with preallocated rings; parity vs Stage 0 `.npy`.

**Depends on:** Stage 0 exit.  
**Blocks:** Stage 2.

## In scope

1. Headers/sources: `cpp/brave_rt/include/brave_rt/pqmf.hpp`, `src/pqmf.cpp`.
2. Load `pqmf.forward_conv.weight (16,1,129)` and `pqmf.inverse_conv.weight (16,16,9)` from Stage 0 h5.
3. Match PyTorch `CachedPQMF.forward` / `inverse` including `reverse_half`, `*n_band`, polyphase layout.
4. Causal left-pad rings (cached_conv semantics); `resetState()` zeros rings.
5. CLI `cpp/brave_rt/tools/pqmf_roundtrip.cpp`.
6. Unit test vs Stage 0 layer refs; streaming across successive 128-sample blocks.

## Pitfalls

- `make_odd` → 129-tap analysis; inverse 9-tap.
- One-shot vs multi-block state continuity.
- Float32 only.

## Out of scope

Encoder/decoder, JUCE, CycleGAN.

## Exit criteria

- [ ] Round-trip SNR / max-abs within Stage 0 PQMF gate on ≥1 s audio, **128-sample streaming**
- [ ] Impulse / random tests pass
- [ ] CLI documented in `brave_rt/README.md` or tool `--help`

## Suggested agent prompt

> Implement only Stage 1 per `brave_rt/docs/STAGE_01_pqmf.md`. Use Stage 0 goldens for parity.
