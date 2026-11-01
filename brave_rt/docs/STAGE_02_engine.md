# Stage 2 — Vanilla BraveEngine

**Goal:** Full Enc→Dec streaming engine; E2E parity @ block 128; **RTF &lt; 1 @ 128** on target Mac.

**Depends on:** Stage 1.  
**Blocks:** Stage 3–4.

## In scope

1. Cached Conv1d / ConvTranspose1d / ResidualStack with AlignBranches delay match.
2. Encoder + Decoder per [`PLAN.md`](PLAN.md) tables; LReLU 0.2; BN eval; groups=2 head; `mod_sigmoid`.
3. Mean-only encode (`chunk(2)[0]`).
4. `BraveEngine::load`, `resetState`, `process(in,out,n)` with Cr=128 ticks + FIFO for other `n`.
5. RTNeural submodule under `third_party/RTNeural` **or** hand-rolled loops with same API (fallback OK).
6. Tools: `parity_cli`, `rtf_bench` (1100 iters, drop 100 warmup; report 128/256/512).
7. Layer-wise parity: `pqmf|encode|decode|e2e`.

## Out of scope

JUCE UI, CycleGAN warp, sidechain.

## Exit criteria

- [x] E2E max-abs / SNR vs mean-encode goldens @ 128 within gates (2.9e-6 / 108.5 dB; see STATUS)
- [~] RTF &lt; 1.0 @ 128 on documented machine: Xeon w7-3565X bench 0.41 mean, but the sustained soak was ≈1.0 under shared-box load; **target Mac still to record** (see STATUS)
- [x] 10+ min soak: no NaN, no state blow-up (11 min, 0 non-finite, peak flat; see STATUS)

## Suggested agent prompt

> Implement only Stage 2 per `brave_rt/docs/STAGE_02_engine.md` and PLAN encoder/decoder maps.
