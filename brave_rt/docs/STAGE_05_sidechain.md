# Stage 5 — Causal loudness sidechain

**Goal:** Port `CausalLoudnessSidechain` (streaming IIR), match nn~; JUCE toggle default on.

**Depends on:** Stage 4.  
**Blocks:** Stage 6 polish that documents sidechain.

## Spec (match `RAVE/rave/canonicalizer/silence_gate.py`)

| Param | Value |
|-------|-------|
| hop | 128 |
| knee_db | −40 → linear 0.01 |
| smooth_frames | 8 → α = 1/8 |
| state | scalar `env` across blocks |

Per hop: mono mean → RMS → IIR env → `g = env/(env+knee)` → expand by hop.

**Do not** port the offline boxcar `sidechain_by_input_loudness` helper as the runtime path.

## In scope

1. `sidechain.hpp` / `.cpp`; apply after iPQMF trim.
2. JUCE parameter `sidechain` default **true**.
3. Parity vs TorchScript with sidechain on (silence gaps).
4. Tests akin to `RAVE/tests/test_cyclegan_silence_gate.py`.

## Exit criteria

- [ ] On/off matches nn~ within gates
- [ ] Quiet input ducks without hard-mute clicks
- [ ] Toggle works in DAW

## Suggested agent prompt

> Implement only Stage 5 per `brave_rt/docs/STAGE_05_sidechain.md`. Port CausalLoudnessSidechain module only.
