# Stage 0 — Goldens and schema

**Goal:** Frozen mean-encode parity oracles, `brave_rt.json` schema, and a Python export stub. **No C++.**

**Depends on:** Docs bootstrap (done).  
**Blocks:** Stage 1+.

## In scope

1. Pick vanilla BRAVE ckpt (e.g. tap foundation `epoch_1500000.ckpt` from a CycleGAN X backbone manifest under `BRAVE-data/` or `exports/`).
2. Mean-encode reference path: PQMF → Enc → `z_raw.chunk(2,1)[0]` → Dec → iPQMF. **Do not** use VariationalScriptedRAVE sampling/PCA as the oracle.
3. Scripts under `brave_rt/python/`:
   - `dump_layer_refs.py` — write `.npy` for PQMF bands, z_mean, decoder multiband, waveform (one 128-sample block + longer clip).
   - `export_brave_rt.py` stub — dump fused weights + `brave_rt.json` (vanilla first).
4. `schemas/brave_rt.schema.json` + example `schemas/examples/brave_vanilla.json`.
5. Optional: `nn_block_reconstruct.py` @ 128/512 for listening only (not the parity oracle if it uses PCA path).
6. Record parity gates in `docs/PARITY_GATES.md` (start: PQMF max-abs &lt; 1e-4; E2E relax to 1e-3 if needed).

## Out of scope

C++, JUCE, CycleGAN export, sidechain, Fader.

## References

- [`PLAN.md`](PLAN.md) — shapes, key names, bundle format
- [`../../configs/brave.gin`](../../configs/brave.gin)
- [`../../RAVE/rave/blocks.py`](../../RAVE/rave/blocks.py), [`../../RAVE/rave/pqmf.py`](../../RAVE/rave/pqmf.py)
- [`../../scripts/export_brave_plugin.py`](../../scripts/export_brave_plugin.py) — strip/fuse pattern
- Tap WAV example: `BRAVE-data/presentation_demos/babbling_brook/0/stems/00_tap.wav`

## Exit criteria

- [ ] Golden WAVs / `.npy` under `brave_rt/testdata/` (or path documented in README)
- [ ] Schema frozen and validates example JSON
- [ ] Export stub produces loadable `weights.h5` + `brave_rt.json` for one vanilla model
- [ ] `PARITY_GATES.md` written

## Suggested agent prompt

> Implement only Stage 0 in `BRAVE/brave_rt` per `docs/STAGE_00_goldens.md` and `docs/PLAN.md`. No C++.
