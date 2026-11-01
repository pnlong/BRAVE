# Stage 4 — CycleGAN engine (sidechain off)

**Goal:** `Enc_X → W_xy → Dec_Y` in C++ + plugin; RTF &lt; 1 @ 128; parity vs `.ts` with **sidechain disabled**.

**Depends on:** Stage 2 (engine). Stage 3 shell can be extended in parallel after S2.  
**Blocks:** Stage 5.

## Graph

```
x → pqmf_x → Enc_X → mean → warp_xy → Dec_Y → pqmf_y.inverse → trim to x
→ sidechain HARD OFF in this stage
```

## In scope

1. Extend `export_brave_rt.py` for `model_type=cyclegan_xy`; dual prefixes; read training `cyclegan_latent.manifest.json` for `init_mode`, `geometry`, `latent_n_layers`.
2. `CycleGanEngine`: **separate** PQMF cache rings even if weights match.
3. Warp n_layers=2, hidden 128; `init_mode=random` → MLP only; `identity` → gated residual (see PLAN).
4. Parity vs e.g. `BRAVE-data/exports/cyclegan_ts_done/babbling_brook_10122026.ts` at block 128 with `set_sidechain(false)`.
5. JUCE: load cyclegan bundles; title/status only (no sidechain UI yet).

## Out of scope

Sidechain implementation (Stage 5), Fader, Y→X (optional polish).

## Reference exports

- TS: `BRAVE-data/exports/cyclegan_ts_done/babbling_brook_10122026.ts`
- Manifest example: `BRAVE/exports/tap_babbling_brook_wf_cyc_mlp2/cyclegan_latent.manifest.json`
- Python graph: `RAVE/rave/canonicalizer/export/cyclegan_nn.py`

## Exit criteria

- [ ] RTF &lt; 1 @ 128 for CycleGAN bundle
- [ ] Parity vs `.ts` sidechain-off within gates
- [ ] DAW demo tap→Y at 128

## Suggested agent prompt

> Implement only Stage 4 per `brave_rt/docs/STAGE_04_cyclegan.md`. Keep sidechain hard-off.
