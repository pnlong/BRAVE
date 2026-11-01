# Parity gates (Stage 0+)

Initial thresholds for float32 C++ vs mean-encode PyTorch. Tighten after first green runs.

| Check | Metric | Gate |
|-------|--------|------|
| PQMF round-trip / layer | max abs error | &lt; **1e-4** |
| Encode / decode layer | max abs error | &lt; **1e-3** |
| E2E waveform @ block 128 | max abs error | &lt; **1e-3** (relax to 5e-3 if documented) |
| E2E waveform | SNR | ≥ **60 dB** preferred |

Machine and commit hash used for RTF must be recorded next to `rtf_bench` results.

Oracle path: **mean encode**, full 128-D, **no** PCA / `latent_mean` (see PLAN.md).
