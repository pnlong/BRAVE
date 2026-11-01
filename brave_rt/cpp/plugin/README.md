# BraveRT JUCE plugin (Stages 3–5)

Thin VST3 (+ AU on macOS) shell around `BraveEngine` / `CycleGanEngine`.

## Prerequisites

- JUCE 8 under `brave_rt/third_party/JUCE` (already vendored)
- CMake ≥ 3.22 (`pip install cmake` into the brave env is enough)
- **macOS (preferred):** Xcode + JUCE; AU + VST3
- **Linux:** ALSA, FreeType, Fontconfig (+ GL). If `-dev` packages are missing without root, extract them into `brave_rt/third_party/linux_sysroot` (see `build_plugin.sh`)

## Build

```bash
# From BRAVE repo — uses Release + -march=native (RTF flags)
bash brave_rt/cpp/plugin/build_plugin.sh
# artefacts under cpp/plugin/build_rel/BraveRT_artefacts/...
```

Or manually:

```bash
export PATH="/data/hai-res/p1long/micromamba/envs/brave/bin:$PATH"
cd brave_rt/cpp/plugin
cmake -S . -B build_rel -DBRAVE_RT_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_rel -j
```

## DAW checklist

1. Install the VST3/AU into the host’s plug-in folder.
2. Set project to **44.1 kHz**, buffer **128** (also try 256/512).
3. Load a bundle folder (`brave_rt.json` + `npy_weights/`), e.g.
   - vanilla: `testdata/bundles/babbling_brook_foundation_y_70d`
   - CycleGAN: `testdata/bundles/tap_babbling_brook_cyc` (Sidechain toggle)
4. Compare audibly to goldens under `testdata/goldens/`.

## Latency

`setLatencySamples(128 + engine.latency_samples())`:

- `128` — model onset / PQMF pad used for host delay compensation (`kModelLatencySamples`)
- `engine.latency_samples()` — FIFO prime for non-multiple-of-128 host blocks (0 when block % 128 == 0)

## UI

- **Load bundle...** — folder picker
- **Block size** — choice `128` (default) / `256` / `512`. Match this to the DAW I/O buffer (Logic → Preferences → Audio → Buffer Size). Status warns on mismatch. Model still ticks at Cr=128 internally; larger host blocks are batched.
- **Output gain (dB)** — −24 … +12
- **Sidechain** — CycleGAN only; default **on** (matches nn~). Disabled for vanilla AE bundles.
- Bypass via host parameter
- Status / title show bundle display name (`brave_rt.json` `display_name`)
