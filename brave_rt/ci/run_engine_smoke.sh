#!/usr/bin/env bash
# Linux CI / local smoke: build engine tools + run PQMF / sidechain / vanilla
# parity (+ CycleGAN parity when goldens+bundle exist).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENG="$ROOT/brave_rt/cpp/brave_rt"
BUILD="$ENG/build_ci"
BUNDLE_V="$ROOT/brave_rt/testdata/bundles/tap_foundation_x_shotgun_contact_70d"
GOLD_V="$ROOT/brave_rt/testdata/goldens/tap_foundation_x_shotgun_contact_70d"
BUNDLE_C="$ROOT/brave_rt/testdata/bundles/tap_babbling_brook_cyc"
GOLD_C="$ROOT/brave_rt/testdata/goldens/cyclegan_brook"

# Prefer brave env cmake/g++ when present (lab boxes often lack system cmake).
if [[ -d /data/hai-res/p1long/micromamba/envs/brave/bin ]]; then
  export PATH="/data/hai-res/p1long/micromamba/envs/brave/bin:$PATH"
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found on PATH" >&2
  exit 1
fi

mkdir -p "$BUILD"
cmake -S "$ENG" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBRAVE_RT_NATIVE=OFF
cmake --build "$BUILD" -j"$(nproc)"

"$BUILD/sidechain_smoke"

if [[ -d "$BUNDLE_V" && -d "$GOLD_V" ]]; then
  "$BUILD/pqmf_roundtrip" \
    "$BUNDLE_V/pqmf_forward_conv_weight.npy" \
    "$BUNDLE_V/pqmf_inverse_conv_weight.npy" \
    "$GOLD_V/full_x.npy" \
    "$GOLD_V/full_pqmf_bands.npy" \
    128
  "$BUILD/parity_cli" "$BUNDLE_V" "$GOLD_V" e2e stream 128
else
  echo "SKIP vanilla PQMF/parity (missing bundle/goldens)"
fi

if [[ -d "$BUNDLE_C/npy_weights" && -f "$GOLD_C/stream128_sc0_y.npy" ]]; then
  "$BUILD/cyclegan_parity_cli" "$BUNDLE_C" \
    "$GOLD_C/stream128_sc0_y.npy" "$GOLD_C/full_x.npy" 0 128
  if [[ -f "$GOLD_C/stream128_sc1_y.npy" ]]; then
    "$BUILD/cyclegan_parity_cli" "$BUNDLE_C" \
      "$GOLD_C/stream128_sc1_y.npy" "$GOLD_C/full_x.npy" 1 128
  fi
else
  echo "SKIP CycleGAN parity (missing bundle/goldens — export + cyclegan_ts_ref first)"
fi

if [[ -f "$GOLD_C/sc_y_ref.npy" ]]; then
  "$BUILD/sidechain_parity_cli" \
    "$GOLD_C/sc_x.npy" "$GOLD_C/sc_y_in.npy" "$GOLD_C/sc_y_ref.npy" 128
fi

# Stage 8: descriptor extractor + attr_mode semantics (Fader bundle optional).
BUNDLE_F="$ROOT/brave_rt/testdata/bundles/water_fader_scene"
if [[ -d "$BUNDLE_F/npy_weights" ]]; then
  "$BUILD/extract_smoke" "$BUNDLE_F"
else
  "$BUILD/extract_smoke"
fi

echo "engine smoke PASS"
