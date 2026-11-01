#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="$ROOT/third_party/linux_sysroot"
export PATH="/data/hai-res/p1long/micromamba/envs/brave/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export CXXFLAGS="-I/usr/include/freetype2 -I/usr/include/libpng16 -I$PREFIX/usr/include ${CXXFLAGS:-}"
export CFLAGS="$CXXFLAGS"
export LDFLAGS="-L/data/hai-res/p1long/BRAVE/brave_rt/third_party/linux_sysroot/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/data/hai-res/p1long/BRAVE/brave_rt/third_party/linux_sysroot/usr/lib/x86_64-linux-gnu ${LDFLAGS:-}"
cd "$(dirname "$0")"
BUILD_DIR="${BUILD_DIR:-build_rel}"
rm -rf "$BUILD_DIR" 2>/dev/null || true
if [[ -e "$BUILD_DIR" ]]; then
  BUILD_DIR="build_$(date +%s)"
fi
echo "Configuring in $BUILD_DIR ..."
cmake -S . -B "$BUILD_DIR" -DBRAVE_RT_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
echo "Building ..."
cmake --build "$BUILD_DIR" -j"$(nproc)"
echo "BUILD_OK dir=$BUILD_DIR"
find "$BUILD_DIR" -type d -name '*.vst3' 2>/dev/null | head
find "$BUILD_DIR" -path '*artefacts*' -type f 2>/dev/null | head -30
