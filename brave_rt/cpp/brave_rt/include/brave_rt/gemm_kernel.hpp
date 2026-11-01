#pragma once

#include <cstddef>
#include <cstring>

#if defined(__AVX512F__) || defined(__F16C__)
#include <immintrin.h>
#endif

// vf16 is 64 bytes; on targets without AVX-512 GCC notes the (inline-only)
// vector-return ABI. Harmless here: these helpers never cross a TU boundary.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wpsabi"
#endif

namespace brave_rt {

/**
 * Packed GEMM micro-kernel shared by every conv / convT layer.
 *
 *   out[t, ob*16 + i] = sum_j col[t, j] * Wp[ob, j, i]
 *
 * Weights are packed in panels of kOB=16 output channels, [ob][j][16], stored
 * as fp32 or fp16 (`_Float16`, widened in-register). Each panel is streamed
 * sequentially once per tile of up to 4 time steps, and all accumulators
 * stay in registers for the whole reduction over j.
 *
 * At Cr=128 the engine is weight-bandwidth bound (T = 1–8 per tick), so
 * fp16 storage halves the dominant cost. Written with GCC/Clang vector
 * extensions → AVX2/AVX-512 on x86, NEON on Apple silicon; no intrinsics.
 */
constexpr int kOB = 16;

typedef float vf16 __attribute__((vector_size(kOB * sizeof(float))));
typedef _Float16 vh16 __attribute__((vector_size(kOB * sizeof(_Float16))));

inline vf16 load_panel(const float* p) {
  vf16 v;
  std::memcpy(&v, p, sizeof v);
  return v;
}

// GCC lowers __builtin_convertvector(_Float16→float) to scalar vcvtsh2ss on
// x86, so use the packed hardware conversions there; Clang/NEON is fine.
inline vf16 load_panel(const _Float16* p) {
#if defined(__AVX512F__)
  return (vf16)_mm512_cvtph_ps(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
#elif defined(__F16C__)
  const __m256 lo =
      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  const __m256 hi =
      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8)));
  vf16 v;
  std::memcpy(&v, &lo, sizeof lo);
  std::memcpy(reinterpret_cast<char*>(&v) + sizeof lo, &hi, sizeof hi);
  return v;
#else
  vh16 h;
  std::memcpy(&h, p, sizeof h);
  return __builtin_convertvector(h, vf16);
#endif
}

template <class WT, int TT>
inline void gemm_tile(const float* __restrict col, int ldc, int J,
                      const WT* __restrict Wb, float* __restrict out,
                      int ldo) {
  vf16 acc[TT];
  for (int tt = 0; tt < TT; ++tt) acc[tt] = vf16{};
  for (int j = 0; j < J; ++j) {
    const vf16 w = load_panel(Wb + static_cast<size_t>(j) * kOB);
    for (int tt = 0; tt < TT; ++tt) acc[tt] += w * col[tt * ldc + j];
  }
  for (int tt = 0; tt < TT; ++tt)
    std::memcpy(out + static_cast<size_t>(tt) * ldo, &acc[tt], sizeof(vf16));
}

/**
 * col: (T, ldc) row-major, first J columns used.
 * Wp: (nob, J, 16) packed panels.
 * out: (T, ldo) row-major; writes columns [0, nob*16). Overwrites.
 */
template <class WT>
inline void gemm_packed(const float* col, int T, int ldc, int J, const WT* Wp,
                        int nob, float* out, int ldo) {
  for (int ob = 0; ob < nob; ++ob) {
    const WT* Wb = Wp + static_cast<size_t>(ob) * J * kOB;
    float* o = out + ob * kOB;
    int t = 0;
    for (; t + 4 <= T; t += 4)
      gemm_tile<WT, 4>(col + static_cast<size_t>(t) * ldc, ldc, J, Wb,
                       o + static_cast<size_t>(t) * ldo, ldo);
    const float* c = col + static_cast<size_t>(t) * ldc;
    float* oo = o + static_cast<size_t>(t) * ldo;
    switch (T - t) {
      case 3: gemm_tile<WT, 3>(c, ldc, J, Wb, oo, ldo); break;
      case 2: gemm_tile<WT, 2>(c, ldc, J, Wb, oo, ldo); break;
      case 1: gemm_tile<WT, 1>(c, ldc, J, Wb, oo, ldo); break;
      default: break;
    }
  }
}

inline int round_up_ob(int n) { return (n + kOB - 1) / kOB * kOB; }

}  // namespace brave_rt
