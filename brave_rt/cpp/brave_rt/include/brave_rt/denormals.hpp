#pragma once

#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define BRAVE_RT_DENORMALS_X86 1
#endif

namespace brave_rt {

/**
 * RAII flush-to-zero / denormals-are-zero for the calling thread (same idea
 * as juce::ScopedNoDenormals). Decaying conv caches and silent input push
 * activations into subnormals, which are 10–100× slower on x86; without this
 * RTF degrades unpredictably during quiet passages. Restores on scope exit.
 */
class ScopedNoDenormals {
 public:
  ScopedNoDenormals() {
#if defined(BRAVE_RT_DENORMALS_X86)
    saved_ = _mm_getcsr();
    _mm_setcsr(saved_ | 0x8040u);  // FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__)
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    saved_ = fpcr;
    asm volatile("msr fpcr, %0" : : "r"(fpcr | (1ull << 24)));  // FZ
#endif
  }
  ~ScopedNoDenormals() {
#if defined(BRAVE_RT_DENORMALS_X86)
    _mm_setcsr(static_cast<unsigned>(saved_));
#elif defined(__aarch64__)
    asm volatile("msr fpcr, %0" : : "r"(saved_));
#endif
  }
  ScopedNoDenormals(const ScopedNoDenormals&) = delete;
  ScopedNoDenormals& operator=(const ScopedNoDenormals&) = delete;

 private:
  uint64_t saved_ = 0;
};

}  // namespace brave_rt
