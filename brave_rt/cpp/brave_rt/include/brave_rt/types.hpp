#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brave_rt {

using float32 = float;

constexpr int kSampleRate = 44100;
constexpr int kNBand = 16;
constexpr int kLatentSize = 128;
constexpr int kCapacity = 32;
constexpr int kCompressionRatio = 128;
constexpr int kTickSamples = 128;

/**
 * Default encoder/decoder weight storage (compute is always fp32; PQMF always
 * fp32). fp16 halves weight traffic (~1.2x faster) but misses the 1e-3 / 60 dB
 * parity gates on the reference bundle (SNR ≈ 55 dB), so fp32 is the default.
 * Per-engine override: BraveEngine::load(dir, Precision{...}).
 */
#ifndef BRAVE_RT_FP16_WEIGHTS
#define BRAVE_RT_FP16_WEIGHTS 0
#endif
constexpr bool kHalfWeightsDefault = BRAVE_RT_FP16_WEIGHTS != 0;

}  // namespace brave_rt
