#pragma once

#include <string>
#include <vector>

#include "brave_rt/cached_conv1d.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/**
 * Streaming CachedPQMF matching RAVE/rave/pqmf.py (n_band=16, atten=40).
 * Analysis: Conv1d in=1 out=16 k=129 stride=16, causal pad_left=128.
 * Synthesis: reverse_half → Conv1d in=16 out=16 k=9 stride=1 pad_left=8 → *16 → flip/reshape.
 */
class CachedPqmf {
 public:
  CachedPqmf();

  void load_weights_npy(const std::string& forward_npy,
                        const std::string& inverse_npy);
  void reset();

  /** Mono in (T,) → bands (16, T/16). T must be multiple of 16 for clean frames. */
  std::vector<float> analysis(const float* x, int T);

  /** bands (16, T_b) → mono (T_b * 16). */
  std::vector<float> synthesis(const float* bands, int T_b);

  /** Hot path: owned output buffers, valid until the next call of the same kind. */
  const float* analysis_run(const float* x, int T, int& T_b);
  const float* synthesis_run(const float* bands, int T_b, int& T_y);

  /** Round-trip streaming helper on mono buffer. */
  std::vector<float> roundtrip(const float* x, int T);

 private:
  CachedConv1d forward_;  // 1→16
  CachedConv1d inverse_;  // 16→16
  std::vector<float> syn_in_, syn_mono_;
  static void reverse_half(float* bands, int n_band, int T_b);
};

}  // namespace brave_rt
