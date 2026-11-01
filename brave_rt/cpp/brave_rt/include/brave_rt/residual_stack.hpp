#pragma once

#include <string>
#include <vector>

#include "brave_rt/batchnorm.hpp"
#include "brave_rt/cached_conv1d.hpp"
#include "brave_rt/npy.hpp"

namespace brave_rt {

/**
 * BRAVE ResidualStack with causal mode (AlignBranches delays all 0).
 * dilations_list = [[3,1],[9,1],[27,1],[36,1]], kernel=3.
 * Residual path: x + (LReLU→Conv dil_a→LReLU→Conv dil_b)(x)
 */
class ResidualStack {
 public:
  explicit ResidualStack(int dim) : dim_(dim) {
    const int dils[4][2] = {{3, 1}, {9, 1}, {27, 1}, {36, 1}};
    for (int i = 0; i < 4; ++i) {
      layers_[i][0] = CachedConv1d(dim, dim, 3, 1, dils[i][0], 1, false);
      layers_[i][1] = CachedConv1d(dim, dim, 3, 1, dils[i][1], 1, false);
    }
  }

  /**
   * Load from npy_weights dir. Keys like:
   * {key_prefix}__net__{net_idx}__net__branches__0__net__{layer}__net__aligned__branches__0__{1|3}__weight.npy
   * where key_prefix is "decoder" (vanilla) or "decoder_y" / "decoder_x" (CycleGAN).
   * branch 1 = first conv, 3 = second conv in ResidualLayer sequential.
   */
  void load(const std::string& npy_dir, int net_idx,
            const std::string& key_prefix = "decoder") {
    for (int layer = 0; layer < 4; ++layer) {
      for (int c = 0; c < 2; ++c) {
        const int branch_idx = (c == 0) ? 1 : 3;
        const std::string path =
            npy_dir + "/" + key_prefix + "__net__" + std::to_string(net_idx) +
            "__net__branches__0__net__" + std::to_string(layer) +
            "__net__aligned__branches__0__" + std::to_string(branch_idx) +
            "__weight.npy";
        auto w = load_npy_f32(path);
        layers_[layer][c].set_weight(w.data());
      }
    }
  }

  /** Prefer calling before load (fp16→fp32 conversion is lossy). */
  void set_half_weights(bool half) {
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 2; ++c) layers_[i][c].set_half_weights(half);
  }

  void set_pool(SpinPool* pool) {
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 2; ++c) layers_[i][c].set_pool(pool);
  }

  void reset() {
    for (int i = 0; i < 4; ++i)
      for (int c = 0; c < 2; ++c) layers_[i][c].reset();
  }

  /** x: (dim, T) → owned scratch (dim, T); valid until next call, caller may mutate. */
  float* run(const float* x, int T) {
    const size_t n = static_cast<size_t>(dim_) * T;
    ensure_size(cur_, n);
    ensure_size(h_, n);
    std::copy_n(x, n, cur_.data());
    int T1 = T;
    for (int i = 0; i < 4; ++i) {
      std::copy_n(cur_.data(), n, h_.data());
      leaky_relu(h_.data(), static_cast<int>(n), 0.2f);
      float* h = layers_[i][0].run(h_.data(), T, T1);
      leaky_relu(h, static_cast<int>(n), 0.2f);
      const float* r = layers_[i][1].run(h, T, T1);
      for (size_t j = 0; j < n; ++j) cur_[j] += r[j];
    }
    return cur_.data();
  }

  std::vector<float> forward(const float* x, int T) {
    const float* y = run(x, T);
    return std::vector<float>(y, y + static_cast<size_t>(dim_) * T);
  }

 private:
  int dim_;
  CachedConv1d layers_[4][2];
  std::vector<float> cur_, h_;
};

}  // namespace brave_rt
