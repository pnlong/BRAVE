#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "brave_rt/npy.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/**
 * LatentCanonicalizer with n_layers=2 (1×1 Conv → LReLU → 1×1 Conv).
 * init_mode "random": MLP only. "identity": gated residual z + σ(α)·h
 * (Python _warp returns z+h for identity, then gate → z + σ(α)·h).
 */
class LatentWarp2 {
 public:
  void load(const std::string& npy_dir, const std::string& file_prefix,
            const std::string& init_mode) {
    if (init_mode != "random" && init_mode != "identity")
      throw std::runtime_error("LatentWarp2: init_mode must be random|identity");
    init_mode_ = init_mode;
    const std::string p = npy_dir + "/" + file_prefix;
    conv1_w_ = load_npy_f32(p + "conv1__weight.npy", &conv1_shape_);
    conv1_b_ = load_npy_f32(p + "conv1__bias.npy");
    conv2_w_ = load_npy_f32(p + "conv2__weight.npy", &conv2_shape_);
    conv2_b_ = load_npy_f32(p + "conv2__bias.npy");
    try {
      auto a = load_npy_f32(p + "alpha.npy");
      alpha_ = a.empty() ? 0.f : a[0];
    } catch (...) {
      alpha_ = 0.f;
    }
    // Conv1d weights are (out, in, 1) → out = hidden, in = latent
    hidden_ = static_cast<int>(conv1_b_.size());
    if (hidden_ < 1 || static_cast<int>(conv2_b_.size()) != kLatentSize)
      throw std::runtime_error("LatentWarp2: unexpected bias sizes");
    if (static_cast<int>(conv1_w_.size()) != hidden_ * kLatentSize)
      throw std::runtime_error("LatentWarp2: conv1 weight size mismatch");
    if (static_cast<int>(conv2_w_.size()) != kLatentSize * hidden_)
      throw std::runtime_error("LatentWarp2: conv2 weight size mismatch");
  }

  /** z, out: channel-major (latent, T). */
  void forward(const float* z, float* out, int T = 1) const {
    std::vector<float> h(static_cast<size_t>(hidden_ * T));
    for (int t = 0; t < T; ++t) {
      for (int o = 0; o < hidden_; ++o) {
        float acc = conv1_b_[static_cast<size_t>(o)];
        for (int i = 0; i < kLatentSize; ++i)
          acc += conv1_w_[static_cast<size_t>(o * kLatentSize + i)] *
                 z[static_cast<size_t>(i * T + t)];
        h[static_cast<size_t>(o * T + t)] = acc > 0.f ? acc : 0.2f * acc;
      }
    }
    std::vector<float> mlp(static_cast<size_t>(kLatentSize * T));
    for (int t = 0; t < T; ++t) {
      for (int o = 0; o < kLatentSize; ++o) {
        float acc = conv2_b_[static_cast<size_t>(o)];
        for (int i = 0; i < hidden_; ++i)
          acc += conv2_w_[static_cast<size_t>(o * hidden_ + i)] *
                 h[static_cast<size_t>(i * T + t)];
        mlp[static_cast<size_t>(o * T + t)] = acc;
      }
    }
    if (init_mode_ == "random") {
      std::copy(mlp.begin(), mlp.end(), out);
      return;
    }
    // identity: z + sigmoid(alpha) * h  (h = MLP output)
    const float s = 1.f / (1.f + std::exp(-alpha_));
    for (size_t i = 0; i < mlp.size(); ++i)
      out[i] = z[i] + s * mlp[i];
  }

 private:
  std::string init_mode_ = "random";
  std::vector<float> conv1_w_, conv1_b_, conv2_w_, conv2_b_;
  std::vector<size_t> conv1_shape_, conv2_shape_;
  int hidden_ = kLatentSize;
  float alpha_ = 0.f;
};

}  // namespace brave_rt
