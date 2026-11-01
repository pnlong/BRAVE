#pragma once

#include <cmath>
#include <vector>

namespace brave_rt {

/** Eval-mode BatchNorm1d, folded to per-channel scale/shift at load. */
struct BatchNorm1d {
  int C = 0;
  float eps = 1e-5f;
  std::vector<float> scale, shift;

  void load(int c, const float* w, const float* b, const float* mean,
            const float* var, float eps_ = 1e-5f) {
    C = c;
    eps = eps_;
    scale.resize(static_cast<size_t>(c));
    shift.resize(static_cast<size_t>(c));
    for (int i = 0; i < c; ++i) {
      scale[i] = w[i] / std::sqrt(var[i] + eps);
      shift[i] = b[i] - mean[i] * scale[i];
    }
  }

  /** x: (C, T) in-place. */
  void apply(float* x, int T) const {
    for (int c = 0; c < C; ++c) {
      const float s = scale[c], h = shift[c];
      float* row = x + static_cast<size_t>(c) * T;
      for (int t = 0; t < T; ++t) row[t] = row[t] * s + h;
    }
  }
};

inline void leaky_relu(float* x, int n, float neg = 0.2f) {
  for (int i = 0; i < n; ++i) x[i] = x[i] < 0.f ? x[i] * neg : x[i];
}

}  // namespace brave_rt
