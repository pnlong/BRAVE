#pragma once

#include <cmath>
#include <vector>

namespace brave_rt {

/**
 * CausalLoudnessSidechain — matches RAVE/rave/canonicalizer/silence_gate.py
 * hop=128, knee_db=-40, smooth_frames=8.
 */
class CausalLoudnessSidechain {
 public:
  explicit CausalLoudnessSidechain(int hop = 128, float knee_db = -40.f,
                                   int smooth_frames = 8)
      : hop_(hop),
        knee_(std::pow(10.f, knee_db / 20.f)),
        alpha_(1.f / static_cast<float>(smooth_frames)) {}

  void reset() { env_ = 0.f; }

  /** Scale y by input-loudness envelope of x. Lengths equal. Mono. */
  void process(const float* x, float* y, int n) {
    int i = 0;
    while (i + hop_ <= n) {
      double sumsq = 0;
      for (int t = 0; t < hop_; ++t) {
        const double v = x[i + t];
        sumsq += v * v;
      }
      const float r = static_cast<float>(std::sqrt(sumsq / hop_));
      env_ = (1.f - alpha_) * env_ + alpha_ * r;
      const float g = env_ / (env_ + knee_);
      for (int t = 0; t < hop_; ++t) y[i + t] *= g;
      i += hop_;
    }
    // Tail: use last env
    if (i < n) {
      const float g = env_ / (env_ + knee_);
      for (; i < n; ++i) y[i] *= g;
    }
  }

 private:
  int hop_;
  float knee_;
  float alpha_;
  float env_ = 0.f;
};

}  // namespace brave_rt
