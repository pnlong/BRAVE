#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "brave_rt/cached_conv1d.hpp"
#include "brave_rt/gemm_kernel.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/**
 * Causal CachedConvTranspose1d matching cached_conv (bias=false).
 * PyTorch weight: (in_ch, out_ch, k). padding = ratio//2 → cache len = 2*padding.
 *
 * Each kernel tap k is one packed GEMM: Z_k (T, out) = X^T (T, in) · W[:, :, k].
 * Taps are then overlap-added into the time-major accumulator at t*stride + k,
 * followed by the OLA cache.
 */
class CachedConvTranspose1d {
 public:
  CachedConvTranspose1d(int in_ch = 1, int out_ch = 1, int kernel = 4,
                        int stride = 2, int padding = 1)
      : in_ch_(in_ch),
        out_ch_(out_ch),
        kernel_(kernel),
        stride_(stride),
        padding_(padding) {
    out_pad_ = round_up_ob(out_ch_);
    cache_len_ = 2 * padding_;
    // Zero weights until set_weight() (unloaded layer outputs zeros).
    wf_.assign(static_cast<size_t>(kernel_) * out_pad_ * in_ch_, 0.f);
    half_ = false;
    set_half_weights(kHalfWeightsDefault);
    cache_.assign(static_cast<size_t>(out_ch_ * cache_len_), 0.f);
  }

  /** Weight storage precision; converts already-loaded weights (fp16→fp32 is lossy). */
  void set_half_weights(bool half) {
    if (half == half_) return;
    if (half) {
      wh_.resize(wf_.size());
      for (size_t i = 0; i < wf_.size(); ++i) wh_[i] = static_cast<_Float16>(wf_[i]);
      wf_.clear();
    } else {
      wf_.resize(wh_.size());
      for (size_t i = 0; i < wh_.size(); ++i) wf_[i] = static_cast<float>(wh_[i]);
      wh_.clear();
    }
    half_ = half;
  }

  /** Optional intra-layer thread pool (not owned). nullptr → single thread. */
  void set_pool(SpinPool* pool) { pool_ = pool; }

  /** PyTorch layout (in_ch, out_ch, k) → panels [k][ob][ic][16]. */
  void set_weight(const float* w_ich_och_k) {
    const int nob = out_pad_ / kOB;
    std::vector<float> packed(static_cast<size_t>(kernel_) * out_pad_ * in_ch_, 0.f);
    for (int k = 0; k < kernel_; ++k)
      for (int ic = 0; ic < in_ch_; ++ic)
        for (int oc = 0; oc < out_ch_; ++oc) {
          const int ob = oc / kOB, i = oc % kOB;
          packed[((static_cast<size_t>(k) * nob + ob) * in_ch_ + ic) * kOB + i] =
              w_ich_och_k[(static_cast<size_t>(ic) * out_ch_ + oc) * kernel_ + k];
        }
    if (half_) {
      wh_.resize(packed.size());
      for (size_t i = 0; i < packed.size(); ++i)
        wh_[i] = static_cast<_Float16>(packed[i]);
      wf_.clear();
    } else {
      wf_ = std::move(packed);
      wh_.clear();
    }
  }

  void reset() { std::fill(cache_.begin(), cache_.end(), 0.f); }

  int out_len(int T) const { return (T - 1) * stride_ + kernel_ - cache_len_; }

  /** x: (in_ch, T) → y: (out_ch, T_out), T_out = (T-1)*stride + kernel - cache_len. */
  int forward_into(const float* x, int T, float* y) {
    if (T < 1) throw std::runtime_error("CachedConvTranspose1d empty");
    const int T_full = (T - 1) * stride_ + kernel_;
    const int nob = out_pad_ / kOB;

    // X^T: (T, in_ch)
    ensure_size(xt_, static_cast<size_t>(T) * in_ch_);
    for (int ic = 0; ic < in_ch_; ++ic)
      for (int t = 0; t < T; ++t)
        xt_[static_cast<size_t>(t) * in_ch_ + ic] = x[static_cast<size_t>(ic) * T + t];

    ensure_size(z_, static_cast<size_t>(T) * out_pad_);
    ensure_size(acc_, static_cast<size_t>(T_full) * out_pad_);
    std::fill_n(acc_.data(), static_cast<size_t>(T_full) * out_pad_, 0.f);

    // PyTorch: out[b, oc, t*s + k] += w[ic, oc, k] * in[b, ic, t]
    // Work items: output panels; each owns its columns of z_ and acc_.
    auto panels = [&](int p0, int p1) {
      if (p0 >= p1) return;
      const int cnt = p1 - p0;
      for (int k = 0; k < kernel_; ++k) {
        const size_t woff = (static_cast<size_t>(k) * nob + p0) * in_ch_ * kOB;
        float* z = z_.data() + p0 * kOB;
        if (half_)
          gemm_packed(xt_.data(), T, in_ch_, in_ch_, wh_.data() + woff, cnt, z, out_pad_);
        else
          gemm_packed(xt_.data(), T, in_ch_, in_ch_, wf_.data() + woff, cnt, z, out_pad_);
        for (int t = 0; t < T; ++t) {
          float* __restrict a =
              &acc_[static_cast<size_t>(t * stride_ + k) * out_pad_ + p0 * kOB];
          const float* __restrict zz = &z_[static_cast<size_t>(t) * out_pad_ + p0 * kOB];
          for (int i = 0; i < cnt * kOB; ++i) a[i] += zz[i];
        }
      }
    };
    const size_t n_w = static_cast<size_t>(kernel_) * nob * in_ch_ * kOB;
    if (pool_ && pool_->size() > 1 && n_w >= kParallelMinWeights) {
      const int nt = pool_->size();
      auto task = [&](int tid) {
        int b, e;
        SpinPool::split(nob, nt, tid, b, e);
        panels(b, e);
      };
      pool_->run(task);
    } else {
      panels(0, nob);
    }

    // Overlap-add cache on the left, then save right into cache, then crop right.
    for (int t = 0; t < cache_len_; ++t)
      for (int oc = 0; oc < out_ch_; ++oc)
        acc_[static_cast<size_t>(t) * out_pad_ + oc] +=
            cache_[static_cast<size_t>(t) * out_ch_ + oc];
    for (int t = 0; t < cache_len_; ++t)
      for (int oc = 0; oc < out_ch_; ++oc)
        cache_[static_cast<size_t>(t) * out_ch_ + oc] =
            acc_[static_cast<size_t>(T_full - cache_len_ + t) * out_pad_ + oc];

    const int T_out = T_full - cache_len_;
    if (T_out < 1) throw std::runtime_error("CachedConvTranspose1d T_out < 1");
    for (int oc = 0; oc < out_ch_; ++oc)
      for (int t = 0; t < T_out; ++t)
        y[static_cast<size_t>(oc) * T_out + t] =
            acc_[static_cast<size_t>(t) * out_pad_ + oc];
    return T_out;
  }

  float* run(const float* x, int T, int& T_out) {
    ensure_size(out_, static_cast<size_t>(out_ch_) * out_len(T));
    T_out = forward_into(x, T, out_.data());
    return out_.data();
  }

  std::vector<float> forward(const float* x, int T) {
    std::vector<float> y(static_cast<size_t>(out_ch_) * out_len(T));
    forward_into(x, T, y.data());
    return y;
  }

 private:
  int in_ch_, out_ch_, kernel_, stride_, padding_, cache_len_, out_pad_;
  bool half_ = kHalfWeightsDefault;
  SpinPool* pool_ = nullptr;
  std::vector<float> wf_;
  std::vector<_Float16> wh_;
  std::vector<float> cache_;  // (cache_len, out_ch) time-major
  std::vector<float> xt_, z_, acc_, out_;
};

}  // namespace brave_rt
