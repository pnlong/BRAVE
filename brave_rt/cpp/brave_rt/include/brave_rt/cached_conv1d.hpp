#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "brave_rt/gemm_kernel.hpp"
#include "brave_rt/spin_pool.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/** Layers with fewer weights than this run on the caller thread only. */
#ifndef BRAVE_RT_PAR_MIN_WEIGHTS
#define BRAVE_RT_PAR_MIN_WEIGHTS (32 * 1024)
#endif
constexpr size_t kParallelMinWeights = BRAVE_RT_PAR_MIN_WEIGHTS;

/** Grow-only resize: no reallocation once the steady-state size is reached. */
inline void ensure_size(std::vector<float>& v, size_t n) {
  if (v.size() < n) v.resize(n);
}

/**
 * Causal CachedConv1d matching cached_conv.CachedConv1d (v2.5).
 * Supports dilation + groups. Bias optional (AE layers: false).
 * Causal: all pad on left = (k-1)*dilation; r_pad=0 → stride_delay=0 when cd=0.
 *
 * Hot path: per group, im2col (T_out, ipg*k) then the packed GEMM kernel
 * (gemm_kernel.hpp) with weights in 16-output-channel panels (fp32 or fp16).
 * Scratch/output buffers are owned and grow-only → no per-tick allocation.
 */
class CachedConv1d {
 public:
  CachedConv1d(int in_ch = 1, int out_ch = 1, int kernel = 1, int stride = 1,
               int dilation = 1, int groups = 1, bool bias = false)
      : in_ch_(in_ch),
        out_ch_(out_ch),
        kernel_(kernel),
        stride_(stride),
        dilation_(dilation),
        groups_(groups),
        bias_(bias) {
    if (in_ch_ % groups_ != 0 || out_ch_ % groups_ != 0)
      throw std::runtime_error("CachedConv1d groups must divide channels");
    pad_left_ = (kernel_ - 1) * dilation_;
    ipg_ = in_ch_ / groups_;
    opg_ = out_ch_ / groups_;
    opg_pad_ = round_up_ob(opg_);
    J_ = ipg_ * kernel_;
    if (bias_) bias_v_.assign(static_cast<size_t>(out_ch_), 0.f);
    // Zero weights until set_weight() (unloaded layer outputs zeros).
    wf_.assign(static_cast<size_t>(groups_) * opg_pad_ * J_, 0.f);
    half_ = false;
    set_half_weights(kHalfWeightsDefault);
    pad_.assign(static_cast<size_t>(in_ch_ * pad_left_), 0.f);
  }

  /**
   * Rebuild geometry in place (e.g. Fader decoder project: in_ch 128 → 128+D).
   * Weights are zeroed (call set_weight afterwards); half_/pool_ are preserved.
   * Not realtime-safe (allocates) — load-time only.
   */
  void reconfigure(int in_ch, int out_ch, int kernel = 7, int stride = 1,
                   int dilation = 1, int groups = 1, bool bias = false) {
    if (groups < 1 || in_ch % groups != 0 || out_ch % groups != 0)
      throw std::runtime_error("CachedConv1d groups must divide channels");
    in_ch_ = in_ch;
    out_ch_ = out_ch;
    kernel_ = kernel;
    stride_ = stride;
    dilation_ = dilation;
    groups_ = groups;
    bias_ = bias;
    pad_left_ = (kernel_ - 1) * dilation_;
    ipg_ = in_ch_ / groups_;
    opg_ = out_ch_ / groups_;
    opg_pad_ = round_up_ob(opg_);
    J_ = ipg_ * kernel_;
    bias_v_.clear();
    if (bias_) bias_v_.assign(static_cast<size_t>(out_ch_), 0.f);
    const size_t n = static_cast<size_t>(groups_) * opg_pad_ * J_;
    if (half_) {
      wh_.assign(n, static_cast<_Float16>(0.f));
      wf_.clear();
    } else {
      wf_.assign(n, 0.f);
      wh_.clear();
    }
    pad_.assign(static_cast<size_t>(in_ch_ * pad_left_), 0.f);
  }

  /** Weight storage precision; converts already-loaded weights (fp16→fp32 is lossy). PQMF keeps fp32. */
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

  /** PyTorch layout (out_ch, in_ch/groups, k) → panels [g][ob][j=ig*k+kk][16]. */
  void set_weight(const float* w_och_ichg_k) {
    const size_t n = static_cast<size_t>(groups_) * opg_pad_ * J_;
    std::vector<float> packed(n, 0.f);
    for (int g = 0; g < groups_; ++g)
      for (int og = 0; og < opg_; ++og)
        for (int j = 0; j < J_; ++j) {
          const int oc = g * opg_ + og;
          const int ob = og / kOB, i = og % kOB;
          packed[((static_cast<size_t>(g) * (opg_pad_ / kOB) + ob) * J_ + j) *
                     kOB + i] = w_och_ichg_k[static_cast<size_t>(oc) * J_ + j];
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

  void set_bias(const float* b) {
    if (!bias_) return;
    bias_v_.assign(b, b + out_ch_);
  }

  void reset() { std::fill(pad_.begin(), pad_.end(), 0.f); }

  int out_len(int T) const {
    const int eff_k = (kernel_ - 1) * dilation_ + 1;
    return (pad_left_ + T - eff_k) / stride_ + 1;
  }

  /** x: (in_ch, T) → y: (out_ch, T_out). Returns T_out. y must not alias x. */
  int forward_into(const float* x, int T, float* y) {
    if (T < 1) throw std::runtime_error("CachedConv1d empty input");
    const int T_cat = pad_left_ + T;
    ensure_size(cat_, static_cast<size_t>(in_ch_) * T_cat);
    for (int c = 0; c < in_ch_; ++c) {
      float* row = &cat_[static_cast<size_t>(c) * T_cat];
      std::copy_n(&pad_[static_cast<size_t>(c) * pad_left_], pad_left_, row);
      std::copy_n(&x[static_cast<size_t>(c) * T], T, row + pad_left_);
      std::copy_n(row + T, pad_left_, &pad_[static_cast<size_t>(c) * pad_left_]);
    }

    const int T_out = out_len(T);
    if (T_out < 1) throw std::runtime_error("CachedConv1d T_out < 1");

    // col_: (groups, T_out, J). acc_: (T_out, ldo) time-major; group g
    // occupies columns [g*opg_pad, +opg).
    const int ldo = groups_ * opg_pad_;
    const int nob = opg_pad_ / kOB;
    ensure_size(col_, static_cast<size_t>(groups_) * T_out * J_);
    ensure_size(acc_, static_cast<size_t>(T_out) * ldo);
    for (int g = 0; g < groups_; ++g) {
      for (int t = 0; t < T_out; ++t) {
        float* c = &col_[(static_cast<size_t>(g) * T_out + t) * J_];
        for (int ig = 0; ig < ipg_; ++ig) {
          const float* row =
              &cat_[static_cast<size_t>(g * ipg_ + ig) * T_cat + t * stride_];
          for (int k = 0; k < kernel_; ++k) c[ig * kernel_ + k] = row[k * dilation_];
        }
      }
    }

    // Work items: (group, panel) pairs; contiguous runs within a group are
    // issued as one gemm call.
    auto panels = [&](int p0, int p1) {
      while (p0 < p1) {
        const int g = p0 / nob, ob = p0 % nob;
        const int cnt = std::min(p1 - p0, nob - ob);
        const float* c = col_.data() + static_cast<size_t>(g) * T_out * J_;
        const size_t woff = (static_cast<size_t>(g) * nob + ob) * J_ * kOB;
        float* out = acc_.data() + g * opg_pad_ + ob * kOB;
        if (half_)
          gemm_packed(c, T_out, J_, J_, wh_.data() + woff, cnt, out, ldo);
        else
          gemm_packed(c, T_out, J_, J_, wf_.data() + woff, cnt, out, ldo);
        p0 += cnt;
      }
    };
    const int n_items = groups_ * nob;
    const size_t n_w = static_cast<size_t>(n_items) * J_ * kOB;
    if (pool_ && pool_->size() > 1 && n_w >= kParallelMinWeights) {
      const int nt = pool_->size();
      auto task = [&](int tid) {
        int b, e;
        SpinPool::split(n_items, nt, tid, b, e);
        panels(b, e);
      };
      pool_->run(task);
    } else {
      panels(0, n_items);
    }

    for (int g = 0; g < groups_; ++g)
      for (int og = 0; og < opg_; ++og) {
        const int oc = g * opg_ + og;
        const float b = bias_ ? bias_v_[static_cast<size_t>(oc)] : 0.f;
        for (int t = 0; t < T_out; ++t)
          y[static_cast<size_t>(oc) * T_out + t] =
              acc_[static_cast<size_t>(t) * ldo + g * opg_pad_ + og] + b;
      }
    return T_out;
  }

  /** Hot path: writes into an owned buffer; valid until next call, caller may mutate. */
  float* run(const float* x, int T, int& T_out) {
    ensure_size(out_, static_cast<size_t>(out_ch_) * out_len(T));
    T_out = forward_into(x, T, out_.data());
    return out_.data();
  }

  /** Convenience (allocates) — tools / tests only. */
  std::vector<float> forward(const float* x, int T) {
    std::vector<float> y(static_cast<size_t>(out_ch_) * out_len(T));
    forward_into(x, T, y.data());
    return y;
  }

  int in_ch() const { return in_ch_; }
  int out_ch() const { return out_ch_; }
  int stride() const { return stride_; }
  int pad_left() const { return pad_left_; }

 private:
  int in_ch_, out_ch_, kernel_, stride_, dilation_, groups_, pad_left_;
  int ipg_, opg_, opg_pad_, J_;
  bool bias_;
  bool half_ = kHalfWeightsDefault;
  SpinPool* pool_ = nullptr;
  std::vector<float> wf_;
  std::vector<_Float16> wh_;
  std::vector<float> bias_v_, pad_;
  std::vector<float> cat_, col_, acc_, out_;
};

}  // namespace brave_rt
