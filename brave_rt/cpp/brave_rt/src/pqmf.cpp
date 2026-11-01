#include "brave_rt/pqmf.hpp"

#include "brave_rt/npy.hpp"

#include <algorithm>
#include <stdexcept>

namespace brave_rt {

CachedPqmf::CachedPqmf()
    : forward_(1, kNBand, 129, /*stride=*/kNBand),
      inverse_(kNBand, kNBand, 9, /*stride=*/1) {
  // PQMF is tiny and precision-critical: always fp32 weights.
  forward_.set_half_weights(false);
  inverse_.set_half_weights(false);
}

void CachedPqmf::load_weights_npy(const std::string& forward_npy,
                                  const std::string& inverse_npy) {
  std::vector<size_t> shp;
  auto fw = load_npy_f32(forward_npy, &shp);
  if (shp.size() != 3 || shp[0] != kNBand || shp[1] != 1 || shp[2] != 129)
    throw std::runtime_error("forward weight shape != (16,1,129)");
  forward_.set_weight(fw.data());

  auto inv = load_npy_f32(inverse_npy, &shp);
  if (shp.size() != 3 || shp[0] != kNBand || shp[1] != kNBand || shp[2] != 9)
    throw std::runtime_error("inverse weight shape != (16,16,9)");
  inverse_.set_weight(inv.data());
}

void CachedPqmf::reset() {
  forward_.reset();
  inverse_.reset();
}

void CachedPqmf::reverse_half(float* bands, int n_band, int T_b) {
  // mask[..., 1::2, ::2] = -1  → odd channels, even time indices
  for (int c = 1; c < n_band; c += 2) {
    for (int t = 0; t < T_b; t += 2) {
      bands[static_cast<size_t>(c * T_b + t)] *= -1.f;
    }
  }
}

const float* CachedPqmf::analysis_run(const float* x, int T, int& T_b) {
  float* y = forward_.run(x, T, T_b);  // (16, T_b) — T_b may be ceil, not T/16
  reverse_half(y, kNBand, T_b);
  return y;
}

std::vector<float> CachedPqmf::analysis(const float* x, int T) {
  int T_b = 0;
  const float* y = analysis_run(x, T, T_b);
  return std::vector<float>(y, y + static_cast<size_t>(kNBand * T_b));
}

const float* CachedPqmf::synthesis_run(const float* bands_in, int T_b,
                                       int& T_y) {
  const size_t n = static_cast<size_t>(kNBand) * T_b;
  ensure_size(syn_in_, n);
  std::copy_n(bands_in, n, syn_in_.data());
  reverse_half(syn_in_.data(), kNBand, T_b);
  int T_inv = 0;
  const float* y = inverse_.run(syn_in_.data(), T_b, T_inv);  // (16, T_b)
  // pqmf.py: y *= m; y.flip(1); permute(0,2,1); reshape(B,T,-1,m);
  // permute(0,2,1,3); reshape(B,1,-1) → mono[t*16 + m] = y[15-m, t] * 16.
  ensure_size(syn_mono_, n);
  for (int t = 0; t < T_inv; ++t)
    for (int m = 0; m < kNBand; ++m)
      syn_mono_[static_cast<size_t>(t * kNBand + m)] =
          y[static_cast<size_t>((kNBand - 1 - m) * T_inv + t)] *
          static_cast<float>(kNBand);
  T_y = T_inv * kNBand;
  return syn_mono_.data();
}

std::vector<float> CachedPqmf::synthesis(const float* bands_in, int T_b) {
  int T_y = 0;
  const float* y = synthesis_run(bands_in, T_b, T_y);
  return std::vector<float>(y, y + static_cast<size_t>(T_y));
}

std::vector<float> CachedPqmf::roundtrip(const float* x, int T) {
  auto bands = analysis(x, T);
  const int T_b = static_cast<int>(bands.size() / static_cast<size_t>(kNBand));
  return synthesis(bands.data(), T_b);
}

}  // namespace brave_rt
