#pragma once

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "brave_rt/batchnorm.hpp"
#include "brave_rt/cached_conv1d.hpp"
#include "brave_rt/conv_transpose_cached.hpp"
#include "brave_rt/npy.hpp"
#include "brave_rt/residual_stack.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

inline float mod_sigmoid(float x) {
  const float s = 1.f / (1.f + std::exp(-x));
  return 2.f * std::pow(s, 2.3f) + 1e-7f;
}

/** Causal BRAVE Generator (use_noise=False). z_mean (128,T) → multiband (16, T_mb). */
class Decoder {
 public:
  Decoder()
      : project_(kLatentSize, kCapacity * 16, 7, 1),
        up0_(kCapacity * 16, kCapacity * 8, 4, 2, 1),
        res0_(kCapacity * 8),
        up1_(kCapacity * 8, kCapacity * 4, 4, 2, 1),
        res1_(kCapacity * 4),
        up2_(kCapacity * 4, kCapacity * 2, 4, 2, 1),
        res2_(kCapacity * 2),
        up3_(kCapacity * 2, kCapacity, 3, 1),
        res3_(kCapacity),
        wave_(kCapacity, kNBand, 7, 1),
        loud_(kCapacity, 1, 3, 1) {}

  /** Prefer calling before load (fp16→fp32 conversion is lossy). */
  void set_half_weights(bool half) {
    for (CachedConv1d* c : {&project_, &up3_, &wave_, &loud_})
      c->set_half_weights(half);
    for (CachedConvTranspose1d* c : {&up0_, &up1_, &up2_})
      c->set_half_weights(half);
    for (ResidualStack* r : {&res0_, &res1_, &res2_, &res3_})
      r->set_half_weights(half);
  }

  void set_pool(SpinPool* pool) {
    for (CachedConv1d* c : {&project_, &up3_, &wave_, &loud_}) c->set_pool(pool);
    for (CachedConvTranspose1d* c : {&up0_, &up1_, &up2_}) c->set_pool(pool);
    for (ResidualStack* r : {&res0_, &res1_, &res2_, &res3_}) r->set_pool(pool);
  }

  /**
   * Load decoder weights. key_prefix "decoder" (vanilla) or "decoder_y" /
   * "decoder_x" (CycleGAN).
   */
  void load_npy_dir(const std::string& dir,
                    const std::string& key_prefix = "decoder") {
    auto loadw = [&](const std::string& suffix) {
      return load_npy_f32(dir + "/" + key_prefix + "__" + suffix);
    };
    {
      // Fader: project weight is (512, 128+D, 7); vanilla is (512, 128, 7).
      std::vector<size_t> shape;
      std::vector<float> w =
          load_npy_f32(dir + "/" + key_prefix + "__net__0__weight.npy", &shape);
      if (shape.size() != 3)
        throw std::runtime_error("decoder project weight must be (out,in,k)");
      const int out = static_cast<int>(shape[0]);
      const int in = static_cast<int>(shape[1]);
      const int k = static_cast<int>(shape[2]);
      if (in < kLatentSize)
        throw std::runtime_error("decoder project in_ch < 128");
      if (in != project_.in_ch()) project_.reconfigure(in, out, k, 1);
      project_.set_weight(w.data());
      latent_in_ = in;
    }
    up0_.set_weight(loadw("net__1__net__1__weight.npy").data());
    res0_.load(dir, /*net_idx=*/2, key_prefix);
    up1_.set_weight(loadw("net__3__net__1__weight.npy").data());
    res1_.load(dir, /*net_idx=*/4, key_prefix);
    up2_.set_weight(loadw("net__5__net__1__weight.npy").data());
    res2_.load(dir, /*net_idx=*/6, key_prefix);
    up3_.set_weight(loadw("net__7__net__1__weight.npy").data());
    res3_.load(dir, /*net_idx=*/8, key_prefix);
    wave_.set_weight(loadw("synth__branches__0__weight.npy").data());
    loud_.set_weight(loadw("synth__branches__1__weight.npy").data());
    loaded_ = true;
  }

  void reset() {
    project_.reset();
    up0_.reset();
    res0_.reset();
    up1_.reset();
    res1_.reset();
    up2_.reset();
    res2_.reset();
    up3_.reset();
    res3_.reset();
    wave_.reset();
    loud_.reset();
  }

  /** Hot path. z (latent_in(), T_lat; caller concats attrs rows 128..) → multiband (16, T_mb) in an owned buffer. */
  const float* run(const float* z, int T_lat, int& T_mb) {
    int T = 0;
    float* h = project_.run(z, T_lat, T);

    leaky_relu(h, kCapacity * 16 * T, 0.2f);
    h = up0_.run(h, T, T);
    h = res0_.run(h, T);

    leaky_relu(h, kCapacity * 8 * T, 0.2f);
    h = up1_.run(h, T, T);
    h = res1_.run(h, T);

    leaky_relu(h, kCapacity * 4 * T, 0.2f);
    h = up2_.run(h, T, T);
    h = res2_.run(h, T);

    leaky_relu(h, kCapacity * 2 * T, 0.2f);
    h = up3_.run(h, T, T);
    const float* r = res3_.run(h, T);

    int Tw = 0;
    const float* wave = wave_.run(r, T, Tw);
    const float* loud = loud_.run(r, T, Tw);
    // tanh(wave) * mod_sigmoid(loud); loud is (1, T)
    ensure_size(mb_, static_cast<size_t>(kNBand) * T);
    for (int t = 0; t < T; ++t) {
      const float g = mod_sigmoid(loud[t]);
      for (int c = 0; c < kNBand; ++c)
        mb_[static_cast<size_t>(c * T + t)] =
            std::tanh(wave[static_cast<size_t>(c * T + t)]) * g;
    }
    T_mb = T;
    return mb_.data();
  }

  /** z (latent_in(), T_lat) → multiband (16, T_mb). */
  std::vector<float> forward(const float* z, int T_lat) {
    int T = 0;
    const float* mb = run(z, T_lat, T);
    return std::vector<float>(mb, mb + static_cast<size_t>(kNBand * T));
  }

  bool loaded() const { return loaded_; }

  /** Decoder input channels: 128 (vanilla) or 128+D (Fader). */
  int latent_in() const { return latent_in_; }
  /** D = number of concatenated (normalized) attribute channels. */
  int num_attr_channels() const {
    return latent_in_ > kLatentSize ? latent_in_ - kLatentSize : 0;
  }

 private:
  int latent_in_ = kLatentSize;
  CachedConv1d project_;
  CachedConvTranspose1d up0_;
  ResidualStack res0_;
  CachedConvTranspose1d up1_;
  ResidualStack res1_;
  CachedConvTranspose1d up2_;
  ResidualStack res2_;
  CachedConv1d up3_;
  ResidualStack res3_;
  CachedConv1d wave_, loud_;
  std::vector<float> mb_;
  bool loaded_ = false;
};

}  // namespace brave_rt
