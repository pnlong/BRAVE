#pragma once

#include <initializer_list>
#include <string>
#include <vector>

#include "brave_rt/batchnorm.hpp"
#include "brave_rt/cached_conv1d.hpp"
#include "brave_rt/npy.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/** Causal BRAVE encoder → mean of variational head (128, T_lat). */
class Encoder {
 public:
  Encoder()
      : stem_(kNBand, kCapacity, 7, 1),
        ds0_(kCapacity, kCapacity * 2, 5, 2),
        ds1_(kCapacity * 2, kCapacity * 4, 5, 2),
        ds2_(kCapacity * 4, kCapacity * 8, 5, 2),
        ds3_(kCapacity * 8, kCapacity * 16, 3, 1),
        head_(kCapacity * 16, kLatentSize * 2, 5, 1, 1, /*groups=*/2) {}

  /** Prefer calling before load (fp16→fp32 conversion is lossy). */
  void set_half_weights(bool half) {
    for (CachedConv1d* c : {&stem_, &ds0_, &ds1_, &ds2_, &ds3_, &head_})
      c->set_half_weights(half);
  }

  void set_pool(SpinPool* pool) {
    for (CachedConv1d* c : {&stem_, &ds0_, &ds1_, &ds2_, &ds3_, &head_})
      c->set_pool(pool);
  }

  /**
   * Load encoder weights. key_prefix "encoder" (vanilla) or "encoder_x" /
   * "encoder_y" (CycleGAN dual-prefix bundles).
   */
  void load_npy_dir(const std::string& dir,
                    const std::string& key_prefix = "encoder") {
    auto loadw = [&](const std::string& suffix) {
      return load_npy_f32(dir + "/" + key_prefix + "__encoder__" + suffix);
    };
    stem_.set_weight(loadw("net__0__weight.npy").data());
    bn0_.load(kCapacity, loadw("net__1__weight.npy").data(),
              loadw("net__1__bias.npy").data(),
              loadw("net__1__running_mean.npy").data(),
              loadw("net__1__running_var.npy").data());
    ds0_.set_weight(loadw("net__3__weight.npy").data());
    bn1_.load(kCapacity * 2, loadw("net__4__weight.npy").data(),
              loadw("net__4__bias.npy").data(),
              loadw("net__4__running_mean.npy").data(),
              loadw("net__4__running_var.npy").data());
    ds1_.set_weight(loadw("net__6__weight.npy").data());
    bn2_.load(kCapacity * 4, loadw("net__7__weight.npy").data(),
              loadw("net__7__bias.npy").data(),
              loadw("net__7__running_mean.npy").data(),
              loadw("net__7__running_var.npy").data());
    ds2_.set_weight(loadw("net__9__weight.npy").data());
    bn3_.load(kCapacity * 8, loadw("net__10__weight.npy").data(),
              loadw("net__10__bias.npy").data(),
              loadw("net__10__running_mean.npy").data(),
              loadw("net__10__running_var.npy").data());
    ds3_.set_weight(loadw("net__12__weight.npy").data());
    head_.set_weight(loadw("net__14__weight.npy").data());
    loaded_ = true;
  }

  void reset() {
    stem_.reset();
    ds0_.reset();
    ds1_.reset();
    ds2_.reset();
    ds3_.reset();
    head_.reset();
  }

  /**
   * Hot path. bands (16, T_b) → z_raw (256, T_lat) in an owned buffer; the
   * mean (chunk(2, dim=1)[0]) is the first 128 channel rows, i.e. the same
   * pointer. Valid until the next call.
   */
  const float* run_mean(const float* bands, int T_b, int& T_lat) {
    int T = 0;
    float* h = stem_.run(bands, T_b, T);
    bn0_.apply(h, T);
    leaky_relu(h, kCapacity * T, 0.2f);

    h = ds0_.run(h, T, T);
    bn1_.apply(h, T);
    leaky_relu(h, kCapacity * 2 * T, 0.2f);

    h = ds1_.run(h, T, T);
    bn2_.apply(h, T);
    leaky_relu(h, kCapacity * 4 * T, 0.2f);

    h = ds2_.run(h, T, T);
    bn3_.apply(h, T);
    leaky_relu(h, kCapacity * 8 * T, 0.2f);

    h = ds3_.run(h, T, T);
    leaky_relu(h, kCapacity * 16 * T, 0.2f);

    const float* z_raw = head_.run(h, T, T_lat);
    return z_raw;
  }

  /** bands (16, T_b) → z_mean (128, T_lat). Also fills z_raw if non-null (256, T_lat). */
  std::vector<float> forward_mean(const float* bands, int T_b,
                                  std::vector<float>* z_raw_out = nullptr) {
    int T = 0;
    const float* z_raw = run_mean(bands, T_b, T);
    if (z_raw_out)
      z_raw_out->assign(z_raw, z_raw + static_cast<size_t>(kLatentSize * 2 * T));
    return std::vector<float>(z_raw, z_raw + static_cast<size_t>(kLatentSize * T));
  }

  bool loaded() const { return loaded_; }

 private:
  CachedConv1d stem_, ds0_, ds1_, ds2_, ds3_, head_;
  BatchNorm1d bn0_, bn1_, bn2_, bn3_;
  bool loaded_ = false;
};

}  // namespace brave_rt
