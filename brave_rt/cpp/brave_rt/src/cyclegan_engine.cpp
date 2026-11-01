#include "brave_rt/cyclegan_engine.hpp"

#include "brave_rt/denormals.hpp"
#include "brave_rt/json_util.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace brave_rt {

void CycleGanEngine::load(const std::string& bundle_dir) {
  Precision p;
  p.encoder_half = kHalfWeightsDefault;
  p.decoder_half = kHalfWeightsDefault;
  load(bundle_dir, p);
}

void CycleGanEngine::load(const std::string& bundle_dir, Precision precision) {
  const std::string manifest_path = bundle_dir + "/brave_rt.json";
  const std::string json = read_file_text(manifest_path);
  model_type_ = json_string_field(json, "model_type", "cyclegan_xy");
  init_mode_ = json_string_field(json, "init_mode", "random");
  display_name_ = json_string_field(json, "display_name", "");

  if (model_type_ == "cyclegan_yx") {
    enc_prefix_ = "encoder_y";
    dec_prefix_ = "decoder_x";
    warp_file_prefix_ = "warp_yx__";
    pqmf_in_prefix_ = "pqmf_y";
    pqmf_out_prefix_ = "pqmf_x";
  } else if (model_type_ == "cyclegan_xy") {
    enc_prefix_ = "encoder_x";
    dec_prefix_ = "decoder_y";
    warp_file_prefix_ = "warp_xy__";
    pqmf_in_prefix_ = "pqmf_x";
    pqmf_out_prefix_ = "pqmf_y";
  } else {
    throw std::runtime_error(
        "CycleGanEngine: model_type must be cyclegan_xy or cyclegan_yx, got " +
        model_type_);
  }

  // Nested sidechain.default_enabled if present
  const bool sc_default = json_bool_field(json, "default_enabled", true);
  sidechain_on_ = sc_default;

  encoder_.set_half_weights(precision.encoder_half);
  decoder_.set_half_weights(precision.decoder_half);

  const std::string npy = bundle_dir + "/npy_weights";
  auto load_pqmf = [&](CachedPqmf& pqmf, const std::string& prefix) {
    try {
      pqmf.load_weights_npy(npy + "/" + prefix + "__forward_conv__weight.npy",
                            npy + "/" + prefix + "__inverse_conv__weight.npy");
    } catch (...) {
      // Flat convenience names at bundle root (rare for CycleGAN)
      pqmf.load_weights_npy(bundle_dir + "/" + prefix + "_forward_conv_weight.npy",
                            bundle_dir + "/" + prefix + "_inverse_conv_weight.npy");
    }
  };
  load_pqmf(pqmf_x_, pqmf_in_prefix_);
  load_pqmf(pqmf_y_, pqmf_out_prefix_);

  encoder_.load_npy_dir(npy, enc_prefix_);
  decoder_.load_npy_dir(npy, dec_prefix_);
  warp_.load(npy, warp_file_prefix_, init_mode_);

  z_scratch_.assign(static_cast<size_t>(kLatentSize * 64), 0.f);

  host_controls_ = load_host_controls_from_bundle(bundle_dir);
  const int D = decoder_.num_attr_channels();
  reconcile_host_controls(host_controls_, D);
  attr_raw_.assign(static_cast<size_t>(D), 0.f);
  for (int i = 0; i < D; ++i)
    attr_raw_[static_cast<size_t>(i)] =
        attr_default_raw(host_controls_.attributes[static_cast<size_t>(i)]);
  z_cond_.assign(static_cast<size_t>((kLatentSize + D) * 64), 0.f);
  reset_state();
  loaded_ = true;
}

int CycleGanEngine::attr_index(const std::string& name) const {
  for (size_t i = 0; i < host_controls_.attributes.size(); ++i)
    if (host_controls_.attributes[i].name == name) return static_cast<int>(i);
  return -1;
}

void CycleGanEngine::set_num_threads(int n) {
  encoder_.set_pool(nullptr);
  decoder_.set_pool(nullptr);
  pool_.reset();
  if (n > 1) {
    pool_ = std::make_unique<SpinPool>(n);
    encoder_.set_pool(pool_.get());
    decoder_.set_pool(pool_.get());
  }
}

int CycleGanEngine::min_fifo_latency(int host_block) {
  if (host_block <= 0) return kTickSamples - 1;
  return kTickSamples - std::gcd(host_block, kTickSamples);
}

void CycleGanEngine::prepare(int host_block, bool variable_block_sizes) {
  prepared_prime_ =
      variable_block_sizes ? kTickSamples - 1 : min_fifo_latency(host_block);
  prime_ = prepared_prime_;
}

void CycleGanEngine::reset_state() {
  pqmf_x_.reset();
  pqmf_y_.reset();
  encoder_.reset();
  decoder_.reset();
  sidechain_.reset();
  fifo_in_.clear();
  fifo_out_.clear();
  fifo_in_.reserve(kMaxBlock + kTickSamples);
  fifo_out_.reserve(2 * kMaxBlock + 2 * kTickSamples);
  prime_ = prepared_prime_;
  primed_ = false;
  underflows_ = 0;
}

void CycleGanEngine::run_ticks(const float* in, int n_samples, float* out) {
  ScopedNoDenormals no_denormals;
  int T_b = 0, T_lat = 0, T_mb = 0, T_y = 0;
  const float* bands = pqmf_x_.analysis_run(in, n_samples, T_b);
  const float* z_mean = encoder_.run_mean(bands, T_b, T_lat);
  if (static_cast<int>(z_scratch_.size()) < kLatentSize * T_lat)
    z_scratch_.assign(static_cast<size_t>(kLatentSize * T_lat), 0.f);
  warp_.forward(z_mean, z_scratch_.data(), T_lat);
  const float* z_dec = z_scratch_.data();
  const int D = decoder_.num_attr_channels();
  if (D > 0) {
    ensure_size(z_cond_, static_cast<size_t>(kLatentSize + D) * T_lat);
    std::copy_n(z_scratch_.data(), static_cast<size_t>(kLatentSize) * T_lat,
                z_cond_.data());
    normalize_broadcast(host_controls_, attr_raw_.data(), D, T_lat,
                        z_cond_.data() + static_cast<size_t>(kLatentSize) * T_lat);
    z_dec = z_cond_.data();
  }
  const float* mb = decoder_.run(z_dec, T_lat, T_mb);
  const float* y = pqmf_y_.synthesis_run(mb, T_mb, T_y);
  const int n = std::min(n_samples, T_y);
  std::copy_n(y, n, out);
  std::fill(out + n, out + n_samples, 0.f);
  if (sidechain_on_) sidechain_.process(in, out, n_samples);
}

void CycleGanEngine::process(const float* in, float* out, int n) {
  if (!loaded_) throw std::runtime_error("CycleGanEngine not loaded");
  if (!primed_) {
    if (prime_ < 0) prime_ = min_fifo_latency(n);
    fifo_out_.assign(static_cast<size_t>(prime_), 0.f);
    primed_ = true;
  }
  fifo_in_.insert(fifo_in_.end(), in, in + n);
  const int n_ticks = static_cast<int>(fifo_in_.size()) / kTickSamples;
  if (n_ticks > 0) {
    const int ns = n_ticks * kTickSamples;
    const size_t old = fifo_out_.size();
    fifo_out_.resize(old + static_cast<size_t>(ns));
    run_ticks(fifo_in_.data(), ns, fifo_out_.data() + old);
    fifo_in_.erase(fifo_in_.begin(), fifo_in_.begin() + ns);
  }
  const int take = std::min(n, static_cast<int>(fifo_out_.size()));
  std::copy_n(fifo_out_.data(), take, out);
  if (take) fifo_out_.erase(fifo_out_.begin(), fifo_out_.begin() + take);
  if (take < n) {
    ++underflows_;
    std::fill(out + take, out + n, 0.f);
  }
}

}  // namespace brave_rt
