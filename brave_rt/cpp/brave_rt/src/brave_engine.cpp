#include "brave_rt/brave_engine.hpp"

#include "brave_rt/denormals.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace brave_rt {

void BraveEngine::load(const std::string& bundle_dir) {
  Precision p;
  p.encoder_half = kHalfWeightsDefault;
  p.decoder_half = kHalfWeightsDefault;
  load(bundle_dir, p);
}

void BraveEngine::load(const std::string& bundle_dir, Precision precision) {
  encoder_.set_half_weights(precision.encoder_half);
  decoder_.set_half_weights(precision.decoder_half);
  const std::string npy = bundle_dir + "/npy_weights";
  // Prefer dedicated pqmf copies at bundle root (Stage 0 layout), else npy_weights.
  try {
    pqmf_.load_weights_npy(bundle_dir + "/pqmf_forward_conv_weight.npy",
                           bundle_dir + "/pqmf_inverse_conv_weight.npy");
  } catch (...) {
    pqmf_.load_weights_npy(npy + "/pqmf__forward_conv__weight.npy",
                           npy + "/pqmf__inverse_conv__weight.npy");
  }
  encoder_.load_npy_dir(npy);
  decoder_.load_npy_dir(npy);

  // Stage 7: schema from brave_rt.json / host_controls.json; D from weight shape.
  host_controls_ = load_host_controls_from_bundle(bundle_dir);
  const int D = decoder_.num_attr_channels();
  reconcile_host_controls(host_controls_, D);
  attr_raw_.assign(static_cast<size_t>(D), 0.f);
  for (int i = 0; i < D; ++i)
    attr_raw_[static_cast<size_t>(i)] =
        attr_default_raw(host_controls_.attributes[static_cast<size_t>(i)]);
  attr_eff_ = attr_raw_;
  attr_override_.assign(static_cast<size_t>(D), 0);
  attr_kind_.assign(static_cast<size_t>(D), -1);
  for (int i = 0; i < D; ++i) {
    const auto& a = host_controls_.attributes[static_cast<size_t>(i)];
    if (a.kind == AttrKind::Continuous)
      attr_kind_[static_cast<size_t>(i)] = DescriptorExtract::kind_from_name(a.name);
  }
  z_cond_.assign(static_cast<size_t>((kLatentSize + D) * 64), 0.f);
  reset_state();
  loaded_ = true;
}

void BraveEngine::set_num_threads(int n) {
  encoder_.set_pool(nullptr);
  decoder_.set_pool(nullptr);
  pool_.reset();
  if (n > 1) {
    pool_ = std::make_unique<SpinPool>(n);
    encoder_.set_pool(pool_.get());
    decoder_.set_pool(pool_.get());
  }
}

int BraveEngine::min_fifo_latency(int host_block) {
  if (host_block <= 0) return kTickSamples - 1;
  return kTickSamples - std::gcd(host_block, kTickSamples);
}

void BraveEngine::prepare(int host_block, bool variable_block_sizes) {
  prepared_prime_ =
      variable_block_sizes ? kTickSamples - 1 : min_fifo_latency(host_block);
  prime_ = prepared_prime_;
}

void BraveEngine::reset_state() {
  pqmf_.reset();
  encoder_.reset();
  decoder_.reset();
  fifo_in_.clear();
  fifo_out_.clear();
  fifo_in_.reserve(kMaxBlock + kTickSamples);
  fifo_out_.reserve(2 * kMaxBlock + 2 * kTickSamples);
  prime_ = prepared_prime_;
  primed_ = false;
  underflows_ = 0;
  extract_.reset();
  last_attr_mode_ = attr_mode();
  attr_eff_ = attr_raw_;
}

std::vector<float> BraveEngine::tick_pqmf_bands(const float* in128) {
  return pqmf_.analysis(in128, kTickSamples);
}

std::vector<float> BraveEngine::tick_z_mean(const float* bands16x8) {
  return encoder_.forward_mean(bands16x8, 8);
}

std::vector<float> BraveEngine::tick_decoder_mb(const float* z128) {
  return decoder_.forward(condition_latent(z128, 1), 1);
}

int BraveEngine::attr_index(const std::string& name) const {
  for (size_t i = 0; i < host_controls_.attributes.size(); ++i)
    if (host_controls_.attributes[i].name == name) return static_cast<int>(i);
  return -1;
}

const float* BraveEngine::condition_latent(const float* z, int T_lat) {
  const int D = decoder_.num_attr_channels();
  if (D == 0) return z;  // vanilla: no copy, no allocation
  ensure_size(z_cond_, static_cast<size_t>(kLatentSize + D) * T_lat);
  std::copy_n(z, static_cast<size_t>(kLatentSize) * T_lat, z_cond_.data());
  // Mode 1 reads the manual array directly (Stage 7 path, unchanged).
  const float* raw = attr_mode() == 1 ? attr_raw_.data() : attr_eff_.data();
  normalize_broadcast(host_controls_, raw, D, T_lat,
                      z_cond_.data() + static_cast<size_t>(kLatentSize) * T_lat);
  return z_cond_.data();
}

std::vector<float> BraveEngine::tick_e2e(const float* in128) {
  std::vector<float> y(kTickSamples);
  run_ticks(in128, kTickSamples, y.data());
  return y;
}

void BraveEngine::update_extracted_attrs(const float* in, int n_samples, int mode) {
  if (mode != last_attr_mode_) {  // entered extract from manual: restart history
    extract_.reset();
    last_attr_mode_ = mode;
  }
  extract_.push(in, n_samples);
  const int D = static_cast<int>(attr_raw_.size());
  const bool ready = extract_.ready();
  for (int i = 0; i < D; ++i) {
    const size_t u = static_cast<size_t>(i);
    float v = attr_raw_[u];  // discrete / non-extractable / not ready / override
    const int kind = attr_kind_[u];
    if (ready && kind >= 0 && !(mode == 0 && attr_override_[u] != 0)) {
      const auto& a = host_controls_.attributes[u];
      v = std::min(std::max(extract_.value(kind), a.min_v), a.max_v);  // nn~ _clamp_raw
    }
    attr_eff_[u] = v;
  }
}

void BraveEngine::run_ticks(const float* in, int n_samples, float* out) {
  ScopedNoDenormals no_denormals;
  if (!attr_raw_.empty()) {
    const int mode = attr_mode();
    if (mode != 1) update_extracted_attrs(in, n_samples, mode);
    else last_attr_mode_ = 1;
  }
  int T_b = 0, T_lat = 0, T_mb = 0, T_y = 0;
  const float* bands = pqmf_.analysis_run(in, n_samples, T_b);
  const float* z = encoder_.run_mean(bands, T_b, T_lat);  // first 128 rows = mean
  const float* mb = decoder_.run(condition_latent(z, T_lat), T_lat, T_mb);
  const float* y = pqmf_.synthesis_run(mb, T_mb, T_y);
  const int n = std::min(n_samples, T_y);
  std::copy_n(y, n, out);
  std::fill(out + n, out + n_samples, 0.f);
}

void BraveEngine::process(const float* in, float* out, int n) {
  if (!loaded_) throw std::runtime_error("BraveEngine not loaded");
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
