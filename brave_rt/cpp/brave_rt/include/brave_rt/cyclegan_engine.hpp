#pragma once

#include <memory>
#include <string>
#include <vector>

#include "brave_rt/attr_controls.hpp"
#include "brave_rt/decoder.hpp"
#include "brave_rt/encoder.hpp"
#include "brave_rt/pqmf.hpp"
#include "brave_rt/sidechain.hpp"
#include "brave_rt/spin_pool.hpp"
#include "brave_rt/types.hpp"
#include "brave_rt/warp.hpp"

namespace brave_rt {

/**
 * Unconditional CycleGAN: PQMF_X → Enc_X (mean) → W_xy → Dec_Y → iPQMF_Y
 * → optional CausalLoudnessSidechain. Separate PQMF rings even if weights match.
 */
class CycleGanEngine {
 public:
  CycleGanEngine() = default;

  struct Precision {
    bool encoder_half = false;
    bool decoder_half = false;
  };

  void set_num_threads(int n);
  int num_threads() const { return pool_ ? pool_->size() : 1; }

  void load(const std::string& bundle_dir);
  void load(const std::string& bundle_dir, Precision precision);

  void reset_state();
  void prepare(int host_block, bool variable_block_sizes = false);
  int latency_samples() const { return prime_ < 0 ? 0 : prime_; }
  long fifo_underflows() const { return underflows_; }
  static int min_fifo_latency(int host_block);

  void set_sidechain(bool on) { sidechain_on_ = on; }
  bool sidechain() const { return sidechain_on_; }

  void process(const float* in, float* out, int n);

  bool loaded() const { return loaded_; }
  const std::string& init_mode() const { return init_mode_; }
  const std::string& model_type() const { return model_type_; }
  const std::string& display_name() const { return display_name_; }

  /**
   * Stage 7 attr API (same semantics as BraveEngine). Unconditional bundles
   * have D=0. If Dec_Y is widened (128+D) attrs are normalized, broadcast and
   * concatenated after the warped z before decode.
   * NOTE: FiLM / conditional-warp conditioning is NOT implemented yet (no
   * conditional CycleGAN ckpt); only decoder-side concat is supported.
   */
  const HostControls& host_controls() const { return host_controls_; }
  int num_attributes() const { return static_cast<int>(attr_raw_.size()); }
  void set_attr_raw(int i, float v) {
    if (i >= 0 && i < num_attributes()) attr_raw_[static_cast<size_t>(i)] = v;
  }
  void set_attrs_raw(const float* v, int n) {
    for (int i = 0; i < n && i < num_attributes(); ++i)
      attr_raw_[static_cast<size_t>(i)] = v[i];
  }
  float attr_raw(int i) const {
    return (i >= 0 && i < num_attributes()) ? attr_raw_[static_cast<size_t>(i)] : 0.f;
  }
  int attr_index(const std::string& name) const;

  static constexpr int kMaxBlock = 8192;

 private:
  bool loaded_ = false;
  bool sidechain_on_ = true;
  std::string init_mode_ = "random";
  std::string model_type_ = "cyclegan_xy";
  std::string display_name_;
  std::string enc_prefix_ = "encoder_x";
  std::string dec_prefix_ = "decoder_y";
  std::string warp_file_prefix_ = "warp_xy__";
  std::string pqmf_in_prefix_ = "pqmf_x";
  std::string pqmf_out_prefix_ = "pqmf_y";

  std::unique_ptr<SpinPool> pool_;
  CachedPqmf pqmf_x_, pqmf_y_;
  Encoder encoder_;
  Decoder decoder_;
  LatentWarp2 warp_;
  CausalLoudnessSidechain sidechain_;

  std::vector<float> fifo_in_;
  std::vector<float> fifo_out_;
  std::vector<float> z_scratch_;
  HostControls host_controls_;
  std::vector<float> attr_raw_;
  std::vector<float> z_cond_;  // ((128+D), T_lat) Dec_Y input when D>0
  int prepared_prime_ = -1;
  int prime_ = -1;
  bool primed_ = false;
  long underflows_ = 0;

  void run_ticks(const float* in, int n_samples, float* out);
};

}  // namespace brave_rt
