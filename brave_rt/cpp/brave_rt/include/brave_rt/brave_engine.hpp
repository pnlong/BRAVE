#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "brave_rt/attr_controls.hpp"
#include "brave_rt/decoder.hpp"
#include "brave_rt/descriptor_extract.hpp"
#include "brave_rt/encoder.hpp"
#include "brave_rt/pqmf.hpp"
#include "brave_rt/spin_pool.hpp"
#include "brave_rt/types.hpp"

namespace brave_rt {

/**
 * Vanilla BRAVE streaming engine.
 * Graph: PQMF → Enc (mean) → Dec → iPQMF. Cr=128 ticks.
 */
class BraveEngine {
 public:
  BraveEngine() = default;

  /**
   * Weight storage per module (compute is always fp32; PQMF always fp32).
   * fp16 halves weight bandwidth, the RTF bottleneck at Cr=128.
   */
  struct Precision {
    bool encoder_half = false;
    bool decoder_half = false;
  };

  /**
   * Intra-layer worker threads (total, including the audio thread). 1 = off.
   * Adds no latency. Call from a non-realtime thread (spawns/joins threads).
   * Layers ≥ kParallelMinWeights split their output panels statically, so
   * each core keeps its weight slice in its own L2.
   */
  void set_num_threads(int n);
  int num_threads() const { return pool_ ? pool_->size() : 1; }

  /** Load npy_weights/ (or flat pqmf_*.npy + npy_weights) from a bundle directory. */
  void load(const std::string& bundle_dir);
  void load(const std::string& bundle_dir, Precision precision);

  void reset_state();

  /**
   * Host block policy (call from prepareToPlay, before reset_state()).
   * Output FIFO is pre-filled with latency_samples() zeros so non-multiple-of-
   * 128 blocks never underflow mid-stream:
   *   multiple of 128           → 0 extra samples
   *   constant block n          → 128 - gcd(n, 128)   (minimal, e.g. 441 → 127)
   *   variable_block_sizes=true → 127 (safe for any block sequence)
   * Without prepare(), the first process() after reset_state() primes using
   * its n (constant-block assumption).
   */
  void prepare(int host_block, bool variable_block_sizes = false);

  /** Extra host-side latency in samples (report via setLatencySamples, plus
   *  the model's own onset latency if you want plugin delay compensation). */
  int latency_samples() const { return prime_ < 0 ? 0 : prime_; }

  /** Calls since reset that had to zero-pad output (should stay 0). */
  long fifo_underflows() const { return underflows_; }

  static int min_fifo_latency(int host_block);

  /**
   * Process mono float samples. Prefer n multiple of 128.
   * Internal FIFO accumulates if n is arbitrary. All whole Cr=128 ticks
   * buffered in one call run as a single batched pass (bit-for-bit the same
   * streaming result up to float summation order, but reuses weights).
   * Allocation-free after load() for n <= kMaxBlock.
   */
  void process(const float* in, float* out, int n);

  bool loaded() const { return loaded_; }

  static constexpr int kMaxBlock = 8192;

  /** Layer-wise helpers for parity_cli (one 128-sample tick, updates state). */
  std::vector<float> tick_pqmf_bands(const float* in128);
  std::vector<float> tick_z_mean(const float* bands16x8);
  std::vector<float> tick_decoder_mb(const float* z128);
  std::vector<float> tick_e2e(const float* in128);

  /**
   * Stage 7 conditioning (Fader bundles, D = decoder.num_attr_channels() > 0).
   * Host sets RAW training-unit values (discrete: class index); the engine
   * normalizes to [-1,1], broadcasts over T_lat and concats after the 128 z
   * rows before decode. D == 0 (vanilla): all setters are no-ops and the
   * hot path is unchanged. Setters are plain float stores (safe from a UI
   * thread; a torn read is at worst one block of stale value).
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
  /** Index of attribute by schema name, or -1. */
  int attr_index(const std::string& name) const;

  /**
   * Stage 8 attribute source (match RAVE/rave/fader/export/nn_module.py).
   *   1 = manual: host knobs only (Stage 7, DEFAULT, bit-identical goldens)
   *   2 = extract-only: continuous attrs in TORCH_EXTRACTABLE∩{rms,flatness,
   *       centroid,zcr} come from the input audio once ~3 s of history exists
   *       (`extract_ready()`), else the manual/default value; discrete and
   *       non-extractable (f0 skipped, timbral) attrs stay manual/default.
   *   0 = extract + per-attr override: like 2, but attrs flagged with
   *       set_attr_override(i,true) use the manual value (scale knobs: n/a).
   * Setter is a relaxed atomic store (UI thread safe). Switching 1→{0,2}
   * restarts the extractor history. No-op effect when D == 0 (vanilla).
   * Manual values (set_attr_raw) are never overwritten; the value actually
   * fed to the decoder is attr_effective().
   */
  void set_attr_mode(int mode) {
    attr_mode_.store(mode == 0 || mode == 2 ? mode : 1, std::memory_order_relaxed);
  }
  int attr_mode() const { return attr_mode_.load(std::memory_order_relaxed); }
  void set_attr_override(int i, bool on) {
    if (i >= 0 && i < num_attributes()) attr_override_[static_cast<size_t>(i)] = on ? 1 : 0;
  }
  /** True once the extractor has a full history (nn~ _extract_history_ready). */
  bool extract_ready() const { return extract_.ready(); }
  /** Value last fed to the decoder (raw units): manual in mode 1. */
  float attr_effective(int i) const {
    if (i < 0 || i >= num_attributes()) return 0.f;
    return attr_mode() == 1 ? attr_raw_[static_cast<size_t>(i)]
                            : attr_eff_[static_cast<size_t>(i)];
  }
  /** True if schema attr i is driven by the extractor in mode 0/2. */
  bool attr_extractable(int i) const {
    return i >= 0 && i < num_attributes() && attr_kind_[static_cast<size_t>(i)] >= 0;
  }
  DescriptorExtract& extractor() { return extract_; }

  Encoder& encoder() { return encoder_; }
  Decoder& decoder() { return decoder_; }
  CachedPqmf& pqmf() { return pqmf_; }

 private:
  bool loaded_ = false;
  std::unique_ptr<SpinPool> pool_;
  CachedPqmf pqmf_;
  Encoder encoder_;
  Decoder decoder_;
  HostControls host_controls_;
  std::vector<float> attr_raw_;  // manual (host) values, raw units
  std::vector<float> attr_eff_;  // Stage 8: manual with extracted rows applied
  std::vector<int> attr_kind_;   // DescriptorExtract::Kind or -1 (not extracted)
  std::vector<uint8_t> attr_override_;
  std::atomic<int> attr_mode_{1};
  int last_attr_mode_ = 1;       // audio-thread only
  DescriptorExtract extract_{kSampleRate};
  std::vector<float> z_cond_;     // ((128+D), T_lat) decoder input when D>0
  std::vector<float> fifo_in_;
  std::vector<float> fifo_out_;
  int prepared_prime_ = -1;  // from prepare(); -1 = auto
  int prime_ = -1;           // active; -1: derive on first process() after reset
  bool primed_ = false;
  long underflows_ = 0;

  /** z (128,T_lat) → decoder input: z itself (D==0) or z_cond_ with attrs. */
  const float* condition_latent(const float* z, int T_lat);

  /** Stage 8: push audio, refresh attr_eff_ (mode != 1, D > 0). */
  void update_extracted_attrs(const float* in, int n_samples, int mode);

  /** n_samples (multiple of 128) through PQMF→Enc→Dec→iPQMF into out. */
  void run_ticks(const float* in, int n_samples, float* out);
};

}  // namespace brave_rt
