#pragma once

/**
 * Stage 8 — hands-free attribute descriptors (C++ port of the spirit of
 * RAVE/rave/fader/export/torch_descriptors.py :: TorchDescriptorExtract).
 *
 * nn~ extractable set: {rms, flatness, centroid, zcr, f0}. Implemented here:
 * rms, centroid, flatness, zcr. f0 (and all timbral attrs: bandwidth,
 * sharpness, booming, …) are NOT extracted → the engine leaves them at the
 * manual / default value, exactly like nn~ does for non-extractable rows.
 *
 * Differences vs the torch module (deliberate, RT-friendly):
 *  - Incremental: one n_fft=2048 STFT frame per 512 new samples (trailing
 *    window, no centering / no look-ahead) instead of re-analysing the whole
 *    3 s history every block.
 *  - Output is ONE scalar per descriptor = mean of the last `smoothing_frames`
 *    frames (default 8 ≈ 93 ms), in RAW training units (Hz for centroid,
 *    linear amplitude for rms, …), not normalized.
 *  - zcr is computed (nn~ currently emits an all-zero row for zcr / f0).
 *  - ready() mirrors nn~ `_extract_history_ready`: false until
 *    max_history (131072) samples have been pushed since reset().
 *
 * No heap allocation after construction; push() is audio-thread safe.
 */

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace brave_rt {

class DescriptorExtract {
 public:
  static constexpr int kNFft = 2048;
  static constexpr int kHop = 512;
  static constexpr int kMaxHistory = 131072;  // ≈ 2.97 s @ 44.1 kHz
  static constexpr int kMaxFrames = kMaxHistory / kHop;

  enum Kind { kRms = 0, kFlatness, kCentroid, kZcr, kNumKinds };

  /** True for names in TORCH_EXTRACTABLE that this port actually computes. */
  static bool is_supported(const std::string& name);
  /** True for any name in TORCH_EXTRACTABLE (incl. f0, which we skip). */
  static bool is_torch_extractable(const std::string& name);
  /** Kind index for a supported name, or -1. */
  static int kind_from_name(const std::string& name);

  explicit DescriptorExtract(int sample_rate = 44100, int max_history = kMaxHistory);

  /** Clear history, frame ring and sample counter (ready() → false). */
  void reset();

  /** Push mono samples; computes a frame for every 512 new samples. */
  void push(const float* mono, int n);

  /** History filled to max_history since last reset(). */
  bool ready() const { return total_ >= static_cast<long>(max_history_); }
  long samples_pushed() const { return total_; }
  int frames_available() const { return n_frames_; }

  /** Number of trailing frames averaged by value() (1..kMaxFrames). */
  void set_smoothing_frames(int k);
  int smoothing_frames() const { return smooth_; }

  /** Optional per-kind multiplicative calibration (nn~ _row_scale), default 1. */
  void set_scale(int kind, float s) {
    if (kind >= 0 && kind < kNumKinds) scale_[static_cast<size_t>(kind)] = s;
  }

  /** Raw-unit scalar for a kind (mean of last smoothing_frames frames); 0 if none. */
  float value(int kind) const;

 private:
  int sr_;
  int max_history_;
  int smooth_ = 8;
  long total_ = 0;
  int since_frame_ = 0;
  int n_frames_ = 0;  // valid frames in ring (≤ kMaxFrames)
  int head_ = 0;      // next ring write position

  std::vector<float> hist_;  // sample ring (max_history_)
  int hist_pos_ = 0;

  // Scratch (all preallocated)
  std::vector<float> window_;     // Hann (periodic, = torch.hann_window)
  std::vector<float> freqs_;      // n_fft/2+1 bin centre Hz
  std::vector<float> frame_;      // last n_fft samples (linear order)
  std::vector<float> re_, im_;    // FFT work
  std::vector<float> cos_, sin_;  // twiddles (n_fft/2)
  std::vector<int> bitrev_;
  std::vector<float> mag_, power_;
  std::array<std::vector<float>, kNumKinds> frames_;  // per-kind ring
  std::array<float, kNumKinds> scale_{{1.f, 1.f, 1.f, 1.f}};

  void compute_frame();
  void fft_inplace();
};

}  // namespace brave_rt
