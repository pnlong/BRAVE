#include "brave_rt/descriptor_extract.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace brave_rt {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr float kEps = 1e-8f;
}  // namespace

bool DescriptorExtract::is_torch_extractable(const std::string& n) {
  return n == "rms" || n == "flatness" || n == "centroid" || n == "zcr" || n == "f0";
}

int DescriptorExtract::kind_from_name(const std::string& n) {
  if (n == "rms") return kRms;
  if (n == "flatness") return kFlatness;
  if (n == "centroid") return kCentroid;
  if (n == "zcr") return kZcr;
  return -1;
}

bool DescriptorExtract::is_supported(const std::string& n) {
  return kind_from_name(n) >= 0;
}

DescriptorExtract::DescriptorExtract(int sample_rate, int max_history)
    : sr_(sample_rate), max_history_(std::max(max_history, kNFft)) {
  hist_.assign(static_cast<size_t>(max_history_), 0.f);
  window_.resize(kNFft);
  for (int i = 0; i < kNFft; ++i)  // periodic Hann == torch.hann_window(n_fft)
    window_[static_cast<size_t>(i)] =
        static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / kNFft));
  const int nb = kNFft / 2 + 1;
  freqs_.resize(static_cast<size_t>(nb));
  for (int k = 0; k < nb; ++k)
    freqs_[static_cast<size_t>(k)] =
        static_cast<float>(0.5 * sr_ * static_cast<double>(k) / (nb - 1));
  frame_.assign(kNFft, 0.f);
  re_.assign(kNFft, 0.f);
  im_.assign(kNFft, 0.f);
  cos_.resize(kNFft / 2);
  sin_.resize(kNFft / 2);
  for (int i = 0; i < kNFft / 2; ++i) {
    cos_[static_cast<size_t>(i)] = static_cast<float>(std::cos(2.0 * kPi * i / kNFft));
    sin_[static_cast<size_t>(i)] = static_cast<float>(-std::sin(2.0 * kPi * i / kNFft));
  }
  bitrev_.resize(kNFft);
  int bits = 0;
  while ((1 << bits) < kNFft) ++bits;
  for (int i = 0; i < kNFft; ++i) {
    int r = 0;
    for (int b = 0; b < bits; ++b)
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    bitrev_[static_cast<size_t>(i)] = r;
  }
  mag_.assign(static_cast<size_t>(nb), 0.f);
  power_.assign(static_cast<size_t>(nb), 0.f);
  for (auto& f : frames_) f.assign(kMaxFrames, 0.f);
  reset();
}

void DescriptorExtract::reset() {
  std::fill(hist_.begin(), hist_.end(), 0.f);
  hist_pos_ = 0;
  total_ = 0;
  since_frame_ = 0;
  n_frames_ = 0;
  head_ = 0;
  for (auto& f : frames_) std::fill(f.begin(), f.end(), 0.f);
}

void DescriptorExtract::set_smoothing_frames(int k) {
  smooth_ = std::min(std::max(k, 1), kMaxFrames);
}

void DescriptorExtract::push(const float* mono, int n) {
  for (int i = 0; i < n; ++i) {
    hist_[static_cast<size_t>(hist_pos_)] = mono[i];
    if (++hist_pos_ == max_history_) hist_pos_ = 0;
    ++total_;
    if (++since_frame_ == kHop) {
      since_frame_ = 0;
      if (total_ >= kNFft) compute_frame();
    }
  }
}

void DescriptorExtract::fft_inplace() {
  // Iterative radix-2 DIT on (re_, im_), input already bit-reversed.
  for (int len = 2; len <= kNFft; len <<= 1) {
    const int half = len >> 1;
    const int step = kNFft / len;
    for (int i = 0; i < kNFft; i += len) {
      for (int j = 0; j < half; ++j) {
        const float wr = cos_[static_cast<size_t>(j * step)];
        const float wi = sin_[static_cast<size_t>(j * step)];
        const size_t a = static_cast<size_t>(i + j);
        const size_t b = a + static_cast<size_t>(half);
        const float tr = re_[b] * wr - im_[b] * wi;
        const float ti = re_[b] * wi + im_[b] * wr;
        re_[b] = re_[a] - tr;
        im_[b] = im_[a] - ti;
        re_[a] += tr;
        im_[a] += ti;
      }
    }
  }
}

void DescriptorExtract::compute_frame() {
  // Last n_fft samples in chronological order.
  int p = hist_pos_ - kNFft;
  if (p < 0) p += max_history_;
  const int first = std::min(kNFft, max_history_ - p);
  std::memcpy(frame_.data(), hist_.data() + p, static_cast<size_t>(first) * sizeof(float));
  if (first < kNFft)
    std::memcpy(frame_.data() + first, hist_.data(),
                static_cast<size_t>(kNFft - first) * sizeof(float));

  // rms (rectangular window), zcr (fraction of sign changes, librosa-style).
  double sumsq = 0.0;
  int zc = 0;
  bool prev_neg = std::signbit(frame_[0]);
  for (int i = 0; i < kNFft; ++i) {
    const float v = frame_[static_cast<size_t>(i)];
    sumsq += static_cast<double>(v) * v;
    if (i > 0) {
      const bool neg = std::signbit(v);
      zc += (neg != prev_neg);
      prev_neg = neg;
    }
  }
  const float rms = static_cast<float>(std::sqrt(sumsq / kNFft));
  const float zcr = static_cast<float>(zc) / static_cast<float>(kNFft);

  // Windowed STFT magnitude.
  for (int i = 0; i < kNFft; ++i) {
    const int r = bitrev_[static_cast<size_t>(i)];
    re_[static_cast<size_t>(r)] = frame_[static_cast<size_t>(i)] * window_[static_cast<size_t>(i)];
    im_[static_cast<size_t>(r)] = 0.f;
  }
  fft_inplace();

  const int nb = kNFft / 2 + 1;
  double msum = 0.0, fmsum = 0.0, psum = 0.0, logsum = 0.0;
  for (int k = 0; k < nb; ++k) {
    const float m = std::sqrt(re_[static_cast<size_t>(k)] * re_[static_cast<size_t>(k)] +
                              im_[static_cast<size_t>(k)] * im_[static_cast<size_t>(k)]);
    const float pw = m * m + kEps;  // torch: mag.pow(2) + eps
    msum += m;
    fmsum += static_cast<double>(freqs_[static_cast<size_t>(k)]) * m;
    psum += pw;
    logsum += std::log(static_cast<double>(pw) + kEps);
  }
  const float centroid =
      static_cast<float>(fmsum / std::max(msum, static_cast<double>(kEps)));
  const float geo = static_cast<float>(std::exp(logsum / nb));
  const float arith = static_cast<float>(psum / nb) + kEps;
  const float flatness = geo / arith;

  const size_t w = static_cast<size_t>(head_);
  frames_[kRms][w] = rms;
  frames_[kFlatness][w] = flatness;
  frames_[kCentroid][w] = centroid;
  frames_[kZcr][w] = zcr;
  head_ = (head_ + 1) % kMaxFrames;
  if (n_frames_ < kMaxFrames) ++n_frames_;
}

float DescriptorExtract::value(int kind) const {
  if (kind < 0 || kind >= kNumKinds || n_frames_ == 0) return 0.f;
  const int k = std::min(smooth_, n_frames_);
  double acc = 0.0;
  int idx = head_;
  for (int i = 0; i < k; ++i) {
    idx = (idx + kMaxFrames - 1) % kMaxFrames;
    acc += frames_[static_cast<size_t>(kind)][static_cast<size_t>(idx)];
  }
  return static_cast<float>(acc / k) * scale_[static_cast<size_t>(kind)];
}

}  // namespace brave_rt
