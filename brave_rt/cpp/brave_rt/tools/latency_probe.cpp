// Latency probe for BraveEngine. Three components of what a player feels:
//
//  1. Model (algorithmic) latency: onset-in → onset-out, measured on
//     (a) unit clicks, (b) 5 ms decaying noise bursts (tap-like), both at known
//     positions with random spacing, and (c) the golden tap clip via envelope
//     cross-correlation (also run on the PyTorch golden output as a cross-check;
//     C++ matches PyTorch to ~1e-6, so this is the model's own latency).
//  2. Host buffering latency: extra delay the Cr=128 FIFO adds per host block
//     size, measured as the exact output lag vs. the block-128 stream.
//  3. Compute deadline: per-callback wall time vs. the buffer period at each
//     host block (ms percentiles + fraction of callbacks that would underrun).
//
// Total felt latency ≈ host I/O buffers (DAW/driver) + (2) + (1).
#include "brave_rt/brave_engine.hpp"
#include "brave_rt/npy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using brave_rt::BraveEngine;
using brave_rt::kSampleRate;

namespace {

double ms(double samples) { return 1000.0 * samples / kSampleRate; }

std::vector<float> run_engine(BraveEngine& e, const std::vector<float>& x,
                              int block) {
  e.reset_state();
  std::vector<float> y(x.size(), 0.f);
  size_t i = 0;
  for (; i + block <= x.size(); i += block) e.process(&x[i], &y[i], block);
  return y;
}

/** 1 ms RMS envelope at hop 1 sample (moving window). */
std::vector<double> envelope(const std::vector<float>& x, int win = 44) {
  std::vector<double> env(x.size(), 0.0);
  double acc = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    acc += static_cast<double>(x[i]) * x[i];
    if (i >= static_cast<size_t>(win)) acc -= static_cast<double>(x[i - win]) * x[i - win];
    env[i] = std::sqrt(std::max(acc, 0.0) / win);
  }
  return env;
}

struct Stats {
  double mean = 0, median = 0, sd = 0, lo = 0, hi = 0;
  int n = 0;
};

Stats stats(std::vector<double> v) {
  Stats s;
  s.n = static_cast<int>(v.size());
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  s.median = v[v.size() / 2];
  for (double d : v) s.sd += (d - s.mean) * (d - s.mean);
  s.sd = std::sqrt(s.sd / v.size());
  s.lo = v.front();
  s.hi = v.back();
  return s;
}

/**
 * For each known input onset, find the output onset: first sample in
 * [onset, onset + max_lag) where the output envelope exceeds
 * max(frac * local peak, 4 * pre-onset floor). Returns lags in samples.
 */
std::vector<double> onset_lags(const std::vector<float>& y,
                               const std::vector<long>& onsets, double frac,
                               int max_lag) {
  const auto env = envelope(y);
  std::vector<double> lags;
  for (long o : onsets) {
    if (o < 2205 || o + max_lag >= static_cast<long>(y.size())) continue;
    double floor_rms = 0;
    for (long i = o - 2205; i < o - 441; ++i) floor_rms = std::max(floor_rms, env[i]);
    double peak = 0;
    for (long i = o; i < o + max_lag; ++i) peak = std::max(peak, env[i]);
    const double thr = std::max(frac * peak, 4.0 * floor_rms);
    if (peak <= thr) continue;  // no clear response above the floor
    for (long i = o; i < o + max_lag; ++i)
      if (env[i] > thr) {
        // envelope window is trailing 1 ms → onset is ~win/2 earlier
        lags.push_back(static_cast<double>(i - o) - 22.0);
        break;
      }
  }
  return lags;
}

/** Lag (samples) maximizing correlation of log-ish envelopes, in [0, max_lag]. */
int xcorr_lag(const std::vector<float>& x, const std::vector<float>& y,
              int max_lag) {
  auto ex = envelope(x), ey = envelope(y);
  const size_t n = std::min(ex.size(), ey.size());
  auto norm = [n](std::vector<double>& e) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m += e[i];
    m /= n;
    for (size_t i = 0; i < n; ++i) e[i] -= m;
  };
  norm(ex);
  norm(ey);
  int best = 0;
  double best_c = -1e300;
  for (int lag = 0; lag <= max_lag; ++lag) {
    double c = 0;
    for (size_t i = 0; i + lag < n; ++i) c += ex[i] * ey[i + lag];
    if (c > best_c) best_c = c, best = lag;
  }
  return best;
}

/** Exact lag of b relative to a (b[i+lag] == a[i]) by min abs diff. */
int exact_lag(const std::vector<float>& a, const std::vector<float>& b,
              int max_lag, size_t n_valid, double* err_out) {
  int best = 0;
  double best_e = 1e300;
  const size_t n = n_valid - max_lag;  // only samples both runs produced
  for (int lag = 0; lag <= max_lag; ++lag) {
    double e = 0;
    for (size_t i = 44100; i < n; ++i) e = std::max(e, static_cast<double>(std::abs(a[i] - b[i + lag])));
    if (e < best_e) best_e = e, best = lag;
  }
  if (err_out) *err_out = best_e;
  return best;
}

void print_onsets(const char* name, const std::vector<double>& lags, int total) {
  const Stats s = stats(lags);
  std::printf("  %-26s detected %3d/%3d  mean %6.2f ms  median %6.2f  sd %5.2f  [%6.2f, %6.2f]\n",
              name, s.n, total, ms(s.mean), ms(s.median), ms(s.sd), ms(s.lo), ms(s.hi));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "Usage: latency_probe <bundle_dir> <goldens_dir> [threads=1]\n");
    return 2;
  }
  const std::string bundle = argv[1], gold = argv[2];
  const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
  BraveEngine eng;
  eng.load(bundle);
  eng.set_num_threads(threads);

  // ---------------------------------------------------------------- 1. model
  std::printf("== 1. Model latency (onset in -> onset out), host block 128\n");
  std::mt19937 rng(7);
  const int n_events = 60;
  std::vector<long> onsets;
  const long len = static_cast<long>(n_events) * 22050 + 44100;
  std::vector<float> clicks(len, 0.f), bursts(len, 0.f);
  {
    std::uniform_int_distribution<int> gap(13230, 22050);  // 300–500 ms
    std::normal_distribution<float> nz(0.f, 1.f);
    long pos = 22050;
    for (int i = 0; i < n_events && pos + 22050 < len; ++i) {
      onsets.push_back(pos);
      clicks[pos] = 0.9f;
      for (int k = 0; k < 220; ++k)  // 5 ms, exp decay (tau 1 ms)
        bursts[pos + k] = 0.6f * nz(rng) * std::exp(-k / 44.1f);
      pos += gap(rng);
    }
  }
  const int max_lag = 4410;  // 100 ms search window
  for (double frac : {0.1, 0.25}) {
    std::printf(" threshold = %.0f%% of local peak\n", frac * 100);
    print_onsets("clicks", onset_lags(run_engine(eng, clicks, 128), onsets, frac, max_lag),
                 static_cast<int>(onsets.size()));
    print_onsets("noise bursts (5 ms)",
                 onset_lags(run_engine(eng, bursts, 128), onsets, frac, max_lag),
                 static_cast<int>(onsets.size()));
  }
  {
    auto x = brave_rt::load_npy_f32(gold + "/full_x.npy");
    auto y_cpp = run_engine(eng, x, 128);
    auto y_py = brave_rt::load_npy_f32(gold + "/stream128_y.npy");
    y_py.resize(x.size());
    const int lc = xcorr_lag(x, y_cpp, max_lag), lp = xcorr_lag(x, y_py, max_lag);
    std::printf("  %-26s C++ %d samples (%.2f ms) | PyTorch golden %d samples (%.2f ms)\n",
                "golden clip env-xcorr", lc, ms(lc), lp, ms(lp));
  }

  // -------------------------------------------------------------- 2. buffering
  std::printf("\n== 2. Host buffering latency (extra lag vs. block-128 stream)\n");
  {
    auto x = brave_rt::load_npy_f32(gold + "/full_x.npy");
    x.resize(std::min<size_t>(x.size(), 6 * 44100));
    const auto ref = run_engine(eng, x, 128);
    for (int b : {32, 64, 100, 128, 256, 441, 480, 512, 1024}) {
      double err = 0;
      const auto yb = run_engine(eng, x, b);
      const size_t n_valid = std::min(x.size() / 128 * 128, x.size() / b * b);
      const int lag = exact_lag(ref, yb, 256, n_valid, &err);
      std::printf("  block %5d: +%3d samples (%5.2f ms)  reported %3d  underflows %ld"
                  "  match max-abs %.2g%s\n", b, lag, ms(lag), eng.latency_samples(),
                  eng.fifo_underflows(), err, b % 128 == 0 ? "" : "  (non-multiple of 128)");
    }
  }

  // ---------------------------------------------------------------- 3. compute
  std::printf("\n== 3. Compute deadline (threads=%d): callback wall time vs buffer period\n",
              eng.num_threads());
  {
    auto x = brave_rt::load_npy_f32(gold + "/full_x.npy");
    for (int b : {128, 256, 512}) {
      eng.reset_state();
      std::vector<float> out(b);
      const double period = static_cast<double>(b) / kSampleRate;
      const auto w0 = std::chrono::steady_clock::now();
      long pos = 0;
      auto next = [&]() {
        if (pos + b > static_cast<long>(x.size())) pos = 0;
        eng.process(&x[pos], out.data(), b);
        pos += b;
      };
      while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 2.0)
        next();  // let the clock ramp
      // Real-time schedule: callbacks every `period`; the core idles in between,
      // like a host. A callback longer than `period` is an underrun.
      const int n = static_cast<int>(10.0 / period);
      std::vector<double> t(n);
      auto due = std::chrono::steady_clock::now();
      for (int i = 0; i < n; ++i) {
        due += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(period));
        const auto t0 = std::chrono::steady_clock::now();
        next();
        const auto t1 = std::chrono::steady_clock::now();
        t[i] = std::chrono::duration<double>(t1 - t0).count();
        while (std::chrono::steady_clock::now() < due) {}  // spin to next deadline
      }
      const int misses = static_cast<int>(std::count_if(t.begin(), t.end(), [&](double v) { return v > period; }));
      std::sort(t.begin(), t.end());
      std::printf("  block %4d (period %5.2f ms): median %5.3f ms  p99 %5.3f  p99.9 %5.3f  max %5.3f"
                  "  | RTF median %.3f p99 %.3f | underruns %d/%d (%.3f%%)\n",
                  b, period * 1e3, t[n / 2] * 1e3, t[n * 99 / 100] * 1e3, t[n * 999 / 1000] * 1e3,
                  t.back() * 1e3, t[n / 2] / period, t[n * 99 / 100] / period, misses, n,
                  100.0 * misses / n);
    }
  }
  return 0;
}
