#include "brave_rt/brave_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: rtf_bench <bundle_dir> [block_size=128] [iters=1000] "
                 "[warmup_sec=2] [threads=1] [attr_mode=1]\n"
              << "  attr_mode (Fader bundles): 1=manual (default), 2=extract, 0=extract+override.\n"
              << "  attr_mode!=1 feeds a sine+noise signal (and a 3 s extractor warmup).\n"
              << "  Warmup is max(100 calls, warmup_sec of wall time) so the core\n"
              << "  reaches its sustained clock (powersave/ondemand governors ramp\n"
              << "  over ~1-2 s); a DAW runs the engine continuously.\n";
    return 2;
  }
  const std::string bundle = argv[1];
  const int block = argc > 2 ? std::atoi(argv[2]) : 128;
  if (block % 128 != 0) {
    std::cerr << "block must be multiple of 128\n";
    return 2;
  }

  const int threads = argc > 5 ? std::atoi(argv[5]) : 1;
  brave_rt::BraveEngine eng;
  eng.load(bundle);
  eng.set_num_threads(threads);
  const int attr_mode = argc > 6 ? std::atoi(argv[6]) : 1;
  eng.set_attr_mode(attr_mode);
  std::vector<float> in(static_cast<size_t>(block), 0.01f);
  size_t sig_pos = 0;
  const auto fill_input = [&]() {
    if (argc <= 6) return;  // legacy constant input
    for (int i = 0; i < block; ++i, ++sig_pos)
      in[static_cast<size_t>(i)] =
          0.2f * std::sin(6.2831853f * 440.f * static_cast<float>(sig_pos % 44100) / 44100.f) +
          0.05f * std::sin(static_cast<float>(sig_pos) * 12.9898f);
  };
  std::vector<float> out(static_cast<size_t>(block), 0.f);

  constexpr int kWarmup = 100;
  const int kIters = argc > 3 ? std::atoi(argv[3]) : 1000;
  const double warmup_sec = argc > 4 ? std::atof(argv[4]) : 2.0;
  {
    const auto w0 = std::chrono::steady_clock::now();
    for (int i = 0;; ++i) {
      fill_input();
      eng.process(in.data(), out.data(), block);
      // extract: warm until the 131072-sample history is full (~3 s of audio)
      if (i + 1 >= kWarmup && !(attr_mode != 1 && eng.num_attributes() > 0 && !eng.extract_ready()) &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - w0)
                  .count() >= warmup_sec)
        break;
    }
  }

  std::vector<double> rtfs;
  rtfs.reserve(kIters);
  const double audio_sec = static_cast<double>(block) / 44100.0;
  for (int i = 0; i < kIters; ++i) {
    fill_input();
    const auto t0 = std::chrono::steady_clock::now();
    eng.process(in.data(), out.data(), block);
    const auto t1 = std::chrono::steady_clock::now();
    const double dt =
        std::chrono::duration<double>(t1 - t0).count();
    rtfs.push_back(dt / audio_sec);
  }
  const double mean =
      std::accumulate(rtfs.begin(), rtfs.end(), 0.0) / rtfs.size();
  double var = 0;
  for (double r : rtfs) var += (r - mean) * (r - mean);
  var /= rtfs.size();
  const double stdev = std::sqrt(var);
  std::vector<double> sorted = rtfs;
  std::sort(sorted.begin(), sorted.end());
  const auto pct = [&](double p) {
    return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
  };
  std::cout << "block=" << block << " threads=" << threads << " attr_mode=" << attr_mode
            << " extract_ready=" << (eng.extract_ready() ? 1 : 0) << " mean_rtf=" << mean << " std=" << stdev
            << " median=" << pct(0.5) << " p99=" << pct(0.99)
            << " max=" << sorted.back()
            << (mean < 1.0 ? " PASS\n" : " FAIL\n");
  return mean < 1.0 ? 0 : 1;
}
