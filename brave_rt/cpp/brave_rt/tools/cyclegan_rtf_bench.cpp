#include "brave_rt/cyclegan_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: cyclegan_rtf_bench <bundle_dir> [block_size=128] "
                 "[iters=1000] [warmup_sec=2] [threads=1] [sidechain=1]\n";
    return 2;
  }
  const std::string bundle = argv[1];
  const int block = argc > 2 ? std::atoi(argv[2]) : 128;
  if (block % 128 != 0) {
    std::cerr << "block must be multiple of 128\n";
    return 2;
  }
  const int threads = argc > 5 ? std::atoi(argv[5]) : 1;
  const bool sc_on = argc > 6 ? (std::atoi(argv[6]) != 0) : true;

  brave_rt::CycleGanEngine eng;
  eng.load(bundle);
  eng.set_sidechain(sc_on);
  eng.set_num_threads(threads);
  std::vector<float> in(static_cast<size_t>(block), 0.01f);
  std::vector<float> out(static_cast<size_t>(block), 0.f);

  constexpr int kWarmup = 100;
  const int kIters = argc > 3 ? std::atoi(argv[3]) : 1000;
  const double warmup_sec = argc > 4 ? std::atof(argv[4]) : 2.0;
  {
    const auto w0 = std::chrono::steady_clock::now();
    for (int i = 0;; ++i) {
      eng.process(in.data(), out.data(), block);
      if (i + 1 >= kWarmup &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - w0)
                  .count() >= warmup_sec)
        break;
    }
  }

  std::vector<double> rtfs;
  rtfs.reserve(kIters);
  const double audio_sec = static_cast<double>(block) / 44100.0;
  for (int i = 0; i < kIters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    eng.process(in.data(), out.data(), block);
    const auto t1 = std::chrono::steady_clock::now();
    rtfs.push_back(std::chrono::duration<double>(t1 - t0).count() / audio_sec);
  }
  const double mean =
      std::accumulate(rtfs.begin(), rtfs.end(), 0.0) / rtfs.size();
  double var = 0;
  for (double r : rtfs) var += (r - mean) * (r - mean);
  var /= rtfs.size();
  std::vector<double> sorted = rtfs;
  std::sort(sorted.begin(), sorted.end());
  const auto pct = [&](double p) {
    return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
  };
  std::cout << "cyclegan block=" << block << " threads=" << threads
            << " sidechain=" << (sc_on ? 1 : 0) << " mean_rtf=" << mean
            << " std=" << std::sqrt(var) << " median=" << pct(0.5)
            << " p99=" << pct(0.99) << " max=" << sorted.back()
            << (mean < 1.0 ? " PASS\n" : " FAIL\n");
  return mean < 1.0 ? 0 : 1;
}
