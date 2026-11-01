// Stage 2 soak: stream a long signal through BraveEngine and check for
// NaN/Inf and state blow-up. Cycles golden audio, silence, white noise and a
// full-scale sine, mixing host block sizes (incl. non-multiples of 128).
#include "brave_rt/brave_engine.hpp"
#include "brave_rt/npy.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: soak <bundle_dir> <full_x.npy> [minutes=10] [threads=1]\n";
    return 2;
  }
  const double minutes = argc > 3 ? std::atof(argv[3]) : 10.0;
  brave_rt::BraveEngine eng;
  eng.load(argv[1]);
  eng.set_num_threads(argc > 4 ? std::atoi(argv[4]) : 1);
  const auto clip = brave_rt::load_npy_f32(argv[2]);

  const long long total =
      static_cast<long long>(minutes * 60.0 * brave_rt::kSampleRate);
  const int blocks[] = {128, 128, 256, 512, 64, 100, 1024, 128};
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> noise(-1.f, 1.f);
  std::vector<float> in(4096), out(4096);

  long long pos = 0;
  int bi = 0;
  double seg_peak = 0, all_peak = 0;
  long long nonfinite = 0;
  const long long seg_len = 60LL * brave_rt::kSampleRate;
  const auto t0 = std::chrono::steady_clock::now();
  while (pos < total) {
    const int n = blocks[bi++ % 8];
    // 4 regimes of ~15 s each: golden clip, silence, noise, 0 dBFS sine.
    const int regime = static_cast<int>((pos / (15 * brave_rt::kSampleRate)) % 4);
    for (int i = 0; i < n; ++i) {
      const long long p = pos + i;
      float v = 0.f;
      if (regime == 0) v = clip[static_cast<size_t>(p % clip.size())];
      else if (regime == 2) v = noise(rng);
      else if (regime == 3) v = std::sin(2.0 * M_PI * 220.0 * p / brave_rt::kSampleRate);
      in[i] = v;
    }
    eng.process(in.data(), out.data(), n);
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(out[i])) { ++nonfinite; continue; }
      const double a = std::abs(out[i]);
      if (a > seg_peak) seg_peak = a;
    }
    const long long prev = pos;
    pos += n;
    if (pos / seg_len != prev / seg_len) {
      std::cout << "minute " << pos / seg_len << " peak=" << seg_peak
                << " nonfinite=" << nonfinite << std::endl;
      if (seg_peak > all_peak) all_peak = seg_peak;
      seg_peak = 0;
    }
  }
  const double wall = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
  const double audio = static_cast<double>(total) / brave_rt::kSampleRate;
  // Decoder output is tanh*mod_sigmoid per band (|.| <= ~2) → bounded audio.
  const bool ok = nonfinite == 0 && all_peak < 10.0;
  std::cout << "threads=" << eng.num_threads() << " audio_s=" << audio << " wall_s=" << wall
            << " rtf=" << wall / audio << " peak=" << all_peak
            << " nonfinite=" << nonfinite << (ok ? " PASS\n" : " FAIL\n");
  return ok ? 0 : 1;
}
