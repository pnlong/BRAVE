#include "brave_rt/brave_engine.hpp"
#include "brave_rt/npy.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using brave_rt::load_npy_f32;

static void stats(const char* name, const std::vector<float>& a,
                  const std::vector<float>& b) {
  if (a.size() != b.size()) {
    std::cout << name << " SIZE_MISMATCH " << a.size() << " vs " << b.size()
              << "\n";
    return;
  }
  double max_abs = 0, sum_sq = 0, ref_sq = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(a[i] - b[i]);
    if (d > max_abs) max_abs = d;
    sum_sq += d * d;
    ref_sq += static_cast<double>(b[i]) * b[i];
  }
  const double rmse = std::sqrt(sum_sq / a.size());
  const double snr =
      ref_sq > 0 ? 10.0 * std::log10(ref_sq / (sum_sq + 1e-30)) : 0.0;
  std::cout << name << " max_abs=" << max_abs << " rmse=" << rmse
            << " snr_db=" << snr << "\n";
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: parity_cli <bundle_dir> <goldens_dir> <stage> "
                 "<block|stream> [host_block=128] [threads=1]\n"
              << "  stage: pqmf|encode|decode|e2e\n"
              << "  host_block (stream only): multiple of 128; exercises the\n"
              << "  batched multi-tick path when > 128\n";
    return 2;
  }
  const std::string bundle = argv[1];
  const std::string gold = argv[2];
  const std::string stage = argv[3];
  const std::string mode = argv[4];

  brave_rt::BraveEngine eng;
  eng.load(bundle);
  if (argc > 6) eng.set_num_threads(std::atoi(argv[6]));

  if (mode == "block") {
    auto x = load_npy_f32(gold + "/block128_x.npy");
    // (1,1,128) → flat 128
    if (x.size() < 128) {
      std::cerr << "block128_x too short\n";
      return 1;
    }
    if (stage == "pqmf") {
      auto bands = eng.tick_pqmf_bands(x.data());
      auto ref = load_npy_f32(gold + "/block128_pqmf_bands.npy");
      stats("pqmf", bands, ref);
    } else if (stage == "encode") {
      auto bands = eng.tick_pqmf_bands(x.data());
      auto z = eng.tick_z_mean(bands.data());
      auto ref = load_npy_f32(gold + "/block128_z_mean.npy");
      stats("z_mean", z, ref);
    } else if (stage == "decode") {
      // Use golden z for isolated decode (reset decoder-only path via e2e z)
      auto zref = load_npy_f32(gold + "/block128_z_mean.npy");
      auto mb = eng.tick_decoder_mb(zref.data());
      auto ref = load_npy_f32(gold + "/block128_decoder_mb.npy");
      stats("decoder_mb", mb, ref);
    } else if (stage == "e2e") {
      auto y = eng.tick_e2e(x.data());
      auto ref = load_npy_f32(gold + "/block128_y.npy");
      // ref may be (1,1,128)
      if (ref.size() > y.size()) ref.resize(y.size());
      stats("e2e_block", y, ref);
    } else {
      std::cerr << "unknown stage\n";
      return 2;
    }
  } else if (mode == "stream") {
    auto x = load_npy_f32(gold + "/full_x.npy");
    auto ref = load_npy_f32(gold + "/stream128_y.npy");
    const int T = static_cast<int>(x.size());
    // Compare only whole Cr=128 ticks (tail handled by host FIFO pad separately).
    const int host_block = argc > 5 ? std::atoi(argv[5]) : 128;
    if (host_block < 128 || host_block % 128 != 0) {
      std::cerr << "host_block must be a positive multiple of 128\n";
      return 2;
    }
    const int T_cmp = (T / host_block) * host_block;
    std::vector<float> y(static_cast<size_t>(T_cmp), 0.f);
    eng.reset_state();
    for (int i = 0; i < T_cmp; i += host_block) {
      eng.process(x.data() + i, y.data() + i, host_block);
    }
    ref.resize(static_cast<size_t>(T_cmp));
    std::cout << "host_block=" << host_block << " threads=" << eng.num_threads() << " ";
    stats("e2e_stream128", y, ref);
    const double gate = 1e-3;
    double max_abs = 0;
    for (size_t i = 0; i < y.size(); ++i)
      max_abs = std::max(max_abs, static_cast<double>(std::abs(y[i] - ref[i])));
    std::cout << (max_abs < gate ? "PASS\n" : "FAIL\n");
    return max_abs < gate ? 0 : 1;
  } else {
    std::cerr << "mode must be block|stream\n";
    return 2;
  }
  return 0;
}
