#include "brave_rt/cyclegan_engine.hpp"
#include "brave_rt/npy.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using brave_rt::load_npy_f32;

static int compare(const char* name, const std::vector<float>& y,
                   const std::vector<float>& ref, double gate) {
  if (y.size() != ref.size()) {
    std::cout << name << " SIZE_MISMATCH " << y.size() << " vs " << ref.size()
              << "\n";
    return 1;
  }
  double max_abs = 0, sum_sq = 0, ref_sq = 0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double d = std::abs(y[i] - ref[i]);
    if (d > max_abs) max_abs = d;
    sum_sq += d * d;
    ref_sq += static_cast<double>(ref[i]) * ref[i];
  }
  const double rmse = std::sqrt(sum_sq / y.size());
  const double snr =
      ref_sq > 0 ? 10.0 * std::log10(ref_sq / (sum_sq + 1e-30)) : 0.0;
  std::cout << name << " max_abs=" << max_abs << " rmse=" << rmse
            << " snr_db=" << snr << (max_abs < gate ? " PASS\n" : " FAIL\n");
  return max_abs < gate ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr
        << "Usage: cyclegan_parity_cli <bundle_dir> <ref_npy> <x_npy> "
           "[sidechain=0|1] [host_block=128] [gate=1e-3]\n"
        << "  Streams x through CycleGanEngine in host_block chunks and\n"
        << "  compares against ref (same length, float32 npy).\n";
    return 2;
  }
  const std::string bundle = argv[1];
  const std::string ref_path = argv[2];
  const std::string x_path = argv[3];
  const bool sc_on = argc > 4 ? (std::atoi(argv[4]) != 0) : false;
  const int host_block = argc > 5 ? std::atoi(argv[5]) : 128;
  const double gate = argc > 6 ? std::atof(argv[6]) : 1e-3;

  if (host_block < 128 || host_block % 128 != 0) {
    std::cerr << "host_block must be a positive multiple of 128\n";
    return 2;
  }

  brave_rt::CycleGanEngine eng;
  eng.load(bundle);
  eng.set_sidechain(sc_on);
  eng.prepare(host_block, false);
  eng.reset_state();

  auto x = load_npy_f32(x_path);
  auto ref = load_npy_f32(ref_path);
  const int T = static_cast<int>(std::min(x.size(), ref.size()));
  const int T_cmp = (T / host_block) * host_block;
  std::vector<float> y(static_cast<size_t>(T_cmp), 0.f);
  for (int i = 0; i < T_cmp; i += host_block)
    eng.process(x.data() + i, y.data() + i, host_block);
  ref.resize(static_cast<size_t>(T_cmp));
  std::cout << "model=" << eng.model_type() << " init=" << eng.init_mode()
            << " sidechain=" << (sc_on ? 1 : 0)
            << " host_block=" << host_block << " display=" << eng.display_name()
            << "\n";
  return compare("cyclegan_stream", y, ref, gate);
}
