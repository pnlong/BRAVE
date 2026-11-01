#include "brave_rt/sidechain.hpp"
#include "brave_rt/npy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using brave_rt::load_npy_f32;

/** Compare C++ CausalLoudnessSidechain vs a precomputed gain/ref npy. */
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: sidechain_parity_cli <x.npy> <y_in.npy> <y_ref.npy> "
                 "[hop=128]\n"
              << "  Applies sidechain(x, y_in) and compares to y_ref.\n";
    return 2;
  }
  const int hop = argc > 4 ? std::atoi(argv[4]) : 128;
  auto x = load_npy_f32(argv[1]);
  auto y = load_npy_f32(argv[2]);
  auto ref = load_npy_f32(argv[3]);
  const int n = static_cast<int>(std::min({x.size(), y.size(), ref.size()}));
  y.resize(static_cast<size_t>(n));
  brave_rt::CausalLoudnessSidechain sc(hop, -40.f, 8);
  sc.process(x.data(), y.data(), n);

  double max_abs = 0, sum_sq = 0, ref_sq = 0;
  for (int i = 0; i < n; ++i) {
    const double d = std::abs(y[static_cast<size_t>(i)] -
                              ref[static_cast<size_t>(i)]);
    if (d > max_abs) max_abs = d;
    sum_sq += d * d;
    ref_sq += static_cast<double>(ref[static_cast<size_t>(i)]) *
              ref[static_cast<size_t>(i)];
  }
  const double snr =
      ref_sq > 0 ? 10.0 * std::log10(ref_sq / (sum_sq + 1e-30)) : 0.0;
  constexpr double gate = 1e-5;
  std::cout << "sidechain_parity max_abs=" << max_abs << " snr_db=" << snr
            << (max_abs < gate ? " PASS\n" : " FAIL\n");
  return max_abs < gate ? 0 : 1;
}
