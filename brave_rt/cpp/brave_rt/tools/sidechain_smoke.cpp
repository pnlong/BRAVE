#include "brave_rt/sidechain.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  brave_rt::CausalLoudnessSidechain sc(128, -40.f, 8);
  const int hops = 64;
  const int n = 128 * hops;
  std::vector<float> x(n, 0.f), y(n, 1.f);
  for (int i = 0; i < 128 * 2; ++i) x[i] = 0.5f;
  sc.process(x.data(), y.data(), n);
  const float g_loud = y[128];
  const float g_quiet = y[n - 1];
  std::cout << "gain_loud=" << g_loud << " gain_quiet_end=" << g_quiet << "\n";
  // Soft sidechain ducks over time; require clear relative attenuation.
  const bool ok = g_loud > 0.7f && g_quiet < g_loud * 0.5f;
  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
