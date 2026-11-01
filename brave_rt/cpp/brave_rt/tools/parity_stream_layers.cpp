#include "brave_rt/brave_engine.hpp"
#include "brave_rt/npy.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static void stats(const char* name, const std::vector<float>& a,
                  const std::vector<float>& b) {
  if (a.size() != b.size()) {
    std::cout << name << " SIZE " << a.size() << " vs " << b.size() << "\n";
    return;
  }
  double max_abs = 0;
  for (size_t i = 0; i < a.size(); ++i)
    max_abs = std::max(max_abs, (double)std::abs(a[i] - b[i]));
  std::cout << name << " max_abs=" << max_abs << "\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: parity_stream_layers <bundle> <stream_layers_dir>\n";
    return 2;
  }
  brave_rt::BraveEngine eng;
  eng.load(argv[1]);
  const std::string g = argv[2];
  for (int bi = 0; bi < 2; ++bi) {
    auto x = brave_rt::load_npy_f32(g + "/b" + std::to_string(bi) + "_x.npy");
    auto bands = eng.tick_pqmf_bands(x.data());
    stats(("b" + std::to_string(bi) + "_pqmf").c_str(), bands,
          brave_rt::load_npy_f32(g + "/b" + std::to_string(bi) +
                                 "_pqmf_bands.npy"));
    auto z = eng.tick_z_mean(bands.data());
    stats(("b" + std::to_string(bi) + "_z").c_str(), z,
          brave_rt::load_npy_f32(g + "/b" + std::to_string(bi) +
                                 "_z_mean.npy"));
    auto mb = eng.tick_decoder_mb(z.data());
    stats(("b" + std::to_string(bi) + "_mb").c_str(), mb,
          brave_rt::load_npy_f32(g + "/b" + std::to_string(bi) +
                                 "_decoder_mb.npy"));
    // note: tick_e2e would re-run pqmf/enc — instead synth from mb
    auto y = eng.pqmf().synthesis(mb.data(), 8);
    if (y.size() > 128) y.resize(128);
    stats(("b" + std::to_string(bi) + "_y").c_str(), y,
          brave_rt::load_npy_f32(g + "/b" + std::to_string(bi) + "_y.npy"));
  }
  return 0;
}
