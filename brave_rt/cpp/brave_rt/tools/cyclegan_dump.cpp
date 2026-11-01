#include "brave_rt/cyclegan_engine.hpp"
#include "brave_rt/npy.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

/** Minimal float32 C-order .npy writer (1D). */
static void save_npy_f32(const std::string& path, const float* data, int n) {
  std::ofstream f(path, std::ios::binary);
  const char* magic = "\x93NUMPY";
  f.write(magic, 6);
  const char ver[2] = {1, 0};
  f.write(ver, 2);
  char header[128];
  int hlen = std::snprintf(
      header, sizeof(header),
      "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", n);
  while ((10 + 2 + hlen + 1) % 16 != 0) header[hlen++] = ' ';
  header[hlen++] = '\n';
  uint16_t hl = static_cast<uint16_t>(hlen);
  f.write(reinterpret_cast<char*>(&hl), 2);
  f.write(header, hlen);
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(n * sizeof(float)));
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "Usage: cyclegan_dump <bundle> <x.npy> <out.npy> [sc=0] [block=128]\n");
    return 2;
  }
  const bool sc = argc > 4 && std::atoi(argv[4]) != 0;
  const int block = argc > 5 ? std::atoi(argv[5]) : 128;
  brave_rt::CycleGanEngine eng;
  eng.load(argv[1]);
  eng.set_sidechain(sc);
  eng.prepare(block, false);
  eng.reset_state();
  auto x = brave_rt::load_npy_f32(argv[2]);
  const int T = (static_cast<int>(x.size()) / block) * block;
  std::vector<float> y(static_cast<size_t>(T), 0.f);
  for (int i = 0; i < T; i += block)
    eng.process(x.data() + i, y.data() + i, block);
  save_npy_f32(argv[3], y.data(), T);
  std::printf("wrote %s (%d)\n", argv[3], T);
  return 0;
}
