#include "brave_rt/npy.hpp"
#include "brave_rt/pqmf.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace brave_rt;

static double snr_db(const std::vector<float>& ref, const std::vector<float>& est) {
  double num = 0, den = 0;
  const size_t n = std::min(ref.size(), est.size());
  for (size_t i = 0; i < n; ++i) {
    num += double(ref[i]) * double(ref[i]);
    const double d = double(ref[i]) - double(est[i]);
    den += d * d;
  }
  return 10.0 * std::log10((num + 1e-20) / (den + 1e-20));
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <forward.npy> <inverse.npy> <input_x.npy> <ref_bands.npy> [block]\n";
    return 2;
  }
  const std::string fw = argv[1], inv = argv[2], xin = argv[3], refb = argv[4];
  const int block = (argc > 5) ? std::atoi(argv[5]) : 128;

  CachedPqmf pqmf;
  pqmf.load_weights_npy(fw, inv);
  pqmf.reset();

  std::vector<size_t> shp;
  auto x = load_npy_f32(xin, &shp);
  // Expect (1,1,T) or (T,)
  size_t T = x.size();
  if (shp.size() == 3) T = shp[2];

  auto ref_bands = load_npy_f32(refb, &shp);
  // (1,16,T_b) or (16,T_b)
  int T_b_ref = static_cast<int>(ref_bands.size() / kNBand);

  // Streaming analysis in blocks
  pqmf.reset();
  std::vector<float> bands_all;
  for (int start = 0; start + block <= static_cast<int>(T); start += block) {
    auto b = pqmf.analysis(x.data() + start, block);
    bands_all.insert(bands_all.end(), b.begin(), b.end());
  }
  // Compare first T_b_ref frames channel-major: we stored analysis per block as
  // (16, block/16) concatenated along time — need to interleave by channel.
  // Easier: process full signal in one call for layer check if T multiple of 16.
  pqmf.reset();
  auto bands_one = pqmf.analysis(x.data(), static_cast<int>(T));
  const int T_b = static_cast<int>(bands_one.size() / kNBand);
  double max_abs = 0;
  const size_t ncmp = std::min(bands_one.size(), ref_bands.size());
  for (size_t i = 0; i < ncmp; ++i) {
    max_abs = std::max(max_abs, std::abs(double(bands_one[i]) - double(ref_bands[i])));
  }
  std::cout << "analysis max_abs vs ref=" << max_abs << " T_b=" << T_b
            << " T_b_ref=" << T_b_ref << "\n";

  // Streaming analysis vs one-shot analysis (same zero-init state continuity)
  pqmf.reset();
  std::vector<float> bands_stream;
  // Channel-major concat is wrong if we naively append; rebuild by time.
  // For stride-16, each block of 128 → 8 frames. Accumulate per-channel.
  std::vector<std::vector<float>> ch(kNBand);
  for (int start = 0; start + block <= static_cast<int>(T); start += block) {
    auto b = pqmf.analysis(x.data() + start, block);
    const int tb = static_cast<int>(b.size() / kNBand);
    for (int c = 0; c < kNBand; ++c)
      for (int t = 0; t < tb; ++t) ch[c].push_back(b[c * tb + t]);
  }
  const int T_bs = static_cast<int>(ch[0].size());
  bands_stream.resize(static_cast<size_t>(kNBand * T_bs));
  for (int c = 0; c < kNBand; ++c)
    for (int t = 0; t < T_bs; ++t)
      bands_stream[static_cast<size_t>(c * T_bs + t)] = ch[c][static_cast<size_t>(t)];

  pqmf.reset();
  auto bands_full = pqmf.analysis(x.data(), (static_cast<int>(T) / block) * block);
  const int T_bf = static_cast<int>(bands_full.size() / kNBand);
  const int T_cmp = std::min(T_bs, T_bf);
  double max_s = 0;
  for (int c = 0; c < kNBand; ++c)
    for (int t = 0; t < T_cmp; ++t)
      max_s = std::max(
          max_s, std::abs(double(bands_stream[c * T_bs + t]) -
                          double(bands_full[c * T_bf + t])));
  std::cout << "stream" << block << "_vs_full analysis max_abs=" << max_s << "\n";

  // Synthesis smoke: inverse of ref bands vs torch-length
  pqmf.reset();
  auto y = pqmf.synthesis(ref_bands.data(), T_b_ref);
  std::cout << "synthesis out_len=" << y.size() << "\n";

  const bool ok = max_abs < 1e-4 && max_s < 1e-4;
  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
