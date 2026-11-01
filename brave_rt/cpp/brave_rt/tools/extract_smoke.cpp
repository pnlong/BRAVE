// Stage 8 smoke: DescriptorExtract sanity (analytic signals) + BraveEngine
// attr_mode semantics on a Fader bundle.
//   extract_smoke [bundle_dir]
#include "brave_rt/brave_engine.hpp"
#include "brave_rt/descriptor_extract.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using brave_rt::BraveEngine;
using brave_rt::DescriptorExtract;

static int fails = 0;
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      ++fails;                                    \
      std::printf("FAIL: " __VA_ARGS__);          \
      std::printf("\n");                          \
    } else {                                      \
      std::printf("ok:   " __VA_ARGS__);          \
      std::printf("\n");                          \
    }                                             \
  } while (0)

static std::vector<float> sine(int n, double f, double a, int sr = 44100) {
  std::vector<float> x(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    x[static_cast<size_t>(i)] = static_cast<float>(a * std::sin(2 * M_PI * f * i / sr));
  return x;
}

static void test_extractor() {
  DescriptorExtract ex(44100);
  const int N = DescriptorExtract::kMaxHistory;
  auto x = sine(N - 1, 1000.0, 0.5);
  ex.push(x.data(), static_cast<int>(x.size()));
  CHECK(!ex.ready(), "not ready at %d samples", N - 1);
  ex.push(x.data(), 1);
  CHECK(ex.ready(), "ready at %d samples", N);
  const float rms = ex.value(DescriptorExtract::kRms);
  const float cen = ex.value(DescriptorExtract::kCentroid);
  const float zcr = ex.value(DescriptorExtract::kZcr);
  CHECK(std::fabs(rms - 0.5 / std::sqrt(2.0)) < 5e-3, "sine rms %.4f ≈ 0.3536", rms);
  CHECK(std::fabs(cen - 1000.0) < 40.0, "sine centroid %.1f Hz ≈ 1000", cen);
  CHECK(std::fabs(zcr - 2.0 * 1000.0 / 44100.0) < 4e-3, "sine zcr %.4f ≈ %.4f", zcr,
        2.0 * 1000.0 / 44100.0);
  CHECK(ex.value(DescriptorExtract::kFlatness) < 0.05f, "sine flatness %.4g low",
        ex.value(DescriptorExtract::kFlatness));

  ex.reset();
  CHECK(!ex.ready() && ex.frames_available() == 0, "reset clears");
  std::mt19937 g(1);
  std::normal_distribution<float> nd(0.f, 0.2f);
  std::vector<float> w(static_cast<size_t>(N));
  for (auto& v : w) v = nd(g);
  ex.push(w.data(), N);
  const float fl = ex.value(DescriptorExtract::kFlatness);
  const float cn = ex.value(DescriptorExtract::kCentroid);
  CHECK(fl > 0.3f, "noise flatness %.3f high", fl);
  CHECK(std::fabs(cn - 11025.f) < 1500.f, "noise centroid %.0f Hz ≈ 11 kHz", cn);
  CHECK(std::fabs(ex.value(DescriptorExtract::kRms) - 0.2f) < 0.01f, "noise rms %.3f ≈ 0.2",
        ex.value(DescriptorExtract::kRms));
  // Chunked push == one-shot push.
  DescriptorExtract a(44100), b(44100);
  a.push(w.data(), N);
  for (int i = 0; i < N; i += 100) b.push(w.data() + i, std::min(100, N - i));
  CHECK(a.value(DescriptorExtract::kCentroid) == b.value(DescriptorExtract::kCentroid),
        "chunk-invariant");
}

static void test_engine(const std::string& bundle) {
  BraveEngine eng;
  eng.load(bundle);
  CHECK(eng.attr_mode() == 1, "default attr_mode == 1");
  const int D = eng.num_attributes();
  std::printf("D=%d\n", D);
  if (D == 0) return;
  const int N = 128 * 1100;  // > 131072 samples
  auto x = sine(N, 700.0, 0.3);
  std::vector<float> y1(static_cast<size_t>(N)), y2(static_cast<size_t>(N)),
      y0(static_cast<size_t>(N));
  const int blk = 128;

  eng.reset_state();
  for (int i = 0; i < N; i += blk) eng.process(x.data() + i, y1.data() + i, blk);

  eng.reset_state();
  eng.set_attr_mode(2);
  const int ci = eng.attr_index("centroid"), ri = eng.attr_index("rms");
  float cen_pre = 0, cen_post = 0;
  for (int i = 0; i < N; i += blk) {
    eng.process(x.data() + i, y2.data() + i, blk);
    if (i + blk == 131072 - 128) cen_pre = eng.attr_effective(ci);
  }
  cen_post = eng.attr_effective(ci);
  const float rms_post = eng.attr_effective(ri);
  CHECK(eng.extract_ready(), "extractor ready after %d samples", N);
  // Before ready, mode 2 must equal mode 1 bit-for-bit.
  size_t pre = 131072 - 1024;
  bool same = true;
  for (size_t i = 0; i < pre; ++i)
    if (y1[i] != y2[i]) { same = false; break; }
  CHECK(same, "mode 2 == mode 1 before history ready");
  CHECK(cen_pre == eng.host_controls().attributes[static_cast<size_t>(ci)].default_v,
        "centroid at default pre-ready (%.1f)", cen_pre);
  CHECK(std::fabs(cen_post - 700.f) < 40.f, "centroid extracted %.1f ≈ 700", cen_post);
  CHECK(std::fabs(rms_post - 0.3f / std::sqrt(2.f)) < 5e-3f, "rms extracted %.4f ≈ 0.212",
        rms_post);
  // Non-extractable / discrete stay at manual defaults.
  for (int i = 0; i < D; ++i)
    if (!eng.attr_extractable(i))
      CHECK(eng.attr_effective(i) == eng.attr_raw(i), "attr %d (%s) stays manual", i,
            eng.host_controls().attributes[static_cast<size_t>(i)].name.c_str());
  // Manual array untouched by extraction.
  CHECK(eng.attr_raw(ci) == eng.host_controls().attributes[static_cast<size_t>(ci)].default_v,
        "manual centroid not overwritten");
  // Post-ready output must differ from manual mode.
  double diff = 0;
  for (size_t i = 131072 + 4096; i < static_cast<size_t>(N); ++i)
    diff += std::fabs(y1[i] - y2[i]);
  CHECK(diff > 1e-3, "mode 2 output differs after ready (L1 %.3g)", diff);

  // Mode 0 with override on extractable attrs == manual after ready.
  eng.reset_state();
  eng.set_attr_mode(0);
  for (int i = 0; i < D; ++i) eng.set_attr_override(i, true);
  for (int i = 0; i < N; i += blk) eng.process(x.data() + i, y0.data() + i, blk);
  bool eq = true;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    if (y0[i] != y1[i]) { eq = false; break; }
  CHECK(eq, "mode 0 + override-all == mode 1");

  // Back to mode 1 → identical to initial run.
  eng.set_attr_mode(1);
  eng.reset_state();
  std::vector<float> y3(static_cast<size_t>(N));
  for (int i = 0; i < N; i += blk) eng.process(x.data() + i, y3.data() + i, blk);
  eq = true;
  for (size_t i = 0; i < static_cast<size_t>(N); ++i)
    if (y3[i] != y1[i]) { eq = false; break; }
  CHECK(eq, "mode 1 repeat bit-identical");
}

int main(int argc, char** argv) {
  test_extractor();
  if (argc > 1) test_engine(argv[1]);
  std::printf(fails ? "extract_smoke FAIL (%d)\n" : "extract_smoke PASS\n", fails);
  return fails ? 1 : 0;
}
