// Fader / conditional BraveRT bundle runner (Stage 7, attr_mode=1 knobs only).
//
//   fader_parity_cli <bundle_dir> --in x.wav|x.npy --out y.wav|y.npy
//       [--attr name=value]... [--block 512] [--threads 1] [--list]
//       [--attr-mode 1|2|0]   (Stage 8: 1 manual default, 2 extract, 0 extract+override)
//
// --attr values are RAW training units (discrete: class index or class label).
// name may also be a numeric schema index. Unlisted attrs keep bundle defaults.

#include "brave_rt/brave_engine.hpp"
#include "brave_rt/npy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

uint32_t rd32(const unsigned char* p) {
  return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t rd16(const unsigned char* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

/** Minimal RIFF/WAVE reader: PCM 16/24/32 or float32; first channel only. */
std::vector<float> read_wav(const std::string& path, int& sr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
  if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4))
    throw std::runtime_error("not a RIFF/WAVE file: " + path);
  int fmt_tag = 0, ch = 1, bits = 16;
  sr = 44100;
  size_t i = 12;
  while (i + 8 <= b.size()) {
    const uint32_t sz = rd32(&b[i + 4]);
    const unsigned char* d = &b[i + 8];
    if (!std::memcmp(&b[i], "fmt ", 4)) {
      fmt_tag = rd16(d);
      ch = rd16(d + 2);
      sr = static_cast<int>(rd32(d + 4));
      bits = rd16(d + 14);
      if (fmt_tag == 0xFFFE && sz >= 26) fmt_tag = rd16(d + 24);
    } else if (!std::memcmp(&b[i], "data", 4)) {
      const size_t avail = std::min<size_t>(sz, b.size() - (i + 8));
      const size_t bps = static_cast<size_t>(bits / 8) * ch;
      const size_t n = bps ? avail / bps : 0;
      std::vector<float> x(n);
      for (size_t t = 0; t < n; ++t) {
        const unsigned char* p = d + t * bps;
        float v = 0.f;
        if (fmt_tag == 3 && bits == 32) {
          uint32_t u = rd32(p);
          std::memcpy(&v, &u, 4);
        } else if (bits == 16) {
          v = static_cast<int16_t>(rd16(p)) / 32768.f;
        } else if (bits == 24) {
          int32_t s = (p[0] << 8) | (p[1] << 16) | (p[2] << 24);
          v = static_cast<float>(s >> 8) / 8388608.f;
        } else if (bits == 32) {
          v = static_cast<int32_t>(rd32(p)) / 2147483648.f;
        } else {
          throw std::runtime_error("unsupported WAV format");
        }
        x[t] = v;
      }
      return x;
    }
    i += 8 + sz + (sz & 1);
  }
  throw std::runtime_error("no data chunk in " + path);
}

void write_wav(const std::string& path, const std::vector<float>& y, int sr) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  const uint32_t bytes = static_cast<uint32_t>(y.size() * 4);
  f.write("RIFF", 4);
  w32(36 + bytes);
  f.write("WAVEfmt ", 8);
  w32(16);
  w16(3);  // IEEE float
  w16(1);
  w32(static_cast<uint32_t>(sr));
  w32(static_cast<uint32_t>(sr) * 4);
  w16(4);
  w16(32);
  f.write("data", 4);
  w32(bytes);
  f.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(bytes));
}

void write_npy(const std::string& path, const std::vector<float>& y) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                  std::to_string(y.size()) + ",), }";
  while ((10 + h.size() + 1) % 64) h.push_back(' ');
  h.push_back('\n');
  f.write("\x93NUMPY\x01\x00", 8);
  const uint16_t hl = static_cast<uint16_t>(h.size());
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  f.write(reinterpret_cast<const char*>(y.data()),
          static_cast<std::streamsize>(y.size() * 4));
}

void print_schema(const brave_rt::BraveEngine& eng) {
  const auto& hc = eng.host_controls();
  std::cout << "D=" << eng.num_attributes() << " attr_mode_default="
            << hc.attr_mode_default << " domain_y='" << hc.domain_y_display
            << "' domain_x='" << hc.domain_x_display << "'\n";
  for (int i = 0; i < eng.num_attributes(); ++i) {
    const auto& a = hc.attributes[static_cast<size_t>(i)];
    std::cout << "  [" << i << "] " << a.name << " " << a.control << " "
              << (a.kind == brave_rt::AttrKind::Discrete ? "discrete" : "continuous")
              << " out=" << a.applies_to_output << " min=" << a.min_v
              << " max=" << a.max_v << " default=" << a.default_v
              << " K=" << a.num_classes << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: fader_parity_cli <bundle_dir> --in x.wav|x.npy "
                 "--out y.wav|y.npy [--attr name=value]... [--block 512] "
                 "[--threads 1] [--list]\n";
    return 2;
  }
  const std::string bundle = argv[1];
  std::string in_path, out_path;
  std::vector<std::pair<std::string, std::string>> attrs;
  int block = 512, threads = 1, attr_mode = 1;
  bool list_only = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--in") in_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--block") block = std::atoi(next().c_str());
    else if (a == "--attr-mode") attr_mode = std::atoi(next().c_str());
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--list") list_only = true;
    else if (a == "--attr") {
      const std::string kv = next();
      const auto eq = kv.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--attr expects name=value\n";
        return 2;
      }
      attrs.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    } else {
      std::cerr << "unknown flag " << a << "\n";
      return 2;
    }
  }

  try {
    brave_rt::BraveEngine eng;
    eng.load(bundle);
    if (threads > 1) eng.set_num_threads(threads);
    eng.set_attr_mode(attr_mode);
    print_schema(eng);

    for (const auto& [name, val] : attrs) {
      int idx = eng.attr_index(name);
      if (idx < 0 && !name.empty() &&
          name.find_first_not_of("0123456789") == std::string::npos)
        idx = std::atoi(name.c_str());
      if (idx < 0 || idx >= eng.num_attributes()) {
        std::cerr << "unknown attribute '" << name << "' (D=" << eng.num_attributes()
                  << ")\n";
        return 2;
      }
      const auto& spec = eng.host_controls().attributes[static_cast<size_t>(idx)];
      char* end = nullptr;
      float v = std::strtof(val.c_str(), &end);
      if (end == val.c_str() || *end != '\0') {
        const auto it = std::find(spec.class_labels.begin(), spec.class_labels.end(), val);
        if (it == spec.class_labels.end()) {
          std::cerr << "bad value '" << val << "' for " << name << "\n";
          return 2;
        }
        v = static_cast<float>(it - spec.class_labels.begin());
      }
      eng.set_attr_raw(idx, v);
      std::cout << "attr " << spec.name << " raw=" << v << " norm="
                << brave_rt::normalize_raw(spec, v) << "\n";
    }
    if (list_only || in_path.empty()) return 0;
    if (out_path.empty()) {
      std::cerr << "--out required\n";
      return 2;
    }
    if (block < 1) block = 512;

    int sr = brave_rt::kSampleRate;
    std::vector<float> x;
    if (ends_with(in_path, ".npy")) x = brave_rt::load_npy_f32(in_path);
    else x = read_wav(in_path, sr);

    std::vector<float> y(x.size(), 0.f);
    eng.prepare(block, false);
    eng.reset_state();
    for (size_t i = 0; i < x.size(); i += static_cast<size_t>(block)) {
      const int n = static_cast<int>(std::min<size_t>(block, x.size() - i));
      eng.process(x.data() + i, y.data() + i, n);
    }
    // Drop FIFO prime latency so output is time-aligned with input.
    const int lat = eng.latency_samples();
    if (lat > 0 && static_cast<size_t>(lat) < y.size()) {
      std::rotate(y.begin(), y.begin() + lat, y.end());
      std::fill(y.end() - lat, y.end(), 0.f);
    }
    if (ends_with(out_path, ".npy")) write_npy(out_path, y);
    else write_wav(out_path, y, sr);
    std::cout << "wrote " << out_path << " (" << y.size() << " samples, underflows="
              << eng.fifo_underflows() << ")\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
