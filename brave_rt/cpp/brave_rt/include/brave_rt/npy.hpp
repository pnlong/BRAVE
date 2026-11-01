#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brave_rt {

/** Minimal little-endian float32 .npy reader (C-order). */
inline std::vector<float> load_npy_f32(const std::string& path,
                                       std::vector<size_t>* shape_out = nullptr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY")
    throw std::runtime_error("bad npy magic: " + path);
  uint8_t v_major = 0, v_minor = 0;
  f.read(reinterpret_cast<char*>(&v_major), 1);
  f.read(reinterpret_cast<char*>(&v_minor), 1);
  uint16_t header_len = 0;
  if (v_major == 1) {
    f.read(reinterpret_cast<char*>(&header_len), 2);
  } else {
    uint32_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 4);
    header_len = static_cast<uint16_t>(hl);
  }
  std::string header(header_len, '\0');
  f.read(&header[0], header_len);

  // Expect descr '<f4' or '|f4' and fortran_order False
  if (header.find("'fortran_order': False") == std::string::npos &&
      header.find("\"fortran_order\": False") == std::string::npos &&
      header.find("fortran_order': False") == std::string::npos) {
    // still try; most torch/numpy dumps are C-order
  }

  std::vector<size_t> shape;
  auto pos = header.find("(");
  auto end = header.find(")", pos);
  if (pos == std::string::npos || end == std::string::npos)
    throw std::runtime_error("npy shape parse failed: " + path);
  std::string shape_str = header.substr(pos + 1, end - pos - 1);
  std::stringstream ss(shape_str);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.find_first_not_of(" \t") == std::string::npos) continue;
    shape.push_back(static_cast<size_t>(std::stoul(tok)));
  }
  size_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float> data(n);
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (!f) throw std::runtime_error("npy read truncated: " + path);
  if (shape_out) *shape_out = shape;
  return data;
}

}  // namespace brave_rt
