#pragma once

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brave_rt {

/** Tiny helpers for brave_rt.json (no full JSON parser). */
inline std::string read_file_text(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

/** Decode JSON string escapes including \\uXXXX (BMP only → UTF-8). */
inline std::string json_unescape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 >= s.size()) {
      out.push_back(s[i]);
      continue;
    }
    const char e = s[++i];
    switch (e) {
      case '"':
      case '\\':
      case '/':
        out.push_back(e);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u':
        if (i + 4 < s.size()) {
          unsigned code = 0;
          bool ok = true;
          for (int k = 1; k <= 4; ++k) {
            const char c = s[i + static_cast<size_t>(k)];
            code <<= 4;
            if (c >= '0' && c <= '9')
              code |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f')
              code |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
              code |= static_cast<unsigned>(c - 'A' + 10);
            else {
              ok = false;
              break;
            }
          }
          if (ok) {
            i += 4;
            if (code < 0x80) {
              out.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (code >> 6)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (code >> 12)));
              out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
        }
        out.push_back('\\');
        out.push_back('u');
        break;
      default:
        out.push_back('\\');
        out.push_back(e);
        break;
    }
  }
  return out;
}

inline std::string json_string_field(const std::string& json,
                                     const std::string& key,
                                     const std::string& fallback = "") {
  const std::string pat = "\"" + key + "\"";
  auto pos = json.find(pat);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + pat.size());
  if (pos == std::string::npos) return fallback;
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos) return fallback;
  // Scan to closing quote, honoring backslash escapes.
  size_t end = pos + 1;
  while (end < json.size()) {
    if (json[end] == '\\' && end + 1 < json.size()) {
      end += 2;
      continue;
    }
    if (json[end] == '"') break;
    ++end;
  }
  if (end >= json.size()) return fallback;
  return json_unescape(json.substr(pos + 1, end - pos - 1));
}

inline bool json_bool_field(const std::string& json, const std::string& key,
                            bool fallback = false) {
  const std::string pat = "\"" + key + "\"";
  auto pos = json.find(pat);
  if (pos == std::string::npos) return fallback;
  pos = json.find(':', pos + pat.size());
  if (pos == std::string::npos) return fallback;
  while (pos + 1 < json.size() &&
         (json[pos + 1] == ' ' || json[pos + 1] == '\t' || json[pos + 1] == '\n'))
    ++pos;
  if (json.compare(pos + 1, 4, "true") == 0) return true;
  if (json.compare(pos + 1, 5, "false") == 0) return false;
  return fallback;
}

}  // namespace brave_rt
