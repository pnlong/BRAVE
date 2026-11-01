#pragma once

/**
 * Stage 7 — Fader / conditional attribute controls (knob-only, attr_mode=1).
 *
 * Host supplies RAW training-unit values; the engine normalizes to [-1, 1],
 * broadcasts over T_lat and concatenates with the 128-D content latent
 * before decode (matches RAVE/rave/fader/export/trace_model.py normalize_all).
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "brave_rt/json_util.hpp"

namespace brave_rt {

enum class AttrKind { Continuous, Discrete };

struct AttrSpec {
  std::string name, label, control;  // control: "knob" | "menu"
  AttrKind kind = AttrKind::Continuous;
  bool applies_to_output = true;
  float min_v = 0, max_v = 1, default_v = 0;
  int num_classes = 0;
  std::vector<std::string> class_labels;
};

struct HostControls {
  int attr_mode_default = 1;
  std::string domain_y_display;  // empty if none
  std::string domain_x_display;
  std::vector<AttrSpec> attributes;  // full schema order (== decoder D rows)
};

/** Raw training-unit value → normalized decoder input (continuous is unclamped, as in training). */
inline float normalize_raw(const AttrSpec& s, float raw) {
  if (s.kind == AttrKind::Discrete) {
    if (s.num_classes <= 1) return 0.f;
    int idx = static_cast<int>(raw);  // trunc toward zero, like tensor.long()
    idx = std::min(std::max(idx, 0), s.num_classes - 1);
    return 2.f * (static_cast<float>(idx) / static_cast<float>(s.num_classes - 1)) - 1.f;
  }
  return 2.f * ((raw - s.min_v) / (s.max_v - s.min_v + 1e-8f) - 0.5f);
}

/**
 * raw_D (D) → out_D_x_T channel-major (D, T): out[d*T + t] = norm(raw[d]).
 * Rows beyond hc.attributes.size() (no schema) pass through clamped to [-1,1].
 */
inline void normalize_broadcast(const HostControls& hc, const float* raw_D, int D,
                                int T_lat, float* out_D_x_T) {
  for (int d = 0; d < D; ++d) {
    float v;
    if (static_cast<size_t>(d) < hc.attributes.size())
      v = normalize_raw(hc.attributes[static_cast<size_t>(d)], raw_D[d]);
    else
      v = std::min(std::max(raw_D[d], -1.f), 1.f);
    float* row = out_D_x_T + static_cast<size_t>(d) * T_lat;
    std::fill(row, row + T_lat, v);
  }
}

/** Default raw value for a spec (midpoint for continuous when no default given). */
inline float attr_default_raw(const AttrSpec& s) { return s.default_v; }

namespace attr_detail {

inline size_t skip_ws(const std::string& s, size_t i) {
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  return i;
}

/** s[i]=='"' → index just past closing quote. */
inline size_t skip_string(const std::string& s, size_t i) {
  ++i;
  while (i < s.size()) {
    if (s[i] == '\\') {
      i += 2;
      continue;
    }
    if (s[i] == '"') return i + 1;
    ++i;
  }
  return s.size();
}

/** Index just past the JSON value starting at s[i]. */
inline size_t skip_value(const std::string& s, size_t i) {
  if (i >= s.size()) return i;
  if (s[i] == '"') return skip_string(s, i);
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    while (i < s.size()) {
      const char c = s[i];
      if (c == '"') {
        i = skip_string(s, i);
        continue;
      }
      if (c == '{' || c == '[') ++depth;
      if (c == '}' || c == ']') {
        if (--depth == 0) return i + 1;
      }
      ++i;
    }
    return s.size();
  }
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
         !std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  return i;
}

/** Raw value text of a direct member `key` of object `obj` ('{...}'). */
inline bool find_member(const std::string& obj, const std::string& key,
                        std::string& out) {
  size_t i = skip_ws(obj, 0);
  if (i >= obj.size() || obj[i] != '{') return false;
  ++i;
  while (i < obj.size()) {
    i = skip_ws(obj, i);
    if (i >= obj.size() || obj[i] == '}') break;
    if (obj[i] == ',') {
      ++i;
      continue;
    }
    if (obj[i] != '"') return false;
    const size_t kend = skip_string(obj, i);
    const std::string k = json_unescape(obj.substr(i + 1, kend - i - 2));
    i = skip_ws(obj, kend);
    if (i >= obj.size() || obj[i] != ':') return false;
    i = skip_ws(obj, i + 1);
    const size_t vend = skip_value(obj, i);
    if (k == key) {
      out = obj.substr(i, vend - i);
      return true;
    }
    i = vend;
  }
  return false;
}

inline bool is_object(const std::string& v) {
  const size_t i = skip_ws(v, 0);
  return i < v.size() && v[i] == '{';
}

inline bool value_to_string(const std::string& v, std::string& out) {
  const size_t i = skip_ws(v, 0);
  if (i >= v.size() || v[i] != '"') return false;
  const size_t e = skip_string(v, i);
  out = json_unescape(v.substr(i + 1, e - i - 2));
  return true;
}

inline bool value_to_double(const std::string& v, double& out) {
  const size_t i = skip_ws(v, 0);
  if (i >= v.size()) return false;
  char* end = nullptr;
  const double d = std::strtod(v.c_str() + i, &end);
  if (end == v.c_str() + i) return false;
  out = d;
  return true;
}

inline bool value_to_bool(const std::string& v, bool& out) {
  const size_t i = skip_ws(v, 0);
  if (v.compare(i, 4, "true") == 0) { out = true; return true; }
  if (v.compare(i, 5, "false") == 0) { out = false; return true; }
  return false;
}

/** Elements (raw text) of a JSON array value. */
inline std::vector<std::string> array_elements(const std::string& v) {
  std::vector<std::string> els;
  size_t i = skip_ws(v, 0);
  if (i >= v.size() || v[i] != '[') return els;
  ++i;
  while (i < v.size()) {
    i = skip_ws(v, i);
    if (i >= v.size() || v[i] == ']') break;
    if (v[i] == ',') {
      ++i;
      continue;
    }
    const size_t e = skip_value(v, i);
    els.push_back(v.substr(i, e - i));
    i = e;
  }
  return els;
}

inline std::string member_string(const std::string& obj, const std::string& key,
                                 const std::string& fallback = "") {
  std::string v, s;
  if (find_member(obj, key, v) && value_to_string(v, s)) return s;
  return fallback;
}

inline double member_double(const std::string& obj, const std::string& key,
                            double fallback, bool* found = nullptr) {
  std::string v;
  double d;
  if (find_member(obj, key, v) && value_to_double(v, d)) {
    if (found) *found = true;
    return d;
  }
  if (found) *found = false;
  return fallback;
}

inline bool member_bool(const std::string& obj, const std::string& key,
                        bool fallback) {
  std::string v;
  bool b;
  if (find_member(obj, key, v) && value_to_bool(v, b)) return b;
  return fallback;
}

/** Display name of a domain entry ({"display_name": ...} or plain string). */
inline std::string domain_display(const std::string& domains_obj,
                                  const std::string& which) {
  std::string v;
  if (!find_member(domains_obj, which, v)) return "";
  std::string s;
  if (value_to_string(v, s)) return s;
  if (is_object(v)) {
    s = member_string(v, "display_name", "");
    if (s.empty()) s = member_string(v, "name", "");
    if (s.empty()) s = member_string(v, "id", "");
  }
  return s;
}

inline AttrSpec parse_attr(const std::string& obj) {
  AttrSpec a;
  a.name = member_string(obj, "name");
  a.label = member_string(obj, "label", a.name);
  a.control = member_string(obj, "control", "");
  a.applies_to_output = member_bool(obj, "applies_to_output", true);

  std::vector<std::string> labels;
  std::string lv;
  if (find_member(obj, "class_labels", lv) ||
      find_member(obj, "discrete_class_labels", lv)) {
    for (const auto& e : array_elements(lv)) {
      std::string s;
      labels.push_back(value_to_string(e, s) ? s : e);
    }
  }
  a.class_labels = labels;

  const std::string kind = member_string(obj, "kind", "");
  const bool menu_ctrl = a.control == "menu" || a.control == "choice";
  if (kind == "discrete")
    a.kind = AttrKind::Discrete;
  else if (kind == "continuous")
    a.kind = AttrKind::Continuous;
  else
    a.kind = (menu_ctrl || !labels.empty() || member_double(obj, "num_classes", 0) > 0)
                 ? AttrKind::Discrete
                 : AttrKind::Continuous;
  if (a.control.empty()) a.control = a.kind == AttrKind::Discrete ? "menu" : "knob";
  if (a.control == "choice") a.control = "menu";

  bool f_min = false, f_max = false, f_def = false;
  a.min_v = static_cast<float>(member_double(obj, "min", 0.0, &f_min));
  a.max_v = static_cast<float>(member_double(obj, "max", 1.0, &f_max));
  if (!f_min) a.min_v = static_cast<float>(member_double(obj, "lo", a.min_v, &f_min));
  if (!f_max) a.max_v = static_cast<float>(member_double(obj, "hi", a.max_v, &f_max));
  const int nc = static_cast<int>(member_double(obj, "num_classes", 0));
  a.num_classes = nc > 0 ? nc : static_cast<int>(labels.size());
  a.default_v = static_cast<float>(member_double(obj, "default", 0.0, &f_def));
  if (!f_def)
    a.default_v = a.kind == AttrKind::Continuous ? 0.5f * (a.min_v + a.max_v) : 0.f;
  return a;
}

}  // namespace attr_detail

/**
 * Pragmatic parser for the STAGE_07 host_controls schema. `json` may be the
 * whole brave_rt.json (host_controls nested) or a standalone host_controls.json.
 * Missing host_controls → empty attributes.
 */
inline HostControls parse_host_controls_json(const std::string& json) {
  using namespace attr_detail;
  HostControls hc;
  std::string root = json;
  std::string nested;
  if (find_member(json, "host_controls", nested)) {
    if (!is_object(nested)) return hc;  // null / invalid → empty
    root = nested;
  } else if (!is_object(json)) {
    return hc;
  }

  hc.attr_mode_default =
      static_cast<int>(member_double(root, "attr_mode_default", 1.0));

  std::string dom;
  if (find_member(root, "domains", dom) && is_object(dom)) {
    hc.domain_y_display = domain_display(dom, "y");
    hc.domain_x_display = domain_display(dom, "x");
  }
  // Legacy exporter field: {"x": "Tap", "y": "Brook"}
  if (hc.domain_y_display.empty() && hc.domain_x_display.empty() &&
      find_member(root, "domain_display", dom) && is_object(dom)) {
    hc.domain_y_display = domain_display(dom, "y");
    hc.domain_x_display = domain_display(dom, "x");
  }

  std::string attrs;
  if (find_member(root, "attributes", attrs)) {
    for (const auto& el : array_elements(attrs))
      if (is_object(el)) hc.attributes.push_back(parse_attr(el));
  }
  return hc;
}

/** brave_rt.json (nested host_controls) first, else host_controls.json. */
inline HostControls load_host_controls_from_bundle(const std::string& bundle_dir) {
  auto try_file = [](const std::string& path, HostControls& out) {
    std::ifstream f(path);
    if (!f) return false;
    out = parse_host_controls_json(read_file_text(path));
    return true;
  };
  HostControls hc;
  bool have = try_file(bundle_dir + "/brave_rt.json", hc);
  if (!have || hc.attributes.empty()) {
    HostControls side;
    if (try_file(bundle_dir + "/host_controls.json", side) &&
        !side.attributes.empty())
      hc = side;
  }
  return hc;
}

/**
 * Reconcile a schema with the decoder's attr channel count D. Empty schema →
 * synthetic pass-through specs (continuous [-1,1], default 0). Non-empty
 * schema must have exactly D rows when D>0; D==0 drops the schema rows.
 */
inline void reconcile_host_controls(HostControls& hc, int D) {
  if (D <= 0) {
    hc.attributes.clear();
    return;
  }
  if (hc.attributes.empty()) {
    for (int i = 0; i < D; ++i) {
      AttrSpec a;
      a.name = "attr" + std::to_string(i);
      a.label = a.name;
      a.control = "knob";
      a.min_v = -1.f;
      a.max_v = 1.f;
      a.default_v = 0.f;
      hc.attributes.push_back(a);
    }
    return;
  }
  if (static_cast<int>(hc.attributes.size()) != D)
    throw std::runtime_error(
        "host_controls has " + std::to_string(hc.attributes.size()) +
        " attributes but decoder expects D=" + std::to_string(D));
}

}  // namespace brave_rt
