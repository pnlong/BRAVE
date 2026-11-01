# Stage 7 — Conditional / Fader knobs only

**Goal:** Live conditional CycleGAN / FaderRAVE via host sliders/menus (`attr_mode=1`). No attribute extract.

**Depends on:** Stage 4–6 (CycleGAN path + plugin).  
**Blocks:** Stage 8 (optional).

**Not in S0–S6.** Knob-only is enough for live conditional use.

## In scope

- Normalize with `attribute_stats.yaml` (embedded or sidecar in the BraveRT bundle)
- Broadcast attrs to `T_lat`
- Concat / condition as in training (decode and/or conditional warp)
- **Data-driven editor:** rebuild controls from bundle JSON on every successful `loadBundle` (no hard-coded attr list in C++)
- JUCE: one control per **output-domain-relevant** attribute (`applies_to_output: true`)
- Parity vs nn~ with `set_attr_mode(1)`

## Editor built from bundle JSON

On load, read `host_controls` from `brave_rt.json` (or sidecar `model_host_controls.json`). The editor:

1. Clears any previous attr widgets.
2. Shows Y `display_name` (section title) when domains are present.
3. For each attribute with `applies_to_output: true`, adds a **knob** or **menu** from the `control` field.
4. **Unconditional / empty controls:** if `host_controls` is missing, `attributes` is `[]`, or no rows apply to output — show a short status line instead of knobs, e.g.  
   **“No conditioning features in this bundle”**  
   (vanilla BRAVE and unconditional CycleGAN). Do not leave a blank mysterious panel.

Stage 3’s fixed “Load bundle / gain” chrome stays; only the conditioning strip is generated from JSON.

## Bundle UI metadata (export-time; Stage 4+ can foreshadow)

When a conditional / joint-conditional bundle is loaded, the plugin must **not** hard-code control types. Export writes a host-controls block (in `brave_rt.json` and/or `model_host_controls.json`) so the editor builds automatically.

### Domain names (display)

```json
"domains": {
  "x": { "id": "x", "display_name": "Tap" },
  "y": { "id": "y", "display_name": "Babbling brook" }
}
```

Show **Y’s display name** in the editor (title / section header): live X→Y transfer only exposes **output-domain (Y)** controls. X’s name is still stored for clarity (tooltips, future Y→X, dual-panel UIs).

### Per-attribute control descriptor

Aligned with nn~ / `*_host_controls.json` and `attribute_stats.yaml` order:

| Field | Purpose |
|-------|---------|
| `name` | Stats / tensor row key |
| `domain` | `"x"` \| `"y"` \| `"both"` (joint schema) |
| `applies_to_output` | `true` if this row is fed to **Dec_Y / warp→Y** path (what the live plugin needs) |
| `control` | `"knob"` (continuous slider) \| `"menu"` / `"choice"` (discrete class) |
| `label` | Human string for the UI |
| `unit` / `min` / `max` / `default` | Continuous raw training units |
| `class_labels` | Discrete menu entries (`discrete_class_labels`) |

Example sketch:

```json
"host_controls": {
  "attr_mode_default": 1,
  "domains": {
    "x": { "display_name": "Tap" },
    "y": { "display_name": "Babbling brook" }
  },
  "attributes": [
    {
      "name": "rms",
      "domain": "both",
      "applies_to_output": true,
      "control": "knob",
      "label": "RMS",
      "min": 0.0, "max": 0.2, "default": 0.05
    },
    {
      "name": "water_scene",
      "domain": "y",
      "applies_to_output": true,
      "control": "menu",
      "label": "Water scene",
      "class_labels": ["stream", "rain", "..."],
      "default": 0
    },
    {
      "name": "tap_only_attr",
      "domain": "x",
      "applies_to_output": false,
      "control": "knob",
      "label": "Tap-only (hidden in X→Y)"
    }
  ]
}
```

### Joint models — hide X-only attrs

For **joint** conditional foundations (schema \(x \cup y\) + availability masks):

- Attributes that exist only on **X** (or are masked off for Y at decode) must set `applies_to_output: false` (or equivalent).
- Stage 7 UI **hides** those controls for the default X→Y engine; they must not appear as dead knobs.
- Attributes on **Y** or **both** that condition Dec_Y / Y-side FiLM stay visible, under a heading using Y’s `display_name`.

Separate geometry with dual backbones: export still tags each attr with which backbone/domain it belongs to; live X→Y only surfaces Y (and shared) rows.

### Implementation notes

- Prefer generating this block from gin attribute lists + `attribute_stats.yaml` + joint availability masks at `--host brave_rt` export (Stage 4/6 can add the JSON fields even before Stage 7 UI exists).
- Reuse conventions from [`docs/fader_host_controls.md`](../../docs/fader_host_controls.md) (`control` ↔ continuous vs discrete / `live.menu`).
- Unconditional CycleGAN bundles (S4) omit `host_controls` or leave `attributes: []`.

## Out of scope

Attribute extract (Stage 8), Max external, Windows.

## Exit criteria

- [x] Bundle `host_controls` drives editor (knob vs menu; Y display name)
- [x] Unconditional bundle shows “No conditioning features in this bundle” (or equivalent)
- [x] Joint X-only attrs hidden for X→Y (`applies_to_output: false` at export)
- [x] Parity vs mean-encode / attr_mode=1 (water Fader: aligned max-abs 9.5e-7, SNR 118 dB; RTF@128 mean 0.865 quiet / see STATUS)

### JUCE parameter choice (landed)

Fixed max slot map: `attr_0`…`attr_15` as `AudioParameterFloat` in [0,1], remapped to `applies_to_output` rows on each `loadBundle`. Keeps automation / state restore stable across bundle swaps.

## Suggested agent prompt

> Implement only Stage 7 per `brave_rt/docs/STAGE_07_knobs.md`. Knob-only; no extract.
