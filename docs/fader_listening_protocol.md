# Fader attribute swap — listening protocol

Generate paired WAVs with [`RAVE/scripts/generate_attribute_swap_pairs.py`](../RAVE/scripts/generate_attribute_swap_pairs.py), then run a short subjective check.

## Stimuli per source clip

| File | Description |
|------|-------------|
| `*_original.wav` | Reconstruction with matched attributes |
| `*_swapped.wav` | Same `z`, attributes permuted with another clip in the batch |
| `*_self_swap.wav` | Sanity: swap with self (should match original) |

## Suggested procedure

1. Use headphones; match playback level across pairs (LUFS-normalize externally if needed).
2. **ABX**: Is `swapped` closer to the donor clip’s timbre/energy profile than `original`?
3. **Leakage**: On `original`, attributes should sound consistent with the source; `z` should not encode obvious tag/scene information you intended to move to `water_scene`.
4. Note failures: metallic artifacts, unchanged swap, or scene class audible in `z` only.

## Objective metrics

See [`RAVE/scripts/eval_fader_attributes.py`](../RAVE/scripts/eval_fader_attributes.py) for batch L1/cosine on continuous re-extraction after swap.

Automated MOS is out of scope for this repo.
