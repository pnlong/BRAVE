# Stage 6 — Polish and BRAVE integration

**Goal:** First-class `--host brave_rt` in BRAVE export, CI smoke, user docs.

**Depends on:** Stages 2–5 as applicable (vanilla + cyclegan + sidechain).

## In scope

1. Wire `scripts/export_model.py --host brave_rt` (vanilla + cyclegan) calling `brave_rt/python/export_brave_rt.py`.
2. CI: engine unit tests + parity smoke (Linux OK; AU may be mac-only).
3. Update root [`CLAUDE.md`](../../CLAUDE.md) Key Commands; hosts table in scratchpaper.
4. Example bundle README: install VST3/AU, DAW buffer 128, 44.1 kHz.
5. Optional: Y→X using swapped weights (`export_cyclegan_yx` pattern).

## Out of scope

Fader knobs (Stage 7), attribute extract (Stage 8), Max external, Windows.

## Exit criteria

- [ ] One-command export to a BraveRT bundle from BRAVE repo root
- [ ] CI green for engine tests
- [ ] CLAUDE.md / scratchpaper point at `brave_rt/`

## Suggested agent prompt

> Implement only Stage 6 per `brave_rt/docs/STAGE_06_polish.md`.
