# Project page (GitHub Pages)

Static site root for the BRAVE listening / narrative page.

## Enable Pages

In the GitHub repo **Settings → Pages**:

1. Source: **Deploy from a branch**
2. Branch: `main` (or your default)
3. Folder: `/docs`

The site will be served from [`index.html`](index.html). Audio lives under
[`assets/audio/`](assets/audio/) (`*.mp3` + `manifest.json`).

## Regenerate demos

See [`narrative.md`](narrative.md) for the `presentation_phase_demos.py` →
`package_project_page_audio.py` commands. Do not commit WAVs under `docs/`.
