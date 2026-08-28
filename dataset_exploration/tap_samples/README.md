# Tap samples (mic class)

Canonical filenames **start with the mic, then a dash**:

| Prefix | Class |
|--------|--------|
| `Shotgun-` | shotgun |
| `Contact Mic L-` / `Contact Mic R-` | contact |
| `sm57-` | sm57 |

Rewrite song-first contact takes and unlabeled `0.wav` / `1.wav`, then restage shotgun:

```bash
python scripts/stage_tap_mic_subset.py \
  --audio-dir $BRAVE_STORAGE/tap_samples/audio_subset \
  --rename \
  --stage-mic shotgun
```

Writes ``mic_type.yaml`` and (with ``--stage-mic shotgun``) symlinks ``audio_subset_shotgun/``.

**Held-out listening set** — last **20 s** of one take per gesture (`drumroll`, `heavy`, `light`, `scraping`, `sidetap`, `softshoe`, `tapping`). The prefix of that take stays in train, so Lightning val (random ~2% of LMDB chunks) is unchanged and you do not dump whole 3–4 min takes into eval.

```bash
python scripts/stage_tap_mic_subset.py \
  --audio-dir $BRAVE_STORAGE/tap_samples/audio_subset \
  --stage-mic shotgun --holdout-eval
```

`--eval-seconds 0` holds out the whole take instead. Train preprocess: `INPUT_PATH=$BRAVE_STORAGE/tap_samples/audio_subset_shotgun`. Listen / `generate.py --input`: `$BRAVE_STORAGE/tap_samples/audio_subset_shotgun_eval`.


