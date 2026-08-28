"""Filename → tap mic class (shotgun / contact / sm57)."""

import pytest

from mic_type import canonical_tap_filename, classify_tap_mic, counts


def test_shotgun_prefix():
    assert classify_tap_mic("Shotgun-tapping_4.unnormalized.wav") == "shotgun"
    assert classify_tap_mic("shotgun-sidetap_1.wav") == "shotgun"
    assert classify_tap_mic("/data/x/Shotgun-drumroll_1.unnormalized.wav") == "shotgun"


def test_contact_mic_l_and_r_prefix():
    assert classify_tap_mic("Contact Mic L-tapping_2.unnormalized.wav") == "contact"
    assert classify_tap_mic("Contact Mic R-cute.unnormalized.wav") == "contact"
    assert classify_tap_mic("Contact Mic L-take the a train.unnormalized.wav") == "contact"


def test_sm57_prefix():
    assert classify_tap_mic("sm57-0.wav") == "sm57"
    assert classify_tap_mic("sm57-1.wav") == "sm57"


def test_unclassified_raises():
    with pytest.raises(ValueError, match="unclassified"):
        classify_tap_mic("0.wav")
    with pytest.raises(ValueError, match="unclassified"):
        classify_tap_mic("cute-Contact Mic L.unnormalized.wav")


def test_canonical_song_first_contact():
    assert (
        canonical_tap_filename("cute-Contact Mic L.unnormalized.wav")
        == "Contact Mic L-cute.unnormalized.wav"
    )
    assert (
        canonical_tap_filename("take the a train-Contact Mic R.unnormalized.wav")
        == "Contact Mic R-take the a train.unnormalized.wav"
    )


def test_canonical_already_prefixed():
    assert (
        canonical_tap_filename("Contact Mic L-tapping_2.unnormalized.wav")
        == "Contact Mic L-tapping_2.unnormalized.wav"
    )
    assert canonical_tap_filename("Shotgun-heavy_1.unnormalized.wav") == (
        "Shotgun-heavy_1.unnormalized.wav"
    )
    assert canonical_tap_filename("sm57-0.wav") == "sm57-0.wav"


def test_canonical_sm57_unlabeled():
    assert canonical_tap_filename("0.wav") == "sm57-0.wav"
    assert canonical_tap_filename("1.wav") == "sm57-1.wav"


def test_counts():
    mapping = {
        "a.wav": "shotgun",
        "b.wav": "contact",
        "c.wav": "sm57",
        "d.wav": "shotgun",
    }
    assert counts(mapping) == {"shotgun": 2, "contact": 1, "sm57": 1}


def test_tap_style_and_holdout():
    from pathlib import Path

    from mic_type import select_eval_holdouts, tap_style

    assert tap_style("Shotgun-drumroll_2.unnormalized.wav") == "drumroll"
    assert tap_style("Shotgun-tapping_8.unnormalized_1.wav") == "tapping"
    names = [
        "Shotgun-drumroll_1.unnormalized.wav",
        "Shotgun-drumroll_3.unnormalized.wav",
        "Shotgun-heavy_1.unnormalized.wav",
        "Shotgun-heavy_4.unnormalized.wav",
        "Shotgun-light_1.unnormalized.wav",
        "Shotgun-sidetap_1.unnormalized.wav",
        "Shotgun-tapping_4.unnormalized.wav",
        "Shotgun-tapping_11.unnormalized.wav",
    ]
    files = [Path(n) for n in names]
    picks = {p.style: p for p in select_eval_holdouts(files)}
    assert picks["drumroll"].path.name == "Shotgun-drumroll_3.unnormalized.wav"
    assert picks["drumroll"].held_out
    assert picks["heavy"].path.name == "Shotgun-heavy_4.unnormalized.wav"
    assert picks["tapping"].path.name == "Shotgun-tapping_11.unnormalized.wav"
    assert picks["light"].held_out is False
    assert picks["sidetap"].held_out is False
    held = select_eval_holdouts(files, hold_singletons=True)
    assert all(p.held_out for p in held)


def test_split_eval_times_keeps_prefix():
    from mic_type import split_eval_times

    train_end, eval_start, eval_dur = split_eval_times(203.0, 20.0)
    assert train_end == 183.0
    assert eval_start == 183.0
    assert eval_dur == 20.0
    # Whole-file holdout
    assert split_eval_times(32.0, 0.0) == (0.0, 0.0, 32.0)
    # Too short to keep a prefix
    assert split_eval_times(12.0, 20.0) == (0.0, 0.0, 12.0)
