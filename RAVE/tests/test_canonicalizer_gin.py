"""Gin smoke test for brave_canonicalizer.gin."""

import os
from pathlib import Path

import gin

_BRAVE = Path(__file__).resolve().parents[2]


def test_brave_canonicalizer_gin_parses():
    gin.clear_config()
    cfg_dir = _BRAVE / "configs"
    prev = os.getcwd()
    os.chdir(cfg_dir)
    try:
        gin.parse_config_file("brave_fader_pitched.gin")
        gin.parse_config_file("brave_canonicalizer.gin")
    finally:
        os.chdir(prev)
    cfg = gin.config_str()
    assert "WaveformCanonicalizer" in cfg
    assert "LatentCanonicalizer" in cfg
    assert "InDomainAudioDiscriminator" in cfg
