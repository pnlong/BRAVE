"""Tests for canonicalizer gin setup (two-phase parse like train_canonicalizer.py)."""

import os
from pathlib import Path

import gin
import pytest
import torch

import rave
import rave.canonicalizer.callbacks  # noqa: F401
import rave.canonicalizer.in_domain_discriminator  # noqa: F401
import rave.canonicalizer.ir_augmentation  # noqa: F401
import rave.canonicalizer.trainer  # noqa: F401
import rave.canonicalizer.waveform_canonicalizer  # noqa: F401
import rave.canonicalizer.latent_canonicalizer  # noqa: F401
from rave import discriminator, dsp  # noqa: F401
from rave.canonicalizer.gin_setup import (
    build_in_domain_discriminator,
    configure_backbone_gin,
    configure_canonicalizer_gin,
    validate_canonicalizer_gin,
)
from rave.canonicalizer.latent_canonicalizer import LatentCanonicalizer
from rave.canonicalizer.trainer import CanonicalizerTrainer

_BRAVE = Path(__file__).resolve().parents[2]


def test_two_phase_gin_builds_discriminator_and_trainer():
    configure_backbone_gin(_BRAVE / "configs/brave_birdsong.gin", 1)
    model = rave.RAVE(n_channels=1)
    warp = LatentCanonicalizer(latent_size=128)

    configure_canonicalizer_gin(_BRAVE / "configs/brave_canonicalizer.gin", 1)
    disc = build_in_domain_discriminator(1)
    trainer = CanonicalizerTrainer(
        backbone=model,
        warp=warp,
        canonicalizer_type="latent",
        in_domain_disc=disc,
    )

    assert sum(p.numel() for p in disc.parameters()) > 100_000
    assert trainer.lambda_gan == 1.0
    assert trainer.lambda_rec == 1.0
    assert trainer.recon_ood_mode == "both"
    assert trainer.warmup == 2000
    assert len(disc(torch.randn(1, 1, 4096))) == 1


def test_wrong_canonicalizer_gin_fails_with_helpful_error():
    configure_backbone_gin(_BRAVE / "configs/brave_birdsong.gin", 1)

    gin.clear_config()
    prev = os.getcwd()
    os.chdir(_BRAVE / "configs")
    try:
        gin.parse_config_file("brave_fader_texture.gin")
    finally:
        os.chdir(prev)

    with pytest.raises(RuntimeError, match="missing bindings"):
        validate_canonicalizer_gin(canon_cfg=Path("brave_fader_texture.gin"))
