"""Tests for canonicalizer domain profile and manifest."""

import json
import tempfile
from pathlib import Path

import torch
import yaml

from rave.canonicalizer.config import (
    CanonicalizerManifest,
    CycleGANManifest,
    descriptor_loss_attributes,
    save_canonicalizer_checkpoint,
    load_canonicalizer_checkpoint,
    validate_manifest,
)


def test_descriptor_loss_attributes_pitched():
    attrs = descriptor_loss_attributes(
        ["f0", "chroma_class", "rms", "warmth", "centroid"])
    assert "f0" not in attrs
    assert "chroma_class" not in attrs
    assert "rms" in attrs
    assert "warmth" in attrs


def test_manifest_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "waveform_canonicalizer.ckpt"
        manifest = CanonicalizerManifest(
            canonicalizer_type="waveform",
            backbone_config="configs/brave_fader_pitched.gin",
            backbone_ckpt="runs/x.ckpt",
            db_path="/data/lmdb",
            use_reverb=True,
            stats_hash="abc",
        )
        save_canonicalizer_checkpoint(path, {"eq.filters.0.gain_db": torch.zeros(1)}, manifest)
        state, loaded = load_canonicalizer_checkpoint(path)
        assert loaded.canonicalizer_type == "waveform"
        assert "eq.filters.0.gain_db" in state
        assert loaded.latent_n_layers == 1


def test_cyclegan_manifest_latent_layers_roundtrip():
    data = CycleGANManifest(
        canonicalizer_type="latent",
        backbone_x_config="/x.gin",
        backbone_x_ckpt="/x.ckpt",
        backbone_y_config="/y.gin",
        backbone_y_ckpt="/y.ckpt",
        db_path_x="/x",
        db_path_y="/y",
        latent_n_layers=2,
        latent_hidden_size=128,
    ).to_dict()
    loaded = CycleGANManifest.from_dict(data)
    assert loaded.latent_n_layers == 2
    assert loaded.latent_hidden_size == 128
    legacy = CycleGANManifest.from_dict({
        "canonicalizer_type": "latent",
        "backbone_x_config": "/x.gin",
        "backbone_x_ckpt": "/x.ckpt",
        "backbone_y_config": "/y.gin",
        "backbone_y_ckpt": "/y.ckpt",
        "db_path_x": "/x",
        "db_path_y": "/y",
    })
    assert legacy.latent_n_layers == 1
    assert legacy.latent_hidden_size is None


def test_validate_manifest_warns_on_mismatch():
    m = CanonicalizerManifest(
        canonicalizer_type="latent",
        backbone_config="/a.gin",
        backbone_ckpt="/a.ckpt",
        db_path="/a",
    )
    try:
        validate_manifest(
            m,
            config_path="/b.gin",
            ckpt_path="/b.ckpt",
            db_path="/b",
            strict=True,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
