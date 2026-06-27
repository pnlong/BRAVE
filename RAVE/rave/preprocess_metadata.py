"""Helpers for RAVE preprocess LMDB metadata.yaml."""

from __future__ import annotations

import os

import yaml


def read_stored_sec_from_metadata(db_path: str) -> float:
    meta_path = os.path.join(db_path, 'metadata.yaml')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'metadata.yaml not found under {db_path}')
    with open(meta_path, 'r') as metadata:
        ref = yaml.safe_load(metadata)
    if ref is None or 'n_seconds' not in ref:
        raise KeyError(f'metadata.yaml under {db_path} missing n_seconds')
    return float(ref['n_seconds'])
