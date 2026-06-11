"""
Gin-driven training hooks for ``RAVE/scripts/train.py``.

Base BRAVE (``configs/brave.gin``) leaves ``fader=False``. ``configs/brave_fader.gin``
includes ``brave.gin`` and sets ``fader=True`` (single ``--config`` for Fader training).
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import gin
import pytorch_lightning as pl
from torch.utils import data


@gin.configurable
def build_training_model(n_channels: int = 0, fader: bool = False):
    if fader:
        from .fader.model import FaderRAVE

        return FaderRAVE(n_channels=n_channels)
    from .model import RAVE

    return RAVE(n_channels=n_channels)


@gin.configurable
def wrap_training_datasets(
    train: data.Dataset,
    val: data.Dataset,
    *,
    sampling_rate: int,
    n_signal: int,
    db_path: str,
    fader: bool = False,
) -> Tuple[data.Dataset, data.Dataset]:
    if not fader:
        return train, val

    from .fader.dataset import wrap_fader_dataset

    train = wrap_fader_dataset(
        train,
        sampling_rate=sampling_rate,
        n_signal=n_signal,
        db_path=db_path,
    )
    val = wrap_fader_dataset(
        val,
        sampling_rate=sampling_rate,
        n_signal=n_signal,
        db_path=db_path,
    )
    return train, val


@gin.configurable
def finalize_training_model(
    model: pl.LightningModule,
    db_path: str,
    n_signal: int,
    smoke_test: bool = False,
    fader: bool = False,
    rave_root: str | None = None,
) -> None:
    if not fader:
        return

    from .fader.attributes import (
        load_attribute_stats,
        resolve_stats_path,
        validate_discrete_sidecar,
        validate_stats_against_config,
    )
    from .fader.model import FaderRAVE

    if not isinstance(model, FaderRAVE):
        return

    stats_path = resolve_stats_path(db_path)
    if stats_path is None and smoke_test:
        print(
            "attribute_stats.yaml missing; running quick precompute for smoke_test..."
        )
        import subprocess

        root = rave_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        precompute_script = os.path.join(root, "scripts", "precompute_descriptors.py")
        subprocess.run(
            [
                sys.executable,
                precompute_script,
                f"--db_path={db_path}",
                f"--n_signal={n_signal}",
                "--max_chunks=4",
            ],
            check=True,
        )
        stats_path = resolve_stats_path(db_path)
    if stats_path is None:
        raise FileNotFoundError(
            f"Missing attribute_stats.yaml in {db_path}. Run:\n"
            f"  python RAVE/scripts/precompute_descriptors.py "
            f"--db_path {db_path} --config configs/brave_fader_*.gin --n_signal {n_signal}"
        )
    model.load_attribute_stats_from_file(stats_path)
    st = load_attribute_stats(stats_path)
    validate_stats_against_config(
        st,
        model.continuous_attributes,
        model.discrete_attributes,
        n_signal=n_signal,
    )
    validate_discrete_sidecar(
        db_path,
        model.attribute_names,
        model.discrete_attributes,
        model.num_classes_per_attribute,
    )
    split_info = ""
    if st.get("split"):
        split_info = f" split={st['split']}"
    print(f"Loaded {stats_path} (version={st.get('version', '?')}{split_info})")


@gin.configurable
def extra_training_callbacks(fader: bool = False) -> Sequence[pl.Callback]:
    if not fader:
        return []
    from .fader.callbacks import LambdaWarmupCallback

    return [LambdaWarmupCallback()]
