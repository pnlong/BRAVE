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
    db_path_y: str | None = None,
) -> None:
    if not fader:
        return

    from .fader.attributes import (
        load_attribute_stats,
        merge_attribute_stats,
        resolve_stats_path,
        save_attribute_stats,
        validate_discrete_sidecar,
        validate_stats_against_config,
    )
    from .fader.model import FaderRAVE

    if not isinstance(model, FaderRAVE):
        return

    def _ensure_stats(path: str) -> str:
        stats_path = resolve_stats_path(path)
        if stats_path is None and smoke_test:
            print(
                f"attribute_stats.yaml missing in {path}; "
                "running quick precompute for smoke_test..."
            )
            import subprocess

            root = rave_root or os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            precompute_script = os.path.join(
                root, "scripts", "precompute_descriptors.py")
            subprocess.run(
                [
                    sys.executable,
                    precompute_script,
                    f"--db_path={path}",
                    f"--n_signal={n_signal}",
                    "--max_chunks=4",
                ],
                check=True,
            )
            stats_path = resolve_stats_path(path)
        if stats_path is None:
            raise FileNotFoundError(
                f"Missing attribute_stats.yaml in {path}. Run:\n"
                f"  python RAVE/scripts/precompute_descriptors.py "
                f"--db_path {path} --config configs/brave_fader_*.gin "
                f"--n_signal {n_signal}"
            )
        return stats_path

    stats_path_x = _ensure_stats(db_path)
    st = load_attribute_stats(stats_path_x)
    loaded_msg = stats_path_x

    if db_path_y:
        stats_path_y = _ensure_stats(db_path_y)
        st_y = load_attribute_stats(stats_path_y)
        st = merge_attribute_stats(
            st,
            st_y,
            prefer_names=model.attribute_names,
        )
        merged_path = os.path.join(db_path, "attribute_stats_merged.yaml")
        save_attribute_stats(
            merged_path,
            attribute_names=st["attribute_names"],
            min_max_features=st["min_max_features"],
            bin_values=st["bin_values"],
            nb_bins=st["nb_bins"],
            latent_length=st["latent_length"],
            sr=st["sr"],
            continuous_attributes=st["continuous_attributes"],
            discrete_attributes=st["discrete_attributes"],
            attribute_kinds=st["attribute_kinds"],
            discrete_num_classes=st.get("discrete_num_classes"),
            discrete_class_labels=st.get("discrete_class_labels"),
            version=st.get("version", 1),
            split=st.get("split"),
        )
        model._apply_stats(st)
        loaded_msg = (
            f"merged({stats_path_x} + {stats_path_y} → {merged_path})"
        )
        validate_discrete_sidecar(
            db_path_y,
            model.attribute_names,
            model.discrete_attributes,
            model.num_classes_per_attribute,
        )
    else:
        model.load_attribute_stats_from_file(stats_path_x)

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
    print(f"Loaded {loaded_msg} (version={st.get('version', '?')}{split_info})")


@gin.configurable
def extra_training_callbacks(fader: bool = False) -> Sequence[pl.Callback]:
    if not fader:
        return []
    from .fader.callbacks import LambdaWarmupCallback

    return [LambdaWarmupCallback()]
