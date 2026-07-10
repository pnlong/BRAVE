"""Phase 3: marginal OOD sampling, class pools, class-stratified batching."""

import json

import numpy as np
import torch
import yaml

from rave.canonicalizer.attribute_marginals import (
    build_in_domain_class_pools,
    build_ood_discrete_marginals,
    load_discrete_marginal_probs,
    marginal_probs_from_counts,
    read_discrete_class_per_index,
)
from rave.canonicalizer.backbone import assign_ood_target_attrs
from rave.canonicalizer.dataset import (
    DOMAIN_IN,
    DOMAIN_OOD,
    DualSourceCanonicalizerDataset,
    StratifiedCanonicalizerBatchSampler,
    TaggedAudioDataset,
    _iter_class_stratified_in_batches,
    canonicalizer_collate,
)


class _FakeFaderBackbone:
    num_attributes = 2
    attribute_names = ["rms", "texture_class"]
    attribute_kinds = {"rms": "continuous", "texture_class": "discrete"}
    discrete_num_classes = {"texture_class": 4}


class _MockAudioDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 8) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> np.ndarray:
        return np.zeros((1, 64), dtype=np.float32)


def _write_sidecar(tmp_path, attr_name: str, values: dict) -> None:
    payload = {
        "attributes": {
            attr_name: {
                "kind": "discrete",
                "values": values,
            }
        }
    }
    (tmp_path / "attribute_sidecar.yaml").write_text(yaml.safe_dump(payload))


def test_marginal_probs_from_counts_normalized():
    probs = marginal_probs_from_counts({0: 3, 1: 1}, 3)
    assert np.allclose(probs[:2].sum(), 1.0)
    assert probs[2] == 0.0
    assert abs(probs[0] - 0.75) < 1e-6


def test_read_discrete_class_per_index_from_sidecar(tmp_path):
    _write_sidecar(
        tmp_path,
        "texture_class",
        {"00000000": 1, "00000001": 2, "00000002": 1},
    )
    out = read_discrete_class_per_index(tmp_path, "texture_class", 4)
    assert out.tolist() == [1, 2, 1, 0]


def test_load_discrete_marginal_probs_from_class_counts_json(tmp_path):
    counts = {"counts": {0: 8, 1: 2}}
    (tmp_path / "texture_class_class_counts.json").write_text(
        json.dumps(counts))
    probs = load_discrete_marginal_probs(
        tmp_path, "texture_class", 2, num_train_indices=0)
    assert np.allclose(probs, [0.8, 0.2])


def test_build_in_domain_class_pools(tmp_path):
    _write_sidecar(
        tmp_path,
        "texture_class",
        {"00000000": 0, "00000001": 1, "00000002": 0, "00000003": 1},
    )
    pools = build_in_domain_class_pools(
        tmp_path, "texture_class", num_indices=4, n_classes=2)
    assert pools[0] == [0, 2]
    assert pools[1] == [1, 3]


def test_assign_ood_target_attrs_marginal_sampling():
    model = _FakeFaderBackbone()
    marginals = {
        "texture_class": torch.tensor([0.9, 0.05, 0.03, 0.02]),
    }
    attr = torch.zeros(32, 2, 4)
    ood_mask = torch.ones(32, dtype=torch.bool)
    torch.manual_seed(0)
    out = assign_ood_target_attrs(
        attr,
        model,
        ood_mask,
        discrete_sampling="marginal",
        marginal_probs=marginals,
    )
    classes = out[:, 1, 0].long()
    assert torch.all((classes >= 0) & (classes < 4))
    assert classes.float().mean() < 1.5


def test_class_stratified_in_batches_cover_classes():
    pools = {0: [0, 2, 4], 1: [1, 3, 5]}
    batches = _iter_class_stratified_in_batches(
        pools, n_in=4, num_batches=2, shuffle=False)
    assert len(batches) == 2
    assert len(batches[0]) == 4
    assert len(set(batches[0])) >= 2


def test_stratified_sampler_with_class_pools():
    in_ds = TaggedAudioDataset(_MockAudioDataset(size=6), domain=DOMAIN_IN)
    ood_ds = TaggedAudioDataset(_MockAudioDataset(size=6), domain=DOMAIN_OOD)
    dual = DualSourceCanonicalizerDataset(in_ds, ood_ds)
    class_pools = {0: [0, 2, 4], 1: [1, 3, 5]}
    sampler = StratifiedCanonicalizerBatchSampler(
        len_in_domain=dual.len_in_domain,
        len_ood=dual.len_ood,
        batch_size=4,
        in_domain_fraction=0.5,
        shuffle=False,
        class_pools=class_pools,
    )
    batch_indices = next(iter(sampler))
    batch = canonicalizer_collate([dual[i] for i in batch_indices])
    _, _, domains = batch
    assert domains.count(DOMAIN_IN) == 2
    assert domains.count(DOMAIN_OOD) == 2
    in_indices = [i for i in batch_indices if i < dual.len_in_domain]
    assert len(set(in_indices)) >= 2


def test_build_ood_discrete_marginals(tmp_path):
    _write_sidecar(
        tmp_path,
        "texture_class",
        {"00000000": 0, "00000001": 1, "00000002": 0},
    )
    out = build_ood_discrete_marginals(
        ["rms", "texture_class"],
        {"rms": "continuous", "texture_class": "discrete"},
        {"texture_class": 2},
        tmp_path,
        num_train_indices=3,
    )
    assert "texture_class" in out
    assert np.allclose(out["texture_class"], [2 / 3, 1 / 3])


def test_plot_ood_class_summary_returns_figure():
    from rave.canonicalizer.viz import plot_ood_class_summary

    fig = plot_ood_class_summary(
        {0: [-0.5, -0.3], 1: [-0.1]},
        {0: [40.0, 42.0], 1: [38.0]},
        class_labels={0: "ocean", 1: "nature"},
    )
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
