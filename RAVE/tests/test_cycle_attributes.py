"""Unit tests for Case A/B/C resolution and conditioning assembly."""

import torch
import torch.nn as nn

from rave.canonicalizer.cycle_attributes import (
    DomainAttrInfo,
    assemble_return_conditioning,
    assemble_target_conditioning,
    ce_to_uniform,
    ce_to_uniform_on_heads,
    native_head_indices,
    resolve_cases,
)


class _FakeFader(nn.Module):
    def __init__(self, names, kinds, n_cls):
        super().__init__()
        self.attribute_names = list(names)
        self.attribute_kinds = dict(kinds)
        self.num_attributes = len(names)
        self.num_classes = 16
        self.discrete_num_classes = {
            n: n_cls[n] for n in names if kinds[n] == "discrete"
        }
        self.num_classes_per_attribute = [
            n_cls[n] if kinds[n] == "discrete" else 16 for n in names
        ]
        self.latent_size = 8
        self.register_buffer("bin_values", torch.zeros(len(names), 16))
        self.min_max_features = {n: (-1.0, 1.0) for n in names if kinds[n] == "continuous"}

    def _prepare_attributes(self, attr_raw):
        # Minimal: discrete indices → [-1,1] float; cls = rounded indices
        b, d, t = attr_raw.shape
        attr_cls = attr_raw.round().long()
        attr_norm = torch.zeros_like(attr_raw)
        for i, name in enumerate(self.attribute_names):
            if self.attribute_kinds[name] == "discrete":
                k = max(self.discrete_num_classes[name] - 1, 1)
                attr_norm[:, i] = 2.0 * (attr_raw[:, i] / k) - 1.0
            else:
                attr_norm[:, i] = attr_raw[:, i].clamp(-1, 1)
        return attr_norm, attr_cls


def test_resolve_cases_a_b_c():
    cases = resolve_cases(["prog", "local_x"], ["prog", "water_scene"])
    assert cases["prog"] == "A"
    assert cases["water_scene"] == "B"
    assert cases["local_x"] == "C"


def test_resolve_cases_empty():
    assert resolve_cases([], ["water_scene"])["water_scene"] == "B"
    assert resolve_cases(["local"], [])["local"] == "C"
    assert resolve_cases([], []) == {}


def test_assemble_case_b_samples_in_range():
    y = _FakeFader(["water_scene"], {"water_scene": "discrete"}, {"water_scene": 3})
    x_info = DomainAttrInfo()
    y_info = DomainAttrInfo.from_backbone(y)
    cond = assemble_target_conditioning(
        target=y_info,
        source=x_info,
        source_attr_raw=None,
        target_backbone=y,
        batch=4,
        t_lat=5,
        device=torch.device("cpu"),
    )
    assert cond.attr_norm is not None and cond.attr_cls is not None
    assert cond.attr_cls.shape == (4, 1, 5)
    assert int(cond.attr_cls.min()) >= 0
    assert int(cond.attr_cls.max()) < 3


def test_assemble_case_a_passthrough():
    bb = _FakeFader(["prog"], {"prog": "discrete"}, {"prog": 4})
    info = DomainAttrInfo.from_backbone(bb)
    raw = torch.tensor([[[2.0] * 3], [[1.0] * 3]])  # (2, 1, 3)
    cond = assemble_target_conditioning(
        target=info,
        source=info,
        source_attr_raw=raw,
        target_backbone=bb,
        batch=2,
        t_lat=3,
        device=torch.device("cpu"),
        swap_prob=0.0,
    )
    assert torch.equal(cond.attr_cls[:, 0, 0], torch.tensor([2, 1]))


def test_assemble_return_includes_source_attrs():
    x = _FakeFader(["local"], {"local": "discrete"}, {"local": 2})
    info = DomainAttrInfo.from_backbone(x)
    raw = torch.ones(2, 1, 4)
    cond = assemble_return_conditioning(
        source=info,
        source_attr_raw=raw,
        source_backbone=x,
        batch=2,
        t_lat=4,
        device=torch.device("cpu"),
    )
    assert cond.attr_norm is not None
    assert cond.attr_cls.shape == (2, 1, 4)


def test_empty_target_returns_none():
    y = _FakeFader([], {}, {})
    cond = assemble_target_conditioning(
        target=DomainAttrInfo(),
        source=DomainAttrInfo(),
        source_attr_raw=None,
        target_backbone=y,
        batch=2,
        t_lat=4,
        device=torch.device("cpu"),
    )
    assert cond.attr_norm is None and cond.attr_cls is None


def test_ce_to_uniform_finite():
    logits = [torch.zeros(2, 3, 4)]  # uniform already
    loss = ce_to_uniform(logits)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_joint_schema_case_b_native_subset():
    """Shared Fader schema wider than native_y ⇒ Case B on Y-only attr."""
    bb = _FakeFader(
        ["rms", "water_scene"],
        {"rms": "continuous", "water_scene": "discrete"},
        {"rms": 16, "water_scene": 3},
    )
    native_x = DomainAttrInfo()  # tap∅
    native_y = DomainAttrInfo.from_backbone(bb, native_names=["water_scene"])
    assert native_y.names == ["water_scene"]
    cases = resolve_cases(native_x.names, native_y.names)
    assert cases["water_scene"] == "B"

    cond = assemble_target_conditioning(
        target=native_y,
        source=native_x,
        source_attr_raw=None,
        target_backbone=bb,
        batch=4,
        t_lat=5,
        device=torch.device("cpu"),
        cases=cases,
    )
    assert cond.attr_norm is not None
    assert cond.attr_norm.shape == (4, 2, 5)  # full schema width
    assert cond.attr_cls is not None
    assert cond.attr_cls.shape == (4, 1, 5)  # native only
    # Non-native rms channel neutralized
    assert torch.allclose(cond.attr_norm[:, 0, :], torch.zeros(4, 5))
    assert int(cond.attr_cls.max()) < 3


def test_joint_return_empty_native_still_schema_width():
    bb = _FakeFader(
        ["water_scene"],
        {"water_scene": "discrete"},
        {"water_scene": 3},
    )
    native_x = DomainAttrInfo()
    cond = assemble_return_conditioning(
        source=native_x,
        source_attr_raw=None,
        source_backbone=bb,
        batch=2,
        t_lat=4,
        device=torch.device("cpu"),
    )
    assert cond.attr_norm is not None
    assert cond.attr_norm.shape == (2, 1, 4)
    assert cond.attr_cls is None
    assert torch.allclose(cond.attr_norm, torch.zeros(2, 1, 4))


def test_native_head_indices_and_linv_subset():
    bb = _FakeFader(
        ["rms", "water_scene"],
        {"rms": "continuous", "water_scene": "discrete"},
        {"rms": 16, "water_scene": 3},
    )
    native = DomainAttrInfo.from_backbone(bb, native_names=["water_scene"])
    assert native_head_indices(bb, native) == [1]
    logits = [torch.zeros(2, 16, 4), torch.zeros(2, 3, 4)]
    loss = ce_to_uniform_on_heads(logits, [1])
    assert torch.isfinite(loss)
    assert ce_to_uniform_on_heads(logits, []).item() == 0.0
