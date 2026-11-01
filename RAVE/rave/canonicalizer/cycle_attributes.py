"""Per-attribute Case A/B/C resolution and conditioning assembly for CycleGAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attribute_marginals import load_discrete_marginal_probs
from .backbone import backbone_num_attributes

AttrCase = Literal["A", "B", "C"]


@dataclass
class DomainAttrInfo:
    """Native attribute schema for one domain (possibly empty).

    For joint Fader backbones, ``names`` is the **native** list used for Cases
    A/B/C and L_inv. Decode width comes from ``backbone.attribute_names``
    (union schema), which may be a superset of ``names``.
    """

    names: List[str] = field(default_factory=list)
    kinds: Dict[str, str] = field(default_factory=dict)
    num_classes: Dict[str, int] = field(default_factory=dict)
    priors: Dict[str, str] = field(default_factory=dict)  # uniform | marginal
    marginal_probs: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_attributes(self) -> int:
        return len(self.names)

    def restrict(self, names: Sequence[str]) -> "DomainAttrInfo":
        """Return a copy keeping only ``names`` (order preserved)."""
        keep = [n for n in names if n in self.names]
        return DomainAttrInfo(
            names=keep,
            kinds={n: self.kinds[n] for n in keep if n in self.kinds},
            num_classes={
                n: self.num_classes[n] for n in keep if n in self.num_classes
            },
            priors={n: self.priors[n] for n in keep if n in self.priors},
            marginal_probs={
                n: self.marginal_probs[n]
                for n in keep if n in self.marginal_probs
            },
        )

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        *,
        db_path: Optional[str] = None,
        prior: str = "uniform",
        native_names: Optional[Sequence[str]] = None,
    ) -> "DomainAttrInfo":
        names = list(getattr(backbone, "attribute_names", []) or [])
        if native_names is not None:
            # Empty string list ⇒ unconditional native domain on a joint Fader.
            allow = set(native_names)
            names = [n for n in names if n in allow]
        kinds = dict(getattr(backbone, "attribute_kinds", {}) or {})
        ncls = dict(getattr(backbone, "discrete_num_classes", {}) or {})
        default_bins = int(getattr(backbone, "num_classes", 16))
        per_attr_all = list(getattr(backbone, "num_classes_per_attribute", []) or [])
        schema = list(getattr(backbone, "attribute_names", []) or [])
        num_classes: Dict[str, int] = {}
        priors: Dict[str, str] = {}
        marginals: Dict[str, np.ndarray] = {}
        for name in names:
            if name in schema and schema.index(name) < len(per_attr_all):
                num_classes[name] = int(per_attr_all[schema.index(name)])
            elif kinds.get(name) == "discrete":
                num_classes[name] = int(ncls.get(name, 2))
            else:
                num_classes[name] = default_bins
            priors[name] = prior
            if (
                db_path
                and kinds.get(name) == "discrete"
                and priors[name] == "marginal"
            ):
                marginals[name] = load_discrete_marginal_probs(
                    db_path, name, num_classes[name]
                )
        return cls(
            names=names,
            kinds={n: kinds.get(n, "continuous") for n in names},
            num_classes=num_classes,
            priors=priors,
            marginal_probs=marginals,
        )


def resolve_cases(
    names_src: Sequence[str],
    names_tgt: Sequence[str],
) -> Dict[str, AttrCase]:
    """
    For direction src→tgt: case of each attribute in the union.

    A: in both; B: only in target; C: only in source.
    """
    src = set(names_src)
    tgt = set(names_tgt)
    out: Dict[str, AttrCase] = {}
    for name in names_tgt:
        out[name] = "A" if name in src else "B"
    for name in names_src:
        if name not in tgt:
            out[name] = "C"
    return out


@dataclass
class AssembledConditioning:
    """Decoder attr_norm (schema width) + classifier attr_cls (native width)."""

    attr_norm: Optional[torch.Tensor]  # (B, D_schema, T) or None
    attr_cls: Optional[torch.Tensor]  # (B, D_native, T) for AC-GAN


def _schema_names(backbone: nn.Module) -> List[str]:
    return list(getattr(backbone, "attribute_names", []) or [])


def _sample_classes(
    n: int,
    n_cls: int,
    *,
    device: torch.device,
    sampling: str,
    marginal: Optional[np.ndarray],
) -> torch.Tensor:
    if n_cls <= 1:
        return torch.zeros(n, device=device, dtype=torch.long)
    if sampling == "marginal" and marginal is not None:
        probs = torch.as_tensor(marginal, device=device, dtype=torch.float32)
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum()
        return torch.multinomial(probs, n, replacement=True)
    return torch.randint(0, n_cls, (n,), device=device)


def _true_row(
    attr_raw: Optional[torch.Tensor],
    src_names: Sequence[str],
    name: str,
    t_lat: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attr_raw is None or name not in src_names:
        return None
    # attr_raw may be schema-width (joint) or native-width (separate).
    # Prefer indexing by src_names when dims match; else by schema position
    # if attr_raw was loaded with full backbone order.
    if attr_raw.shape[1] == len(src_names):
        idx = list(src_names).index(name)
    else:
        # Fall back: caller may pass schema-aligned raw; search later via name index
        # only works when src_names subset — require matching dim or name in layout.
        return None
    row = attr_raw[:, idx, :]
    if row.shape[-1] != t_lat:
        row = F.interpolate(
            row.unsqueeze(1).float(), size=t_lat, mode="nearest"
        ).squeeze(1)
    return row.to(device)


def _true_row_schema(
    attr_raw: Optional[torch.Tensor],
    schema: Sequence[str],
    name: str,
    t_lat: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attr_raw is None or name not in schema:
        return None
    if attr_raw.shape[1] != len(schema):
        return None
    idx = list(schema).index(name)
    row = attr_raw[:, idx, :]
    if row.shape[-1] != t_lat:
        row = F.interpolate(
            row.unsqueeze(1).float(), size=t_lat, mode="nearest"
        ).squeeze(1)
    return row.to(device)


def assemble_target_conditioning(
    *,
    target: DomainAttrInfo,
    source: DomainAttrInfo,
    source_attr_raw: Optional[torch.Tensor],
    target_backbone: nn.Module,
    batch: int,
    t_lat: int,
    device: torch.device,
    swap_prob: float = 0.0,
    cases: Optional[Dict[str, AttrCase]] = None,
) -> AssembledConditioning:
    """
    Build c for decoding into ``target`` from a ``source`` sample.

    Decode ``attr_norm`` width = backbone schema (joint: |x∪y|).
    AC-GAN ``attr_cls`` width = native ``target.names``.

    Case A: pass-through (optional swap); Case B: sample prior;
    schema slots not in native target → neutral attr_norm.
    """
    schema = _schema_names(target_backbone)
    if not schema or backbone_num_attributes(target_backbone) == 0:
        return AssembledConditioning(None, None)

    cases = cases or resolve_cases(source.names, target.names)
    raw = torch.zeros(batch, len(schema), t_lat, device=device)
    native_set = set(target.names)

    for i, name in enumerate(schema):
        if name not in native_set:
            # Non-native schema slot: leave raw 0; zeroed to neutral after prepare.
            continue
        case = cases.get(name, "B")
        n_cls = int(target.num_classes.get(name, 2))
        if case == "A":
            row = _true_row(
                source_attr_raw, source.names, name, t_lat, device
            )
            if row is None:
                row = _true_row_schema(
                    source_attr_raw, schema, name, t_lat, device
                )
            if row is None:
                ids = _sample_classes(
                    batch,
                    n_cls,
                    device=device,
                    sampling=target.priors.get(name, "uniform"),
                    marginal=target.marginal_probs.get(name),
                )
                raw[:, i, :] = ids.float().unsqueeze(-1).expand(-1, t_lat)
            else:
                if swap_prob > 0.0 and batch > 1 and float(torch.rand(())) < swap_prob:
                    perm = torch.randperm(batch, device=device)
                    row = row[perm]
                raw[:, i, :] = row.float()
        else:  # B
            ids = _sample_classes(
                batch,
                n_cls,
                device=device,
                sampling=target.priors.get(name, "uniform"),
                marginal=target.marginal_probs.get(name),
            )
            raw[:, i, :] = ids.float().unsqueeze(-1).expand(-1, t_lat)

    attr_norm, attr_cls_full = target_backbone._prepare_attributes(raw)
    # Neutral non-native schema channels for decode.
    for i, name in enumerate(schema):
        if name not in native_set:
            attr_norm[:, i, :] = 0.0

    if target.num_attributes == 0:
        attr_cls = None
    else:
        idxs = [schema.index(n) for n in target.names]
        attr_cls = attr_cls_full[:, idxs, :]

    return AssembledConditioning(attr_norm, attr_cls)


def assemble_return_conditioning(
    *,
    source: DomainAttrInfo,
    source_attr_raw: Optional[torch.Tensor],
    source_backbone: nn.Module,
    batch: int,
    t_lat: int,
    device: torch.device,
) -> AssembledConditioning:
    """
    Bookkept return-leg c for decoding back into ``source``.

    Includes Case C (source-local) attrs via true labels when present.
    Joint: emit full schema-width attr_norm (neutral for missing slots).
    """
    schema = _schema_names(source_backbone)
    if not schema or backbone_num_attributes(source_backbone) == 0:
        return AssembledConditioning(None, None)

    raw = torch.zeros(batch, len(schema), t_lat, device=device)
    if source_attr_raw is not None:
        src = source_attr_raw.to(device)
        if src.shape[-1] != t_lat:
            src = F.interpolate(src.float(), size=t_lat, mode="nearest")
        if src.shape[0] != batch:
            raise ValueError(
                f"source_attr_raw batch {src.shape[0]} != expected {batch}"
            )
        if src.shape[1] == len(schema):
            raw = src.float()
        elif src.shape[1] == len(source.names):
            for j, name in enumerate(source.names):
                if name in schema:
                    raw[:, schema.index(name), :] = src[:, j, :].float()
        else:
            raise ValueError(
                f"source_attr_raw D={src.shape[1]} matches neither schema "
                f"{len(schema)} nor native {len(source.names)}"
            )
    elif source.num_attributes > 0:
        raise ValueError(
            "return conditioning requires source_attr_raw when source is Fader"
        )

    attr_norm, attr_cls_full = source_backbone._prepare_attributes(raw)
    native_set = set(source.names)
    for i, name in enumerate(schema):
        if name not in native_set:
            attr_norm[:, i, :] = 0.0

    if source.num_attributes == 0:
        attr_cls = None
    else:
        idxs = [schema.index(n) for n in source.names]
        attr_cls = attr_cls_full[:, idxs, :]

    return AssembledConditioning(attr_norm, attr_cls)


def ce_to_uniform(logits_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean CE of classifier heads vs Uniform over each head's classes."""
    if not logits_list:
        raise ValueError("ce_to_uniform requires at least one logit tensor")
    losses = []
    for logits in logits_list:
        # logits: (B, C, T) — uniform target ⇒ -mean(log_softmax) over class dim
        log_probs = F.log_softmax(logits, dim=1)
        losses.append(-log_probs.mean())
    return sum(losses) / len(losses)


def ce_to_uniform_on_heads(
    logits_list: Sequence[torch.Tensor],
    head_indices: Sequence[int],
) -> torch.Tensor:
    """CE-to-uniform restricted to ``head_indices`` (target-native L_inv)."""
    if not logits_list:
        raise ValueError("ce_to_uniform_on_heads requires logits")
    device = logits_list[0].device
    if not head_indices:
        return torch.zeros((), device=device)
    selected = [logits_list[i] for i in head_indices]
    return ce_to_uniform(selected)


def native_head_indices(
    backbone: nn.Module,
    native: DomainAttrInfo,
) -> List[int]:
    """Map native attr names to indices in backbone.attribute_names / C heads."""
    schema = _schema_names(backbone)
    return [schema.index(n) for n in native.names if n in schema]


def acgan_classification_loss(
    class_logits: Sequence[torch.Tensor],
    attr_cls: torch.Tensor,
) -> torch.Tensor:
    """CE for AC-GAN heads; attr_cls is (B, D, T), logits[i] is (B, C_i)."""
    if not class_logits:
        return torch.tensor(0.0, device=attr_cls.device)
    losses = []
    for i, logits in enumerate(class_logits):
        target = attr_cls[:, i, :].float().mean(dim=-1).round().long()
        n_cls = logits.shape[-1]
        target = target.clamp(0, n_cls - 1)
        losses.append(F.cross_entropy(logits, target))
    return sum(losses) / len(losses)
