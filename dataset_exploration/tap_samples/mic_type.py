"""Classify tap recordings by microphone from a ``{mic}-`` filename prefix.

Canonical names start with one of:
  - ``Shotgun-``
  - ``Contact Mic L-`` / ``Contact Mic R-``
  - ``sm57-``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Sequence, Tuple

MicType = Literal["shotgun", "contact", "sm57"]

MIC_TYPES: tuple[MicType, ...] = ("shotgun", "contact", "sm57")

AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".opus", ".aac")

_CONTACT_TAIL = re.compile(
    r"^(?P<song>.+)-Contact Mic (?P<side>[LR])(?P<rest>\..+)$",
    re.IGNORECASE,
)
_MIC_PREFIX = re.compile(
    r"^(?:Shotgun-|Contact Mic [LR]-|sm57-)",
    re.IGNORECASE,
)
_STYLE_TAKE = re.compile(r"^([A-Za-z]+)(?:_(\d+))?", re.IGNORECASE)


class EvalHoldout(NamedTuple):
    path: Path
    style: str
    held_out: bool
    n_in_style: int


def split_eval_times(
    duration_sec: float, eval_seconds: float
) -> Tuple[float, float, float]:
    """Prefix stays in train; tail is eval.

    Returns ``(train_end, eval_start, eval_dur)``. ``train_end == 0`` means
    the whole file is eval (file shorter than the crop, or ``eval_seconds<=0``
    for a full-file holdout).
    """
    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0
    if eval_seconds <= 0:
        return 0.0, 0.0, duration_sec
    eval_dur = min(float(eval_seconds), duration_sec)
    train_end = duration_sec - eval_dur
    if train_end < 0.5:
        return 0.0, 0.0, duration_sec
    return train_end, train_end, eval_dur


def classify_tap_mic(name: str) -> MicType:
    """Return mic class from a canonical ``{mic}-`` prefix."""
    stem = Path(name).name
    lower = stem.lower()
    if lower.startswith("shotgun-"):
        return "shotgun"
    if lower.startswith("contact mic "):
        return "contact"
    if lower.startswith("sm57-"):
        return "sm57"
    raise ValueError(
        f"unclassified tap filename {stem!r}; "
        "expected prefix Shotgun-, Contact Mic L-/R-, or sm57-"
    )


def canonical_tap_filename(name: str) -> str:
    """Rewrite a basename so it starts with ``Shotgun-``, ``Contact Mic L/R-``, or ``sm57-``."""
    name = Path(name).name
    lower = name.lower()
    if (
        lower.startswith("shotgun-")
        or lower.startswith("contact mic ")
        or lower.startswith("sm57-")
    ):
        return name
    m = _CONTACT_TAIL.match(name)
    if m:
        return f"Contact Mic {m.group('side').upper()}-{m.group('song')}{m.group('rest')}"
    return f"sm57-{name}"


def planned_renames(root: Path) -> List[Tuple[Path, Path]]:
    """``(src, dest)`` pairs that still need a rename (non-canonical names)."""
    root = root.resolve()
    pairs: List[Tuple[Path, Path]] = []
    for p in iter_audio_files(root):
        new_name = canonical_tap_filename(p.name)
        if new_name != p.name:
            pairs.append((p, p.with_name(new_name)))
    return pairs


def rename_tap_files(root: Path, *, dry_run: bool = False) -> List[Tuple[Path, Path]]:
    pairs = planned_renames(root)
    dests = [d for _, d in pairs]
    clash = [d for d in dests if d.exists()]
    if clash:
        raise FileExistsError(
            "rename would overwrite: " + ", ".join(d.name for d in clash)
        )
    seen = [d.name for d in dests]
    if len(seen) != len(set(seen)):
        raise ValueError("rename would collide within the batch")
    if not dry_run:
        for src, dest in pairs:
            src.rename(dest)
    return pairs


def iter_audio_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def classify_dir(root: Path) -> Dict[str, MicType]:
    """Map relative posix path → mic type."""
    root = root.resolve()
    out: Dict[str, MicType] = {}
    for p in iter_audio_files(root):
        rel = p.relative_to(root).as_posix()
        out[rel] = classify_tap_mic(p.name)
    return out


def files_for_mic(root: Path, mic: MicType) -> List[Path]:
    root = root.resolve()
    return [
        root / rel
        for rel, kind in classify_dir(root).items()
        if kind == mic
    ]


def counts(mapping: Dict[str, MicType]) -> Dict[str, int]:
    c = {k: 0 for k in MIC_TYPES}
    for kind in mapping.values():
        c[kind] += 1
    return c


def tap_style(name: str) -> str:
    """Gesture/style token after the mic prefix (``drumroll``, ``heavy``, …)."""
    rest = _MIC_PREFIX.sub("", Path(name).name)
    m = _STYLE_TAKE.match(rest)
    if not m:
        return Path(rest).stem.lower()
    return m.group(1).lower()


def tap_take_key(name: str) -> Tuple[int, int, str]:
    """Later numbered take sorts last; ``.unnormalized_1`` is a later duplicate."""
    stem = Path(name).name
    rest = _MIC_PREFIX.sub("", stem)
    m = _STYLE_TAKE.match(rest)
    take = int(m.group(2)) if m and m.group(2) else 0
    dup = 1 if re.search(r"unnormalized_\d+", stem, re.IGNORECASE) else 0
    return (take, dup, stem.lower())


def select_eval_holdouts(
    files: Sequence[Path],
    *,
    hold_singletons: bool = False,
) -> List[EvalHoldout]:
    """One file per style: highest take. Singleton styles stay in train unless asked."""
    by_style: Dict[str, List[Path]] = {}
    for p in files:
        by_style.setdefault(tap_style(p.name), []).append(p)
    picks: List[EvalHoldout] = []
    for style, group in sorted(by_style.items()):
        chosen = max(group, key=lambda p: tap_take_key(p.name))
        n = len(group)
        held = n > 1 or hold_singletons
        picks.append(
            EvalHoldout(path=chosen, style=style, held_out=held, n_in_style=n)
        )
    return picks
