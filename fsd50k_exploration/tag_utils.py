"""
Shared helpers to read classifier tags from FSD50K JSON sidecars (used by count_tags and build_subset).
"""

from __future__ import annotations

from typing import Any

__all__ = ["iter_clip_tags_raw", "normalize_tag"]


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def _flatten_str_list(xs: Any) -> list[str]:
    if xs is None:
        return []
    if isinstance(xs, str):
        return [xs]
    if isinstance(xs, (list, tuple)):
        out: list[str] = []
        for item in xs:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def iter_clip_tags_raw(metadata: dict[str, Any]) -> list[str]:
    """
    Prefer original_data[\"all_tags\"], then plain all_tags / tag / text (lists of strings).

    Returned strings are *not* normalized; callers normalize for counting/matching.
    """
    od = metadata.get("original_data")
    if isinstance(od, dict):
        all_tags = od.get("all_tags")
        got = _flatten_str_list(all_tags)
        if got:
            return got

    if "all_tags" in metadata:
        got = _flatten_str_list(metadata.get("all_tags"))
        if got:
            return got

    tag = metadata.get("tag")
    got = _flatten_str_list(tag)
    if got:
        return got

    text = metadata.get("text")
    return _flatten_str_list(text)
