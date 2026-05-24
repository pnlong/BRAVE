"""
Shared helpers for normalizing ontology class tokens (e.g. from FSD50K CSV ``labels``).
"""

from __future__ import annotations

__all__ = ["normalize_tag"]


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()
