#!/usr/bin/env python3
"""Print stored audio duration from a RAVE preprocess LMDB metadata.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RAVE_ROOT = Path(__file__).resolve().parents[1] / "RAVE"
if str(_RAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAVE_ROOT))

from rave.preprocess_metadata import read_stored_sec_from_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report stored seconds/hours in a preprocessed LMDB.")
    parser.add_argument(
        "db_path",
        help="Path to LMDB directory (contains metadata.yaml)",
    )
    args = parser.parse_args()

    sec = read_stored_sec_from_metadata(args.db_path)
    print(f"path:     {args.db_path}")
    print(f"seconds:  {sec:.2f}")
    print(f"minutes:  {sec / 60:.2f}")
    print(f"hours:    {sec / 3600:.4f}")


if __name__ == "__main__":
    main()
