"""
scripts/_cli.py
---------------
Argument types shared by the scripts. A script puts the repository root on
`sys.path` first, then imports these as `scripts._cli`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


def positive_float(value: str) -> float:
    x = float(value)
    if not x > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return x


def share(value: str) -> float:
    """A number in [0, 1]."""
    x = float(value)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return x


def named_path(value: str) -> tuple[str, Path]:
    """`NAME=PATH`, or a bare `PATH` named after its file stem."""
    name, sep, path = value.partition("=")
    if not sep:
        return Path(value).stem, Path(value)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected [NAME=]PATH, got {value!r}")
    return name, Path(path)


def comma_list(values: Sequence[str]) -> list[str]:
    """`["a,b", "a", "c,"]` -> ["a", "b", "c"]: comma-separated or repeated, order kept."""
    out: list[str] = []
    for value in values:
        out += [s for s in (item.strip() for item in value.split(",")) if s and s not in out]
    return out
