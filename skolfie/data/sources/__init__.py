"""
skolfie/data/sources/__init__.py
--------------------------------
Registry of corpus source loaders.

Each source module exposes NAME, DESCRIPTION, CITATION, LICENSE_NOTE,
MANUAL_DOWNLOAD and `load(raw_dir, download=True) -> Iterator[Record]`; a loader
may take extra keyword options (persuade: `max_essays`, `seed`), which the
corpus builder passes through `source_options`.
`raw_dir` is the top-level raw data directory; each loader reads (and
downloads into) its own `raw_dir/<NAME>/` subdirectory. Missing data raises
FileNotFoundError naming the expected path, and a failed download raises
RuntimeError naming the URL.
"""

from __future__ import annotations

from types import ModuleType

from skolfie.data.sources import autextification, fresh, hc3, m4gt, persuade

SOURCES: dict[str, ModuleType] = {
    m.NAME: m for m in (hc3, m4gt, autextification, persuade, fresh)
}


def get_source(name: str) -> ModuleType:
    """The loader module registered under `name` (case-insensitive)."""
    try:
        return SOURCES[name.strip().lower()]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; available: {sorted(SOURCES)}") from None


__all__ = ["SOURCES", "get_source"]
