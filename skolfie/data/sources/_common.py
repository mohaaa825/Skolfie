"""
skolfie/data/sources/_common.py
-------------------------------
Small helpers shared by the source loaders: reading JSONL with line
numbers, picking a field from a list of candidate keys (with an error that
lists the keys actually present), guessing a split from a file name, and
downloading from the Hugging Face Hub with errors that name the URL.

`huggingface_hub` is imported lazily so that reading already-downloaded
files never needs it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MISSING = object()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield `(line_index, row)` for each non-blank line; line_index counts from 0."""
    with open(path, encoding="utf-8") as fh:
        index = -1
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            index += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                kind = type(row).__name__
                raise ValueError(f"{path}:{lineno}: expected a JSON object, got {kind}")
            yield index, row


def first_key(row: dict, candidates: Iterable[str]) -> str | None:
    """The first of `candidates` present in `row` (exact match, then case-insensitive)."""
    candidates = list(candidates)
    for key in candidates:
        if key in row:
            return key
    lowered = {str(k).lower(): k for k in row}
    for key in candidates:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def pick(
    row: dict, candidates: Iterable[str], where: str, what: str, default: Any = _MISSING
) -> Any:
    """`row[k]` for the first candidate key k present; error naming the keys seen if none is."""
    candidates = list(candidates)
    key = first_key(row, candidates)
    if key is not None:
        return row[key]
    if default is not _MISSING:
        return default
    raise ValueError(
        f"{where}: no {what} field; looked for {candidates}, found keys {sorted(map(str, row))}"
    )


def split_from_name(name: str) -> str:
    """Guess the source's split from a file or directory name ('' if none)."""
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    if tokens & {"train", "training"}:
        return "train"
    if tokens & {"dev", "valid", "validation", "val"}:
        return "dev"
    if tokens & {"test", "testing"}:
        return "test"
    return ""


def missing_files_error(
    source: str, directory: Path, expected: str, manual: str
) -> FileNotFoundError:
    return FileNotFoundError(
        f"{source}: no data files found in {directory} (expected {expected}).\n{manual.strip()}"
    )


def _hf():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "huggingface_hub is required to download; install it (pip install huggingface_hub) "
            "or place the files manually"
        ) from exc
    return huggingface_hub


def hf_url(repo_id: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def hf_download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download one dataset-repo file into `local_dir`; RuntimeError naming the URL on failure."""
    hub = _hf()
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hub.hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(local_dir)
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not download {hf_url(repo_id, filename)}: {type(exc).__name__}: {exc}"
        ) from exc
    return Path(path)


def hf_snapshot(
    repo_id: str, local_dir: Path, allow_patterns: list[str], revision: str | None = None
) -> Path:
    """Download the files of a dataset repo matching `allow_patterns`; RuntimeError on failure."""
    hub = _hf()
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = hub.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=str(local_dir),
        )
    except Exception as exc:
        url = f"https://huggingface.co/datasets/{repo_id}"
        if revision:
            url += f"/tree/{revision}"
        raise RuntimeError(f"could not download {url}: {type(exc).__name__}: {exc}") from exc
    return Path(path)
