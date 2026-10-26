"""
skolfie/data/corpus.py
-----------------------
Build one versioned training corpus from the registered dataset sources.

`build_corpus()` pulls `Record`s from each source in
`skolfie.data.sources.SOURCES`, cleans the text, recomputes `n_words`,
drops texts outside the length bounds, removes exact duplicates (on a
normalised hash: case-folded, punctuation and whitespace stripped), drops
every copy of a text that appears with conflicting labels, optionally
balances human/ai per domain, and writes:

    out_dir/corpus.parquet   exactly `COLUMNS`, sorted by id
    out_dir/manifest.json    parameters, per-source counts and drop reasons,
                             label x generator x domain counts, sha256 of the
                             parquet file, `corpus_version`
    out_dir/stats.md         human-readable summary tables

Memory: each source is materialised once as a list of dicts, converted to a
compact Arrow table, and its Python objects are released before the next
source is read. Only a 16-byte hash plus id/label/domain per row stays in
Python for deduplication, so all the work after loading is O(n log n)
(the final sort by id).

Near-duplicate removal is not implemented: only exact duplicates of the
normalised text are removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from skolfie.data.schema import COLUMNS, LABELS, Record

log = logging.getLogger(__name__)

BALANCE_MODES = ("none", "downsample")
ROW_GROUP_SIZE = 20_000

#: Drop reasons, in the order they are applied (and shown in stats.md).
DROP_REASONS = (
    "invalid",            # the loader yielded something that is not a valid Record
    "too_short",          # fewer than min_words words after cleaning
    "too_long",           # more than max_words words after cleaning
    "duplicate_id",       # an id already seen in this build
    "duplicate",          # exact duplicate (normalised text) of a kept record
    "conflicting_label",  # same normalised text appears as both human and ai
    "balance",            # removed by --balance downsample
)

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("generator", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("subdomain", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("split_hint", pa.string(), nullable=False),
        pa.field("n_words", pa.int64(), nullable=False),
    ]
)
assert SCHEMA.names == COLUMNS

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

# Line / paragraph separators mapped onto "\n" before whitespace handling.
_NEWLINE_MAP = {
    "\r\n": "\n",
    "\r": "\n",
    "\x0b": "\n",       # vertical tab
    "\x0c": "\n\n",     # form feed: page break, treat as a paragraph break
    "\x85": "\n",       # NEL
    " ": "\n",     # LINE SEPARATOR
    " ": "\n\n",   # PARAGRAPH SEPARATOR
}
_NEWLINE_RE = re.compile("|".join(re.escape(k) for k in _NEWLINE_MAP))

# Every C0/C1 control character except "\n" and "\t" is deleted ("\t" is
# whitespace and gets collapsed below), plus a few invisible format characters
# that carry no meaning in prose: BOM, zero-width space, word joiner.
# ZWJ/ZWNJ (U+200C/U+200D) are kept: some scripts and emoji need them.
_DELETE_TABLE = {
    cp: None
    for cp in [*range(0x00, 0x20), *range(0x7F, 0xA0), 0xFEFF, 0x200B, 0x2060]
    if cp not in (0x0A, 0x09)
}

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_NON_WORD_RE = re.compile(r"[\W_]+")


def clean_text(text: Any) -> str:
    """Normalise one text for the corpus.

    Unicode NFC; CR/CRLF and Unicode line separators become newlines;
    control characters (and BOM / zero-width space / word joiner) are
    removed; a blank line (two or more newlines, possibly with spaces
    between them) is a paragraph break and is written as exactly "\\n\\n";
    every other whitespace run, including a single newline and non-breaking
    or other Unicode spaces, becomes one space; the result is stripped.

    Case and punctuation are left alone: stylometric features need them.
    Non-string input (None, NaN) gives "".
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _NEWLINE_RE.sub(lambda m: _NEWLINE_MAP[m.group(0)], text)
    text = text.translate(_DELETE_TABLE)
    paragraphs = (" ".join(p.split()) for p in _PARAGRAPH_SPLIT_RE.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def normalized_hash(text: str) -> bytes:
    """16-byte hash of `text` case-folded with punctuation/whitespace removed."""
    key = _NON_WORD_RE.sub("", text.casefold())
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()


def count_words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def load_registry() -> Mapping[str, ModuleType]:
    """Return `skolfie.data.sources.SOURCES` (name -> loader module).

    Imported lazily so this module works (and is testable) without the
    source loaders; tests monkeypatch this function.
    """
    from skolfie.data.sources import SOURCES

    return SOURCES


def source_hint(module: Any) -> str:
    """Best-effort description of what a source needs (URL / files), for errors."""
    parts = []
    for attr in ("URL", "URLS", "DATASET_URL", "HF_DATASET", "REPO_ID", "FILES", "REQUIRED_FILES"):
        value = getattr(module, attr, None)
        if value:
            if isinstance(value, (list, tuple, set)):
                value = ", ".join(map(str, value))
            elif isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            parts.append(f"{attr}={value}")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


class SourceError(RuntimeError):
    """A source loader failed (raised only with strict=True)."""

    def __init__(self, source: str, error: BaseException, hint: str = "") -> None:
        self.source = source
        self.error = error
        self.hint = hint
        msg = f"source {source!r} failed: {type(error).__name__}: {error}"
        if hint and "http" not in str(error):  # the loader already named the URL
            msg += f" (source needs: {hint})"
        super().__init__(msg)


class EmptyCorpusError(RuntimeError):
    """No record survived loading and filtering; nothing was written."""

    def __init__(self, source_stats: dict[str, dict[str, Any]]) -> None:
        self.source_stats = source_stats
        super().__init__("no records left after loading and filtering; nothing written")


@dataclass
class CorpusResult:
    out_dir: Path
    parquet_path: Path
    manifest_path: Path
    stats_path: Path
    n_records: int
    corpus_version: str
    manifest: dict[str, Any]
    stats_markdown: str
    #: source name -> error message, for sources that failed (non-strict mode)
    errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _empty_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {"raw": 0}
    stats.update({reason: 0 for reason in DROP_REASONS})
    stats["kept"] = 0
    stats["error"] = None
    return stats


def _coerce_record(item: Any) -> Record:
    if isinstance(item, Record):
        return item
    if isinstance(item, Mapping):
        return Record(**{k: item[k] for k in COLUMNS if k in item})
    raise TypeError(f"expected Record or mapping, got {type(item).__name__}")


def _load_source(
    name: str,
    module: Any,
    raw_dir: Path,
    *,
    download: bool,
    min_words: int,
    max_words: int,
    seen_ids: set[str],
    stats: dict[str, Any],
) -> tuple[pa.Table, list[bytes]]:
    """Read, validate, clean and length-filter one source into an Arrow table.

    Also returns the normalised-text hash of each row (computed here, while
    the text is still a Python string).

    `seen_ids` is only updated once the whole source loaded, so a source that
    fails half-way leaves no trace.
    """
    columns: dict[str, list[Any]] = {c: [] for c in COLUMNS}
    hashes: list[bytes] = []
    local_ids: set[str] = set()
    counts = Counter()
    for item in module.load(raw_dir, download=download):
        counts["raw"] += 1
        try:
            rec = _coerce_record(item)
            if not isinstance(rec.id, str) or not rec.id:
                raise ValueError("empty id")
        except (TypeError, ValueError, KeyError) as exc:
            counts["invalid"] += 1
            if counts["invalid"] <= 3:
                log.warning("%s: invalid record skipped: %s", name, exc)
            continue
        text = clean_text(rec.text)
        n_words = count_words(text)
        if n_words < min_words:
            counts["too_short"] += 1
            continue
        if n_words > max_words:
            counts["too_long"] += 1
            continue
        if rec.id in seen_ids or rec.id in local_ids:
            counts["duplicate_id"] += 1
            continue
        local_ids.add(rec.id)
        columns["id"].append(rec.id)
        columns["text"].append(text)
        columns["label"].append(rec.label)
        columns["generator"].append(str(rec.generator))
        columns["domain"].append(rec.domain)
        columns["subdomain"].append(str(rec.subdomain or ""))
        columns["source"].append(str(rec.source or name))
        columns["split_hint"].append(str(rec.split_hint or ""))
        columns["n_words"].append(n_words)
        hashes.append(normalized_hash(text))
    seen_ids |= local_ids
    for key, value in counts.items():
        stats[key] += value
    return pa.Table.from_pydict(columns, schema=SCHEMA), hashes


def _balance_mask(
    labels: Sequence[str], domains: Sequence[str], ids: Sequence[str], keep: list[bool], seed: int
) -> list[str]:
    """Downsample the majority label per domain, in place on `keep`.

    Returns warnings for domains that lose everything because one label is
    absent there.
    """
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: {lab: [] for lab in LABELS})
    for i, ok in enumerate(keep):
        if ok:
            groups[domains[i]][labels[i]].append(i)
    warnings = []
    rng = random.Random(seed)
    for domain in sorted(groups):
        by_label = groups[domain]
        target = min(len(v) for v in by_label.values())
        if target == 0:
            present = {k: len(v) for k, v in by_label.items() if v}
            warnings.append(
                f"balance: domain {domain!r} has no "
                f"{' / '.join(k for k, v in by_label.items() if not v)} records; "
                f"dropped all {sum(present.values())} of its records"
            )
        for label in LABELS:
            idx = by_label[label]
            if len(idx) <= target:
                continue
            idx.sort(key=lambda i: ids[i])  # independent of source order
            chosen = set(rng.sample(idx, target))
            for i in idx:
                if i not in chosen:
                    keep[i] = False
    return warnings


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty = None
    if commit:
        status = run("status", "--porcelain", "--untracked-files=no")
        dirty = bool(status) if status is not None else None
    return {"git_commit": commit, "git_dirty": dirty}


def _update_content_hash(h: Any, table: pa.Table) -> None:
    """Feed rows (in order) into `h`: a hash independent of the Parquet encoding."""
    cols = [table.column(c).to_pylist() for c in COLUMNS]
    for row in zip(*cols):
        h.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_corpus(
    source_names: Iterable[str],
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    min_words: int = 30,
    max_words: int = 2000,
    download: bool = True,
    seed: int = 0,
    balance: str = "none",
    strict: bool = False,
    registry: Mapping[str, Any] | None = None,
) -> CorpusResult:
    """Build `out_dir/corpus.parquet`, `manifest.json` and `stats.md`.

    Each source is called as `module.load(raw_dir, download=download)`. A
    source that raises is logged and recorded in the manifest (its partial
    output is discarded) and the build continues, unless `strict=True`, in
    which case `SourceError` is raised. If no record survives,
    `EmptyCorpusError` is raised and no file is written.
    """
    if balance not in BALANCE_MODES:
        raise ValueError(f"balance must be one of {BALANCE_MODES}, got {balance!r}")
    if min_words < 0 or max_words < min_words:
        raise ValueError(f"need 0 <= min_words <= max_words, got {min_words}, {max_words}")
    registry = load_registry() if registry is None else registry
    names = list(dict.fromkeys(source_names))
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise KeyError(f"unknown source(s) {unknown}; registered: {sorted(registry)}")
    if not names:
        raise ValueError("no sources given")

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    started = datetime.now(timezone.utc)

    source_stats: dict[str, dict[str, Any]] = {n: _empty_stats() for n in names}
    errors: dict[str, str] = {}
    tables: list[pa.Table] = []
    owner: list[str] = []  # source name per row of the concatenated table
    hashes: list[bytes] = []  # normalised-text hash per row
    seen_ids: set[str] = set()
    for name in names:
        module = registry[name]
        stats = source_stats[name]
        log.info("loading source %s", name)
        try:
            table, table_hashes = _load_source(
                name, module, raw_dir, download=download, min_words=min_words,
                max_words=max_words, seen_ids=seen_ids, stats=stats,
            )
        except Exception as exc:  # any loader failure is reported, not fatal
            hint = source_hint(module)
            err = SourceError(name, exc, hint)
            if strict:
                raise err from exc
            log.error("%s", err)
            stats.update(_empty_stats())
            stats["error"] = str(err)
            errors[name] = str(err)
            continue
        log.info("%s: %d records after cleaning and length filter", name, table.num_rows)
        tables.append(table)
        owner.extend([name] * table.num_rows)
        hashes.extend(table_hashes)
        del table_hashes

    table = pa.concat_tables(tables) if tables else SCHEMA.empty_table()
    del tables

    # --- exact dedupe + conflicting labels (O(n)) -------------------------- #
    ids = table.column("id").to_pylist()
    labels = table.column("label").to_pylist()
    domains = table.column("domain").to_pylist()

    best: dict[bytes, int] = {}
    group_label: dict[bytes, str | None] = {}  # None marks a label conflict
    for i, h in enumerate(hashes):
        if h not in best:
            best[h] = i
            group_label[h] = labels[i]
            continue
        if group_label[h] is not None and group_label[h] != labels[i]:
            group_label[h] = None
        if ids[i] < ids[best[h]]:
            best[h] = i

    keep = [True] * len(ids)
    for i, h in enumerate(hashes):
        if group_label[h] is None:
            keep[i] = False
            source_stats[owner[i]]["conflicting_label"] += 1
        elif best[h] != i:
            keep[i] = False
            source_stats[owner[i]]["duplicate"] += 1
    del hashes, best, group_label

    # --- balance ----------------------------------------------------------- #
    warnings: list[str] = []
    if balance == "downsample":
        before = list(keep)
        warnings = _balance_mask(labels, domains, ids, keep, seed)
        for i, (was, now) in enumerate(zip(before, keep)):
            if was and not now:
                source_stats[owner[i]]["balance"] += 1
    for w in warnings:
        log.warning("%s", w)

    for i, ok in enumerate(keep):
        if ok:
            source_stats[owner[i]]["kept"] += 1
    n_records = sum(keep)
    if n_records == 0:
        raise EmptyCorpusError(source_stats)

    # Kept rows in id order (Python str order == UTF-8 byte order).
    order = sorted((i for i, ok in enumerate(keep) if ok), key=ids.__getitem__)
    del ids, labels, domains, keep, owner
    # One contiguous copy, so that each slice's take() below does not
    # re-concatenate every chunk (which costs several times the table size).
    table = table.combine_chunks()

    # --- write, in row-group slices ---------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "corpus.parquet"
    manifest_path = out_dir / "manifest.json"
    stats_path = out_dir / "stats.md"
    tmp_parquet = parquet_path.with_name(parquet_path.name + ".tmp")
    content_hash = hashlib.sha256()
    combo: Counter = Counter()
    with pq.ParquetWriter(tmp_parquet, SCHEMA, compression="zstd") as writer:
        for start in range(0, len(order), ROW_GROUP_SIZE):
            part = table.take(pa.array(order[start:start + ROW_GROUP_SIZE], type=pa.int64()))
            writer.write_table(part, row_group_size=ROW_GROUP_SIZE)
            _update_content_hash(content_hash, part)
            combo.update(zip(part.column("label").to_pylist(),
                             part.column("generator").to_pylist(),
                             part.column("domain").to_pylist()))
    os.replace(tmp_parquet, parquet_path)
    del table, order, part

    content_sha = content_hash.hexdigest()
    corpus_version = f"v-{content_sha[:16]}"
    counts = [
        {"label": lab, "generator": gen, "domain": dom, "count": n}
        for (lab, gen, dom), n in sorted(combo.items())
    ]

    manifest: dict[str, Any] = {
        "corpus_version": corpus_version,
        "content_sha256": content_sha,
        "parquet_sha256": _file_sha256(parquet_path),
        "parquet_file": parquet_path.name,
        "n_records": n_records,
        "columns": list(COLUMNS),
        "build_time": started.isoformat(timespec="seconds"),
        **_git_info(),
        "versions": {"pyarrow": pa.__version__},
        "parameters": {
            "sources": names,
            "raw_dir": str(raw_dir),
            "min_words": min_words,
            "max_words": max_words,
            "download": download,
            "seed": seed,
            "balance": balance,
            "strict": strict,
            "dedupe": "exact (normalised: case-folded, punctuation/whitespace removed)",
            "near_dedupe": False,
        },
        "sources": source_stats,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
        "totals": {
            "label": _sum_by(counts, "label"),
            "domain": _sum_by(counts, "domain"),
            "generator": _sum_by(counts, "generator"),
        },
    }
    stats_md = render_stats(manifest)
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(stats_path, stats_md)
    log.info("wrote %d records to %s (%s)", n_records, parquet_path, corpus_version)

    return CorpusResult(
        out_dir=out_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        stats_path=stats_path,
        n_records=n_records,
        corpus_version=corpus_version,
        manifest=manifest,
        stats_markdown=stats_md,
        errors=errors,
    )


def _sum_by(counts: list[dict[str, Any]], key: str) -> dict[str, int]:
    total: Counter = Counter()
    for c in counts:
        total[c[key]] += c["count"]
    return dict(sorted(total.items()))


# --------------------------------------------------------------------------- #
# stats.md
# --------------------------------------------------------------------------- #


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_stats(manifest: Mapping[str, Any]) -> str:
    """Human-readable Markdown summary of a manifest."""
    p = manifest["parameters"]
    out = [
        f"# Corpus {manifest['corpus_version']}",
        "",
        f"- records: {manifest['n_records']}",
        f"- built: {manifest['build_time']}"
        + (f" at commit {manifest['git_commit'][:12]}" if manifest.get("git_commit") else ""),
        f"- parameters: min_words={p['min_words']}, max_words={p['max_words']}, "
        f"balance={p['balance']}, seed={p['seed']}, download={p['download']}",
        f"- parquet sha256: {manifest['parquet_sha256']}",
        "",
        "## Sources",
        "",
    ]
    cols = ["raw", *DROP_REASONS, "kept"]
    rows = []
    for name, s in manifest["sources"].items():
        rows.append([name, *(s[c] for c in cols), "FAILED" if s.get("error") else "ok"])
    out.append(_md_table(["source", *cols, "status"], rows))

    if manifest.get("errors"):
        out += ["", "## Errors", ""]
        out += [f"- {msg}" for msg in manifest["errors"].values()]
    if manifest.get("warnings"):
        out += ["", "## Warnings", ""]
        out += [f"- {w}" for w in manifest["warnings"]]

    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_gen: dict[str, Counter] = defaultdict(Counter)
    for c in manifest["counts"]:
        by_domain[c["domain"]][c["label"]] += c["count"]
        by_gen[c["generator"]][c["domain"]] += c["count"]
    out += ["", "## Label x domain", ""]
    rows = [
        [d, *(by_domain[d][lab] for lab in LABELS), sum(by_domain[d].values())]
        for d in sorted(by_domain)
    ]
    rows.append(
        ["**total**", *(sum(by_domain[d][lab] for d in by_domain) for lab in LABELS),
         manifest["n_records"]]
    )
    out.append(_md_table(["domain", *LABELS, "total"], rows))

    domains = sorted(by_domain)
    out += ["", "## Generator x domain", ""]
    rows = [
        [g, *(by_gen[g][d] for d in domains), sum(by_gen[g].values())] for g in sorted(by_gen)
    ]
    out.append(_md_table(["generator", *domains, "total"], rows))
    return "\n".join(out) + "\n"
