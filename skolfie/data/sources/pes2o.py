"""
skolfie/data/sources/pes2o.py
-----------------------------
Loader for peS2o (allenai/peS2o on the Hugging Face Hub; ODC-BY licence):
cleaned full texts of open-access scientific papers from S2ORC, written by
humans before ChatGPT. The detector's AI essays are 800-1,500 words long while
most human student essays (PERSUADE) are shorter, so without long human texts
a model can learn "long text = AI". This source supplies long human academic
prose: one 800-1,500 word section per paper.

On disk: `raw_dir/pes2o/` holding `*.jsonl`, `*.json.gz`, `*.jsonl.gz` or
`*.parquet` files (searched recursively, so the Hub's `data/v2/...` layout
works), one paper per row with at least `id` and `text`, optionally `created`
(a date string such as "2019-03-01" or "2019-03-01T00:00:00.000Z"), `source`
("s2orc/..." for full texts, "s2ag/..." for title + abstract) and `version`.
The text's paragraphs are separated by blank lines (the first one is
usually the title).

Download (`download=True` and no files): `huggingface_hub` (`pip install -e
".[data]"`) lists the dataset repo's files, and shards are chosen
deterministically: the newest version (data/vN) that has `validation-*`
shards in a readable format (json.gz / jsonl / parquet; v3 is .zst and is
skipped), else the first `train-*` shards of that version. Shards are
downloaded one at a time, in name order, into `raw_dir/pes2o/`, until the
candidate pool is large enough (see below). In v2 the validation shards are
about 60 MB and 500 MB. A failed listing or download raises RuntimeError
naming the repo.

Selection, per paper:
  * `created` year must be < `before_year` (default 2022, so before ChatGPT);
    a paper with no `created` date is kept. s2ag rows (abstracts only) are
    skipped.
  * The text is split into paragraphs on blank lines. Paragraphs that look
    like reference lists, tables or numeric junk (`looks_like_junk`) end the
    current run; short heading-like paragraphs (`looks_like_heading`) are
    skipped without ending it.
  * Sections are built by joining consecutive paragraphs until the word count
    reaches [min_words, max_words]; a paragraph that would push a section over
    `max_words` ends it, and sections under `min_words` are dropped
    (`make_sections`). A single paragraph longer than `max_words` is dropped.
  * Sections must look English (`looks_english`: mostly ASCII letters and a
    high share of common English stop words).
  * One section per paper, chosen with an RNG seeded by (seed, paper id), so
    every kept text comes from a different paper and the choice does not
    depend on file order.

Papers are read in file order until `POOL_FACTOR * max_texts` papers have a
usable section; then `max_texts` of them are sampled with `random.Random(seed)`
(max_texts=0 or None keeps the whole pool, i.e. reads every file). The same
files and seed always give the same records.

Records: label/generator "human", domain "academic", subdomain
"research_paper", source "pes2o", id `pes2o:<paper id>:s<k>` where k is the
index of the chosen section among the paper's candidate sections.

`make_sections`, `looks_like_junk`, `looks_like_heading` and `looks_english`
are reused by `skolfie.data.sources.theses`.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from skolfie.data.schema import Record
from skolfie.data.sources._common import _hf, hf_download_file, missing_files_error

log = logging.getLogger(__name__)

NAME = "pes2o"
DESCRIPTION = (
    "peS2o: cleaned full texts of open-access research papers (S2ORC); one 800-1,500 word "
    "section per pre-2022 paper (human only)."
)
CITATION = (
    "Soldaini, L., Lo, K. (2023). peS2o (Pretraining Efficiently on S2ORC) Dataset. "
    "Allen Institute for AI. https://huggingface.co/datasets/allenai/peS2o"
)
LICENSE_NOTE = "ODC-BY 1.0 (https://opendatacommons.org/licenses/by/1-0/)."

REPO_ID = "allenai/peS2o"
URL = f"https://huggingface.co/datasets/{REPO_ID}"

MANUAL_DOWNLOAD = f"""\
peS2o manual download: get one or more shards from {URL} (e.g.
data/v2/validation-00000-of-00002.json.gz) and put them in <raw_dir>/{NAME}/
(.json.gz, .jsonl, .jsonl.gz or .parquet; rows need `id` and `text`).
"""

DEFAULT_MAX_TEXTS = 3000
#: Papers with a usable section to collect before sampling max_texts of them.
POOL_FACTOR = 3

_READABLE = (".json.gz", ".jsonl.gz", ".jsonl", ".json", ".parquet")

# --------------------------------------------------------------------------- #
# Text heuristics (shared with theses.py)
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    "the of and to in a is that for on with as by are this be from at an or which it "
    "was were we not have has can these their its also between more than our been".split()
)
_WORD = re.compile(r"[A-Za-z]+")
_ET_AL = re.compile(r"\bet\s+al\b\.?", re.IGNORECASE)
_YEAR_PAREN = re.compile(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)|\b(?:19|20)\d{2}[a-z]?[.;,)]")
_REF_START = re.compile(r"^\s*(?:\[\d+\]|\d+\.\s+[A-Z][a-z]+,|[A-Z][a-z]+,\s+[A-Z]\.)")


def word_count(text: str) -> int:
    return len(text.split())


def looks_like_heading(paragraph: str) -> bool:
    """A short line without final punctuation: a title or section heading."""
    text = paragraph.strip()
    return word_count(text) < 12 and not text.endswith((".", "?", "!", ":", ";", '"', "”"))


def looks_like_junk(paragraph: str) -> bool:
    """A reference list, table, equation block or other non-prose paragraph."""
    text = paragraph.strip()
    if not text:
        return True
    n_words = word_count(text)
    chars = [c for c in text if not c.isspace()]
    digits = sum(c.isdigit() for c in chars)
    if chars and digits / len(chars) > 0.15:
        return True
    letters = sum(c.isalpha() for c in chars)
    if chars and letters / len(chars) < 0.65:
        return True
    if len(_ET_AL.findall(text)) >= max(2, n_words / 40):
        return True
    if len(_YEAR_PAREN.findall(text)) >= max(3, n_words / 15):
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        if sum(len(ln) for ln in lines) / len(lines) < 25:
            return True
        if sum(bool(_REF_START.match(ln)) for ln in lines) >= len(lines) / 2:
            return True
    return False


def looks_english(text: str, min_ascii: float = 0.97, min_stop: float = 0.2) -> bool:
    """Mostly ASCII letters and a high share of common English stop words."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isascii() for c in letters) / len(letters) < min_ascii:
        return False
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return False
    return sum(w in _STOPWORDS for w in words) / len(words) >= min_stop


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs separated by blank lines, inner whitespace collapsed."""
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]


def make_sections(paragraphs: Iterable[str], min_words: int, max_words: int) -> list[str]:
    """Runs of consecutive prose paragraphs of [min_words, max_words] words.

    Paragraphs are added greedily; the one that would push a section over
    `max_words` ends it (and starts the next one). Junk paragraphs end the run,
    headings are skipped, a paragraph longer than `max_words` on its own is
    dropped (and ends the run), and runs under `min_words` are dropped.
    """
    if min_words < 1 or max_words < min_words:
        raise ValueError(f"need 1 <= min_words <= max_words, got {min_words}, {max_words}")
    sections: list[str] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal current, count
        if current and min_words <= count <= max_words:
            sections.append("\n\n".join(current))
        current, count = [], 0

    for raw in paragraphs:
        paragraph = raw.strip()
        if not paragraph:
            continue
        if looks_like_junk(paragraph):
            flush()
            continue
        if looks_like_heading(paragraph):
            continue
        n = word_count(paragraph)
        if n > max_words:
            flush()
            continue
        if count + n > max_words:
            flush()
        current.append(paragraph)
        count += n
    flush()
    return sections


def year_of(value: object) -> int | None:
    """The first plausible 4-digit year in a date value, else None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        year = int(value)
        return year if 1000 <= year <= 2999 else None
    match = re.search(r"(?<!\d)(1[5-9]\d{2}|2\d{3})(?!\d)", str(value))
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def _is_readable(name: str) -> bool:
    return name.lower().endswith(_READABLE)


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and _is_readable(p.name) and not p.name.startswith(".")
                  and ".cache" not in p.parts)


def _iter_rows(path: Path) -> Iterator[dict]:
    name = path.name.lower()
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
        return
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("pes2o: %s:%d: invalid JSON line skipped", path, lineno)
                continue
            if isinstance(row, dict):
                yield row


def choose_shards(files: list[str]) -> list[str]:
    """Readable shards to download, in order (see the module docstring)."""
    by_version: dict[str, list[str]] = {}
    for f in files:
        match = re.match(r"^data/(v\d+)/[^/]+$", f)
        if match and _is_readable(f):
            by_version.setdefault(match.group(1), []).append(f)
    if not by_version:
        others = sorted(f for f in files if _is_readable(f) and not f.startswith("."))
        return others
    for version in sorted(by_version, key=lambda v: int(v[1:]), reverse=True):
        names = sorted(by_version[version])
        valid = [f for f in names if re.search(r"/(validation|valid|val)[-_.]", f)]
        if valid:
            return valid
        train = [f for f in names if "/train" in f]
        if train:
            return train
    return []


def _list_repo_files() -> list[str]:
    hub = _hf()
    try:
        return list(hub.HfApi().list_repo_files(REPO_ID, repo_type="dataset"))
    except Exception as exc:
        raise RuntimeError(f"could not list the files of {URL}: {type(exc).__name__}: {exc}\n"
                           f"{MANUAL_DOWNLOAD}") from exc


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def paper_section(row: dict, seed: int, min_words: int, max_words: int,
                  before_year: int | None) -> tuple[str, int, str] | None:
    """(paper id, section index, section text) for one row, or None if it has none."""
    paper_id = str(row.get("id") or "").strip()
    text = row.get("text")
    if not paper_id or not isinstance(text, str) or not text.strip():
        return None
    if str(row.get("source") or "").lower().startswith("s2ag"):
        return None
    year = year_of(row.get("created"))
    if before_year is not None and year is not None and year >= before_year:
        return None
    if word_count(text) < min_words:
        return None
    sections = [s for s in make_sections(split_paragraphs(text), min_words, max_words)
                if looks_english(s)]
    if not sections:
        return None
    k = random.Random(f"{seed}:{paper_id}").randrange(len(sections))
    return paper_id, k, sections[k]


def _collect(paths: list[Path], pool: dict[str, tuple[int, str]], limit: int | None,
             seed: int, min_words: int, max_words: int, before_year: int | None) -> bool:
    """Add papers with a usable section to `pool`; True once `limit` is reached."""
    for path in paths:
        n_rows = 0
        for row in _iter_rows(path):
            n_rows += 1
            got = paper_section(row, seed, min_words, max_words, before_year)
            if got is None or got[0] in pool:
                continue
            pool[got[0]] = (got[1], got[2])
            if limit and len(pool) >= limit:
                log.info("pes2o: %s: pool full (%d papers) after %d rows", path, limit, n_rows)
                return True
        log.info("pes2o: %s: %d rows, %d usable papers so far", path, n_rows, len(pool))
    return False


def load(raw_dir: Path, download: bool = True, max_texts: int | None = DEFAULT_MAX_TEXTS,
         seed: int = 0, min_words: int = 800, max_words: int = 1500,
         before_year: int | None = 2022) -> Iterator[Record]:
    """Yield human research-paper sections from `raw_dir/pes2o/` (see the module docstring)."""
    if max_texts is not None and max_texts < 0:
        raise ValueError(f"max_texts must be >= 0 (0 keeps all), got {max_texts}")
    if min_words < 1 or max_words < min_words:
        raise ValueError(f"need 1 <= min_words <= max_words, got {min_words}, {max_words}")
    directory = Path(raw_dir) / NAME
    limit = POOL_FACTOR * max_texts if max_texts else None
    pool: dict[str, tuple[int, str]] = {}
    paths = _files(directory)
    if paths:
        _collect(paths, pool, limit, seed, min_words, max_words, before_year)
    elif download:
        shards = choose_shards(_list_repo_files())
        if not shards:
            raise RuntimeError(f"no readable shard (json.gz/jsonl/parquet) found in {URL}\n"
                               f"{MANUAL_DOWNLOAD}")
        for shard in shards:
            log.info("pes2o: downloading %s from %s", shard, URL)
            path = hf_download_file(REPO_ID, shard, directory)
            if _collect([path], pool, limit, seed, min_words, max_words, before_year):
                break
    else:
        raise missing_files_error(NAME, directory, "peS2o .json.gz/.jsonl/.parquet shards",
                                  MANUAL_DOWNLOAD)

    chosen = sorted(pool)
    if max_texts and len(chosen) > max_texts:
        chosen = sorted(random.Random(seed).sample(chosen, max_texts))
    if max_texts and len(chosen) < max_texts:
        log.warning("pes2o: only %d papers with a usable section (wanted %d)", len(chosen),
                    max_texts)
    for paper_id in chosen:
        k, text = pool[paper_id]
        yield Record(id=f"{NAME}:{paper_id}:s{k}", text=text, label="human", generator="human",
                     domain="academic", subdomain="research_paper", source=NAME)
