"""
skolfie/data/fairness_sets.py
-----------------------------
External test sets for the fairness check: how often human text from
different groups of writers (especially non-native English writers) is
wrongly flagged as AI.

Two public sources, downloaded from raw.githubusercontent.com and cached:

    liang2023  Liang et al. (2023), "GPT detectors are biased against
               non-native English writers". Small sets of human essays and
               abstracts (TOEFL, US college admission essays, US grade-8
               essays, Stanford CS224N abstracts), two AI-edited versions of
               the human sets, and ChatGPT-3.5 essays/abstracts on the same
               prompts. Each set is one folder with `data.json`, a list of
               {"document": str, ...}; a "prompt" field, where present, is
               the instruction sent to the model and is not part of the text.
    ellipse    ELLIPSE (Crossley et al. 2023): 3,911 essays by English
               language learners in US grades 8-12, scored for holistic
               proficiency (`Overall`, 1-5). Only the public train CSV is
               used (the test split is a password-protected zip).

`build_fairness_sets()` returns one DataFrame with the corpus columns
(`skolfie.data.schema.COLUMNS`) plus `FAIRNESS_EXTRA_COLUMNS`:

    set           one of FAIRNESS_SETS (e.g. "toefl_real", "ellipse")
    group         human-readable group name for reports
    writer        "non-native" | "native" | "unknown" | "ai"
    edited_by_ai  True for human essays reworded by GPT-4 (toefl_gpt4polished,
                  hewlett_gptsimplify). They keep label "human" but must be
                  reported on their own, never as plain human false positives.
    proficiency   ELLIPSE `Overall` score (NaN elsewhere)
    grade         ELLIPSE school grade 8-12 (NaN elsewhere)

Text goes through `clean_text` (paragraph breaks kept), after three source
repairs: Windows-1252 bytes that were decoded as C1 control characters
(U+0080-U+009F, e.g. "don\\x92t" in the Hewlett essays) are mapped back to
the characters they stand for, instead of being deleted by `clean_text`;
soft hyphens are removed; and the "Title: ...\\nAbstract:" header of the
human CS224N abstracts is cut so they match the AI abstracts, which have
none. Anonymisation placeholders (Hewlett "@CAPS1", ELLIPSE "Generic_Name",
"STUDENT_NAME") are left as they are: they are what the source contains.
Empty texts are dropped, then exact duplicates (same `normalized_hash`, as in
the corpus; the first kept), each with a count in the log and in
`df.attrs["build_stats"]`.

`drop_corpus_overlap()` removes rows whose `normalized_hash` matches a text
of a built corpus, so nothing trained or evaluated on leaks into these sets.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from skolfie.data.corpus import clean_text, normalized_hash
from skolfie.data.schema import COLUMNS, DOMAINS, Record
from skolfie.eval.report import atomic_path

FAIRNESS_EXTRA_COLUMNS = ["set", "group", "writer", "edited_by_ai", "proficiency", "grade"]
FAIRNESS_COLUMNS = [*COLUMNS, *FAIRNESS_EXTRA_COLUMNS]
WRITERS = ("non-native", "native", "unknown", "ai")

# Download behaviour: per-request timeout (seconds), total attempts, and the
# base of the exponential backoff between attempts (1 s, 2 s, ...).
TIMEOUT = 60.0
ATTEMPTS = 3
BACKOFF = 1.0
USER_AGENT = "skolfie-fairness-sets/1"

LIANG_BASE_URL = (
    "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/Data_and_Results/"
)
ELLIPSE_URL = (
    "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/"
    "ELLIPSE_Final_github_train.csv"
)


@dataclass(frozen=True)
class FairnessSource:
    name: str
    title: str
    url: str
    license: str
    citation: str
    description: str


SOURCES: dict[str, FairnessSource] = {
    "liang2023": FairnessSource(
        name="liang2023",
        title="GPT detectors are biased against non-native English writers",
        url="https://github.com/Weixin-Liang/ChatGPT-Detector-Bias",
        # The repository has no LICENSE file; MIT is stated by its README badge.
        # The essays themselves were collected from public web pages and
        # datasets (TOEFL practice essays, college admission essays, the
        # Hewlett ASAP essays, CS224N project reports) by the authors.
        license="MIT (per the repository README); underlying texts collected by the authors "
                "from public sources",
        citation="Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., Zou, J. (2023). GPT detectors "
                 "are biased against non-native English writers. Patterns 4. "
                 "arXiv:2304.02819.",
        description="91 TOEFL essays by non-native writers (and the same essays word-polished "
                    "by GPT-4), 70 US college admission essays, 88 US grade-8 essays (and "
                    "GPT-4-simplified versions), 145 Stanford CS224N abstracts, and "
                    "ChatGPT-3.5 college essays and abstracts with and without a self-edit "
                    "prompt.",
    ),
    "ellipse": FairnessSource(
        name="ellipse",
        title="English Language Learner Insight, Proficiency and Skills Evaluation corpus",
        url="https://github.com/scrosseye/ELLIPSE-Corpus",
        license="CC BY 4.0",
        citation="Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., "
                 "Benner, B., Picou, A., Boser, U. (2023). Measuring second language "
                 "proficiency using the English Language Learner Insight, Proficiency and "
                 "Skills Evaluation (ELLIPSE) Corpus. International Journal of Learner Corpus "
                 "Research 9(2), 248-269.",
        description="3,911 argumentative essays (train split) by English language learners in "
                    "US grades 8-12, with holistic proficiency scores (1-5).",
    ),
}
FAIRNESS_SOURCES = tuple(SOURCES)


@dataclass(frozen=True)
class FairnessSet:
    name: str
    source: str
    path: str          # file path below the source's base URL (and below cache_dir/<source>/)
    label: str
    generator: str
    writer: str
    edited_by_ai: bool
    group: str
    subdomain: str
    # Every set is student writing: essays, and CS224N abstracts from student
    # project reports. The corpus maps student essays (M4GT's OUTFOX) and
    # paper abstracts to "academic", so these do too; "other" would put them
    # in a domain the detector never saw student writing in.
    domain: str = "academic"


# ChatGPT-3.5 per the repository README ("ChatGPT-3.5 generated college
# admission essays / scientific abstracts"). The folders say "gpt3" and
# name.json says "GPT3", except CS224N_gpt3PromptEng_145 whose name.json says
# "GPT4" while its folder and display name say GPT-3; the README is followed.
_CHATGPT35 = "chatgpt-3.5"


def _liang(name: str, folder: str, **kw: Any) -> FairnessSet:
    return FairnessSet(name=name, source="liang2023", path=f"{folder}/data.json", **kw)


FAIRNESS_SETS: dict[str, FairnessSet] = {
    s.name: s
    for s in (
        _liang("toefl_real", "Human_Data/TOEFL_real_91", label="human", generator="human",
               writer="non-native", edited_by_ai=False,
               group="non-native: TOEFL essays", subdomain="toefl_essay"),
        # The same 91 essays, reworded by GPT-4 ("Enhance the word choices to
        # sound more like that of a native speaker").
        _liang("toefl_gpt4polished", "Human_Data/TOEFL_gpt4polished_91", label="human",
               generator="human", writer="non-native", edited_by_ai=True,
               group="non-native, GPT-4-polished: TOEFL essays", subdomain="toefl_essay"),
        _liang("college_real", "Human_Data/CollegeEssay_real_70", label="human",
               generator="human", writer="native", edited_by_ai=False,
               group="native: US college essays", subdomain="college_essay"),
        # Stanford students' project abstracts: native and non-native writers mixed.
        _liang("cs224n_real", "Human_Data/CS224N_real_145", label="human", generator="human",
               writer="unknown", edited_by_ai=False,
               group="Stanford CS224N abstracts", subdomain="cs224n_abstract"),
        # Hewlett ASAP essays by US 8th graders; the paper treats them as native.
        _liang("hewlett_real", "Human_Data/HewlettStudentEssay_real_88", label="human",
               generator="human", writer="native", edited_by_ai=False,
               group="native: US grade-8 essays", subdomain="student_essay"),
        # The same 88 essays, reworded by GPT-4 ("Simplify word choices as if
        # written by a non-native speaker").
        _liang("hewlett_gptsimplify", "Human_Data/HewlettStudentEssay_GPTsimplify_88",
               label="human", generator="human", writer="native", edited_by_ai=True,
               group="native, GPT-4-simplified: US grade-8 essays", subdomain="student_essay"),
        _liang("college_gpt3", "GPT_Data/CollegeEssay_gpt3_31", label="ai",
               generator=_CHATGPT35, writer="ai", edited_by_ai=False,
               group="AI: ChatGPT-3.5 college essays", subdomain="college_essay"),
        # Second round: "Elevate the provided text by employing literary language".
        _liang("college_gpt3prompteng", "GPT_Data/CollegeEssay_gpt3PromptEng_31", label="ai",
               generator=_CHATGPT35, writer="ai", edited_by_ai=False,
               group="AI: ChatGPT-3.5 college essays, self-edited",
               subdomain="college_essay"),
        _liang("cs224n_gpt3", "GPT_Data/CS224N_gpt3_145", label="ai", generator=_CHATGPT35,
               writer="ai", edited_by_ai=False,
               group="AI: ChatGPT-3.5 CS224N abstracts", subdomain="cs224n_abstract"),
        # Second round: "Elevate the provided text by employing advanced technical language".
        _liang("cs224n_gpt3prompteng", "GPT_Data/CS224N_gpt3PromptEng_145", label="ai",
               generator=_CHATGPT35, writer="ai", edited_by_ai=False,
               group="AI: ChatGPT-3.5 CS224N abstracts, self-edited",
               subdomain="cs224n_abstract"),
        FairnessSet(name="ellipse", source="ellipse", path="ELLIPSE_Final_github_train.csv",
                    label="human", generator="human", writer="non-native", edited_by_ai=False,
                    group="non-native: ELLIPSE learner essays", subdomain="learner_essay"),
    )
}

for _s in FAIRNESS_SETS.values():  # catch a typo in the table at import time
    assert _s.source in SOURCES and _s.writer in WRITERS and _s.domain in DOMAINS, _s
    assert (_s.label == "ai") == (_s.writer == "ai") and not (_s.edited_by_ai and _s.label == "ai")


def source_url(fset: FairnessSet) -> str:
    """The URL `fset`'s file is downloaded from."""
    if fset.source == "liang2023":
        return LIANG_BASE_URL + fset.path
    return ELLIPSE_URL


# --------------------------------------------------------------------------- #
# Download with cache
# --------------------------------------------------------------------------- #


class FairnessDownloadError(RuntimeError):
    """A fairness-set file could not be downloaded (or is missing with download=False)."""


_sleep = time.sleep  # tests replace this to skip the backoff


def download(url: str, dest: Path, *, timeout: float = TIMEOUT, attempts: int = ATTEMPTS,
             backoff: float = BACKOFF) -> Path:
    """Fetch `url` into `dest` (atomically); FairnessDownloadError naming the URL on failure.

    Network errors, timeouts and HTTP 5xx / 429 are retried up to `attempts`
    times in all, waiting backoff * 2**k seconds in between; other HTTP
    errors (404, 403) are not retried since a retry cannot fix them.
    """
    last: BaseException | None = None
    for attempt in range(max(1, attempts)):
        if attempt:
            _sleep(backoff * 2 ** (attempt - 1))
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        else:
            with atomic_path(dest) as tmp:
                tmp.write_bytes(payload)
            return dest
    tries = attempt + 1
    raise FairnessDownloadError(
        f"could not download {url} ({tries} attempt{'s' * (tries > 1)}): "
        f"{type(last).__name__}: {last}"
    ) from last


def fetch(fset: FairnessSet, cache_dir: Path, *, download_missing: bool = True) -> Path:
    """Local path of `fset`'s file under cache_dir/<source>/, downloading it if missing."""
    path = Path(cache_dir) / fset.source / fset.path
    if path.is_file():
        return path
    url = source_url(fset)
    if not download_missing:
        raise FairnessDownloadError(f"{path} is missing and downloading is off; get it from {url}")
    # Looked up at call time so tests can monkeypatch `download`.
    return download(url, path)


# --------------------------------------------------------------------------- #
# Text repairs
# --------------------------------------------------------------------------- #

# U+0080-U+009F are C1 controls that `clean_text` deletes; in these sources
# they are Windows-1252 bytes decoded as Latin-1 ("\x92" is a right quote).
# Bytes undefined in cp1252 (0x81, 0x8D, 0x8F, 0x90, 0x9D) stay controls.
_C1_RE = re.compile("[\x80-\x9f]")
_SOFT_HYPHEN = "\xad"
# "Title: <title>\nAbstract: <abstract>" in the human CS224N set.
_CS224N_HEADER_RE = re.compile(r"\A\s*Title:[^\n]*\n\s*Abstract:\s*")
# ELLIPSE: "Ê" between words is a Mac Roman non-breaking space read as Latin-1
# ("findÊmore"); a lone "â" before a space is what is left of a UTF-8 right
# quote ("itâ going"). Seven occurrences in the train CSV, all of this kind.
_ELLIPSE_REPAIRS = ((re.compile("\u00ca"), " "), (re.compile(r"(?<=[A-Za-z])\u00e2(?=\s)"), "'"))


def _cp1252(match: re.Match[str]) -> str:
    try:
        return bytes([ord(match.group(0))]).decode("cp1252")
    except UnicodeDecodeError:
        return match.group(0)


def repair_text(text: Any, fset: FairnessSet) -> str:
    """Source-specific repairs (module docstring), then `clean_text`."""
    if not isinstance(text, str):
        return ""
    text = _C1_RE.sub(_cp1252, text).replace(_SOFT_HYPHEN, "")
    if fset.name == "cs224n_real":
        text = _CS224N_HEADER_RE.sub("", text, count=1)
    if fset.source == "ellipse":
        for pattern, repl in _ELLIPSE_REPAIRS:
            text = pattern.sub(repl, text)
    return clean_text(text)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _row(fset: FairnessSet, local_id: str, text: str, proficiency: float = float("nan"),
         grade: float = float("nan")) -> dict[str, Any]:
    record = Record(  # validates label / domain / generator
        id=f"{fset.source}:{fset.name}:{local_id}", text=text, label=fset.label,
        generator=fset.generator, domain=fset.domain, subdomain=fset.subdomain,
        source=fset.source,
    )
    return {
        **record.to_dict(), "set": fset.name, "group": fset.group, "writer": fset.writer,
        "edited_by_ai": fset.edited_by_ai, "proficiency": proficiency, "grade": grade,
    }


def _load_liang(fset: FairnessSet, path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    rows = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "document" not in item:
            found = sorted(item) if isinstance(item, dict) else type(item).__name__
            raise ValueError(f"{path}[{index}]: expected an object with 'document', got {found}")
        # The index in data.json is the only stable id these files have.
        rows.append(_row(fset, str(index), repair_text(item["document"], fset)))
    return rows


ELLIPSE_REQUIRED = ("text_id_kaggle", "full_text", "Overall", "grade")


def _load_ellipse(fset: FairnessSet, path: Path) -> list[dict[str, Any]]:
    # Ids as strings: some Kaggle ids are all digits and would become ints.
    df = pd.read_csv(path, dtype={"text_id_kaggle": str})
    missing = [c for c in ELLIPSE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    overall = pd.to_numeric(df["Overall"], errors="coerce")
    grade = pd.to_numeric(df["grade"], errors="coerce")
    return [
        _row(fset, str(tid), repair_text(text, fset), float(o), float(g))
        for tid, text, o, g in zip(df["text_id_kaggle"], df["full_text"], overall, grade)
    ]


_DTYPES = {"n_words": "int64", "edited_by_ai": "bool", "proficiency": "float64",
           "grade": "float64"}


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FAIRNESS_COLUMNS)
    str_cols = [c for c in FAIRNESS_COLUMNS if c not in _DTYPES]
    return df.astype({**{c: str for c in str_cols}, **_DTYPES})


def sets_for(sources: Iterable[str]) -> list[FairnessSet]:
    """The FAIRNESS_SETS of `sources`, in table order; ValueError on an unknown source."""
    sources = list(sources)
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise ValueError(f"unknown fairness source(s) {unknown}; known: {list(SOURCES)}")
    return [s for s in FAIRNESS_SETS.values() if s.source in sources]


def build_fairness_sets(
    cache_dir: str | Path,
    *,
    sources: Sequence[str] = FAIRNESS_SOURCES,
    log: Callable[[str], Any] = print,
    download: bool = True,
) -> pd.DataFrame:
    """Load the fairness test sets of `sources` into one DataFrame (FAIRNESS_COLUMNS).

    Files are downloaded into cache_dir/<source>/ on first use and reused
    afterwards. Rows keep the FAIRNESS_SETS order, then the source order.
    `df.attrs["build_stats"]` holds the raw / empty / duplicate counts per set.
    """
    cache_dir = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    raw: Counter[str] = Counter()
    for fset in sets_for(sources):
        path = fetch(fset, cache_dir, download_missing=download)
        loader = _load_liang if fset.source == "liang2023" else _load_ellipse
        loaded = loader(fset, path)
        raw[fset.name] = len(loaded)
        rows.extend(loaded)

    df = _frame(rows)
    empty = df["text"].str.len() == 0
    empty_by_set = Counter(df.loc[empty, "set"])
    df = df[~empty]
    # Same notion of "exact duplicate" as the corpus: equal `normalized_hash`
    # (case, punctuation and whitespace ignored). TOEFL_real has one essay
    # twice, differing only in a "." vs ",".
    dup = df["text"].map(normalized_hash).duplicated(keep="first")
    dup_by_set = Counter(df.loc[dup, "set"])
    df = df[~dup].reset_index(drop=True)

    for name, n in raw.items():
        log(f"fairness: {name}: {n} texts read")
    if empty_by_set:
        log(f"fairness: dropped {sum(empty_by_set.values())} empty texts {dict(empty_by_set)}")
    log(f"fairness: dropped {sum(dup_by_set.values())} exact duplicates"
        + (f" {dict(dup_by_set)}" if dup_by_set else ""))
    log(f"fairness: {len(df)} texts in {df['set'].nunique()} sets")
    df.attrs["build_stats"] = {
        "raw": dict(raw), "empty": dict(empty_by_set), "duplicate": dict(dup_by_set),
    }
    return df


# --------------------------------------------------------------------------- #
# Corpus overlap and manifest
# --------------------------------------------------------------------------- #


def corpus_hashes(corpus_path: str | Path, batch_size: int = 50_000) -> set[bytes]:
    """`normalized_hash` of every text in a corpus Parquet file, read in batches."""
    import pyarrow.parquet as pq

    hashes: set[bytes] = set()
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=batch_size,
                                                          columns=["text"]):
        hashes.update(normalized_hash(t) for t in batch.column(0).to_pylist() if t)
    return hashes


def drop_corpus_overlap(
    df: pd.DataFrame, hashes: set[bytes]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows of `df` whose normalised text is not in `hashes`, and the drop count per set."""
    hit = df["text"].map(lambda t: normalized_hash(t) in hashes).astype(bool)
    dropped = {str(k): int(v) for k, v in Counter(df.loc[hit, "set"]).items()}
    out = df[~hit].reset_index(drop=True)
    out.attrs = dict(df.attrs)
    return out, dropped


def _count(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()}


def manifest(df: pd.DataFrame, *, corpus: str | Path | None = None,
             corpus_dropped: dict[str, int] | None = None) -> dict[str, Any]:
    """JSON-ready summary: counts per set / label / writer, sources, drops, timestamp."""
    by_set = {}
    for name in [n for n in FAIRNESS_SETS if n in set(df["set"])]:
        part = df[df["set"] == name]
        fset = FAIRNESS_SETS[name]
        by_set[name] = {
            "n": int(len(part)), "source": fset.source, "label": fset.label,
            "writer": fset.writer, "edited_by_ai": fset.edited_by_ai, "group": fset.group,
            "generator": fset.generator, "url": source_url(fset),
            "median_words": int(part["n_words"].median()),
        }
    used = sorted(set(df["source"]))
    corpus_dropped = corpus_dropped or {}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "by_set": by_set,
        "by_label": _count(df, "label"),
        "by_writer": _count(df, "writer"),
        "by_label_writer": {
            f"{label}/{writer}": int(n)
            for (label, writer), n in df.groupby(["label", "writer"]).size().items()
        },
        "edited_by_ai": int(df["edited_by_ai"].sum()),
        "sources": {
            name: {"title": SOURCES[name].title, "url": SOURCES[name].url,
                   "license": SOURCES[name].license, "citation": SOURCES[name].citation}
            for name in used
        },
        "build_stats": df.attrs.get("build_stats", {}),
        "corpus_overlap": {
            "corpus": str(corpus) if corpus is not None else None,
            "dropped": int(sum(corpus_dropped.values())),
            "dropped_by_set": corpus_dropped,
        },
    }
