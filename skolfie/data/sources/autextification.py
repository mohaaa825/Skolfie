"""
skolfie/data/sources/autextification.py
---------------------------------------
Loader for AuTexTification 2023 (IberLEF), subtask 1 English: human vs
generated text in five domains (legal, news, reviews, tweets, wiki = WikiHow).
Train domains are legal, tweets and wiki; test domains are news and reviews.

Two on-disk layouts are read from `raw_dir/autextification/` (searched
recursively; English files only):

1. The original release (Zenodo), TSV or CSV with a header row:
       subtask_1/en/train.tsv, subtask_1/en/test.tsv
       subtask_2/en/train.tsv, subtask_2/en/test.tsv     (optional)
   Columns used: text, label ("human" / "generated"), and when present id,
   domain, model. In subtask 2 the label column holds the generator letter
   A-F.
2. The Hugging Face copy, symanto/autextification2023, configs `detection_en`
   (subtask 1) and optionally `attribution_en` (subtask 2), as Parquet (for
   example detection_en/train/0000.parquet from the auto-converted branch) or
   JSONL. Integer ClassLabel labels are decoded with the label names stored in
   the Parquet schema metadata written by `datasets`.

When both layouts are present only the original TSV/CSV files are used, so
nothing is loaded twice. A path counts as subtask 1 when it contains
"subtask_1"/"subtask1"/"detection", as subtask 2 with "subtask_2"/
"attribution", and as English when a path component or token is "en" or
"english".

Generators: subtask 1 rows may carry no model; then, if subtask 2 files are
present, generated texts are matched to them by exact text to recover the
letter. Letters map to models per the AuTexTification overview paper
(Sarvazyan et al. 2023): A-C are BLOOM-1B7, BLOOM-3B, BLOOM-7B1 and D-F are
GPT-3 babbage, curie and text-davinci-003 (see GENERATOR_LETTERS). Generated
texts with no recoverable model get generator "unknown".
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from skolfie.data.schema import Record
from skolfie.data.sources._common import (
    hf_snapshot,
    iter_jsonl,
    missing_files_error,
    pick,
    split_from_name,
)

log = logging.getLogger(__name__)

NAME = "autextification"
DESCRIPTION = (
    "AuTexTification 2023 subtask 1 (English): ~56k human and machine texts in legal, news, "
    "reviews, tweets and WikiHow domains; generators are three BLOOM and three GPT-3 models."
)
CITATION = (
    "Sarvazyan, A. M., Gonzalez, J. A., Franco-Salvador, M., Rangel, F., Chulvi, B., Rosso, P. "
    "(2023). Overview of AuTexTification at IberLEF 2023: Detection and Attribution of "
    "Machine-Generated Text in Multiple Domains. Procesamiento del Lenguaje Natural 71."
)
LICENSE_NOTE = (
    "Distributed by the organisers for research use (Zenodo record / Hugging Face dataset "
    "symanto/autextification2023, which may require accepting terms); check the licence on "
    "the download page before redistributing."
)

REPO_ID = "symanto/autextification2023"

MANUAL_DOWNLOAD = f"""\
AuTexTification manual download, either of:
 a) the original release linked from https://sites.google.com/view/autextification
    (Zenodo, search "AuTexTification"): unpack it so that these files exist:
      <raw_dir>/{NAME}/subtask_1/en/train.tsv
      <raw_dir>/{NAME}/subtask_1/en/test.tsv
      <raw_dir>/{NAME}/subtask_2/en/train.tsv   (optional: gives generator letters)
      <raw_dir>/{NAME}/subtask_2/en/test.tsv    (optional)
 b) the Hugging Face copy https://huggingface.co/datasets/{REPO_ID}
    (accept its terms if asked; then `huggingface-cli login`), configs detection_en and
    attribution_en, e.g. the Parquet files from
    https://huggingface.co/datasets/{REPO_ID}/tree/refs%2Fconvert%2Fparquet
    placed as <raw_dir>/{NAME}/detection_en/<split>/*.parquet and
    <raw_dir>/{NAME}/attribution_en/<split>/*.parquet.
"""

# Subtask 2 letters -> model (overview paper; ordered by family, then size).
GENERATOR_LETTERS = {
    "a": "bloom-1b7",
    "b": "bloom-3b",
    "c": "bloom-7b1",
    "d": "babbage",
    "e": "curie",
    "f": "text-davinci-003",
}

DOMAIN_MAP = {
    "tweets": "casual",
    "tweet": "casual",
    "reviews": "review",
    "review": "review",
    "news": "news",
    "legal": "legal",
    "wiki": "technical",  # WikiHow how-to articles
    "wikihow": "technical",
}

TEXT_KEYS = ("text",)
LABEL_KEYS = ("label",)
ID_KEYS = ("id",)
DOMAIN_KEYS = ("domain",)
MODEL_KEYS = ("model",)
HUMAN_LABELS = {"human"}
AI_LABELS = {"generated", "machine", "ai", "bot"}

_DATA_SUFFIXES = {".tsv", ".csv", ".parquet", ".jsonl"}
_ORIGINAL_SUFFIXES = {".tsv", ".csv"}
_DETECTION = re.compile(r"subtask[_-]?1|detection")
_ATTRIBUTION = re.compile(r"subtask[_-]?2|attribution")
_ENGLISH = re.compile(r"(^|[^a-z])(en|english)([^a-z]|$)")


def _classify(rel: str) -> str | None:
    """'detection' / 'attribution' for an English data file path, else None."""
    rel = rel.lower()
    if not _ENGLISH.search(rel):
        return None
    if _DETECTION.search(rel):
        return "detection"
    if _ATTRIBUTION.search(rel):
        return "attribution"
    return None


def _find(directory: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """(detection files, attribution files, all data files seen)."""
    if not directory.is_dir():
        return [], [], []
    seen = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES and not p.name.startswith(".")
    )
    found: dict[str, list[Path]] = {"detection": [], "attribution": []}
    for p in seen:
        kind = _classify(p.relative_to(directory).as_posix())
        if kind:
            found[kind].append(p)

    def prefer_original(paths: list[Path]) -> list[Path]:
        original = [p for p in paths if p.suffix.lower() in _ORIGINAL_SUFFIXES]
        return original or paths

    return prefer_original(found["detection"]), prefer_original(found["attribution"]), seen


def _parquet_label_names(path: Path) -> list[str] | None:
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    try:
        feature = json.loads(raw)["info"]["features"]["label"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    names = feature.get("names") if isinstance(feature, dict) else None
    return list(names) if names else None


def _read_rows(path: Path) -> Iterator[tuple[int, dict, list[str] | None]]:
    """Yield (row_index, row, ClassLabel names or None)."""
    suffix = path.suffix.lower()
    if suffix in _ORIGINAL_SUFFIXES:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with open(path, encoding="utf-8-sig", newline="") as fh:  # BOM would rename column 1
            reader = csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")
            for index, row in enumerate(reader):
                yield index, row, None
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        names = _parquet_label_names(path)
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            yield index, row, names
    else:
        for index, row in iter_jsonl(path):
            yield index, row, None


def _decode_label(value: object, names: list[str] | None, where: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if not names or not 0 <= value < len(names):
            raise ValueError(f"{where}: integer label {value} and no ClassLabel names to decode it")
        value = names[value]
    return str(value).strip()


def _download(directory: Path) -> None:
    patterns = ["*detection_en*", "*attribution_en*"]
    errors = []
    for revision in (None, "refs/convert/parquet"):
        try:
            hf_snapshot(REPO_ID, directory, patterns, revision=revision)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if _find(directory)[0]:
            return
    detail = errors[0] if errors else f"no detection_en files in https://huggingface.co/datasets/{REPO_ID}"
    raise RuntimeError(f"{detail}\n{MANUAL_DOWNLOAD}")


def _attribution_map(paths: list[Path]) -> dict[str, str]:
    """Exact text -> generator letter (lowercase) from subtask 2 files."""
    mapping: dict[str, str] = {}
    for path in paths:
        for index, row, names in _read_rows(path):
            where = f"{path}:row {index}"
            text = pick(row, TEXT_KEYS, where, "text")
            letter = _decode_label(pick(row, LABEL_KEYS, where, "label"), names, where).lower()
            if isinstance(text, str) and letter in GENERATOR_LETTERS:
                mapping[text.strip()] = letter
    return mapping


def _generator(row: dict, text: str, letters: dict[str, str], where: str) -> str:
    model = pick(row, MODEL_KEYS, where, "model", default=None)
    if model is not None:
        model = str(model).strip().lower()
        if model in GENERATOR_LETTERS:
            return GENERATOR_LETTERS[model]
        if model and model not in {"human", "none", "nan"}:
            return model
    letter = letters.get(text)
    return GENERATOR_LETTERS[letter] if letter else "unknown"


def load(raw_dir: Path, download: bool = True) -> Iterator[Record]:
    """Yield English subtask-1 Records from `raw_dir/autextification/`."""
    directory = Path(raw_dir) / NAME
    detection, attribution, seen = _find(directory)
    if not detection and download:
        _download(directory)
        detection, attribution, seen = _find(directory)
    if not detection:
        err = missing_files_error(
            NAME, directory, "subtask_1/en/{train,test}.tsv or detection_en/<split>/*.parquet",
            MANUAL_DOWNLOAD,
        )
        if seen:
            listed = ", ".join(p.relative_to(directory).as_posix() for p in seen[:20])
            err = FileNotFoundError(f"{err}\nData files seen but not recognised: {listed}")
        raise err

    letters = _attribution_map(attribution) if attribution else {}

    for path in detection:
        rel = path.relative_to(directory).as_posix()
        split = split_from_name(rel)
        rel_stem = path.relative_to(directory).with_suffix("").as_posix()
        for index, row, names in _read_rows(path):
            where = f"{path}:row {index}"
            text = pick(row, TEXT_KEYS, where, "text")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            raw_label = _decode_label(pick(row, LABEL_KEYS, where, "label"), names, where)
            if raw_label.lower() in HUMAN_LABELS:
                label, generator = "human", "human"
            elif raw_label.lower() in AI_LABELS:
                label, generator = "ai", _generator(row, text, letters, where)
            else:
                raise ValueError(
                    f"{where}: unknown label {raw_label!r}; expected one of "
                    f"{sorted(HUMAN_LABELS | AI_LABELS)}"
                )
            subdomain = str(pick(row, DOMAIN_KEYS, where, "domain", default="")).strip().lower()
            domain = DOMAIN_MAP.get(subdomain, "other")
            # The source's own id is unique within a split; without one, fall back to
            # the row position within this file.
            local_id = pick(row, ID_KEYS, where, "id", default=None)
            if local_id is None or str(local_id).strip() == "":
                record_id = f"{NAME}:{rel_stem}:row{index}"
            else:
                record_id = f"{NAME}:{split or rel_stem}:{local_id}"
            yield Record(
                id=record_id, text=text, label=label, generator=generator,
                domain=domain, subdomain=subdomain, source=NAME, split_hint=split,
            )
