"""
skolfie/data/sources/persuade.py
--------------------------------
Loader for PERSUADE 2.0: about 25,000 argumentative essays written by US
students in grades 6-12 for 15 assignments (Crossley et al. 2024; CC BY 4.0;
https://github.com/scrosseye/persuade_corpus_2.0). These are the human
student essays the detector is meant for; the fresh Claude essays of
`skolfie.data.generation.student` answer the same 15 assignments, so a topic
is never a label shortcut.

On disk: `raw_dir/persuade/*.csv`, or a `.zip` holding the csv. The released
train file (`persuade_corpus_2.0_train.csv`) has one row per *discourse
element* with the whole essay repeated in every row, so rows are deduplicated
by essay id; the essay-level file of the repository
(`persuade_2.0_human_scores_demo_id_github.csv`) works too. Columns are found
by name, case-insensitively, from these candidates:

    essay id    essay_id_comp, essay_id, id          (required)
    text        full_text, text, essay_text, essay   (required)
    prompt      prompt_name, prompt, assignment_name (required)
    grade       grade_level, grade                    (optional)
    ELL         ell_status, ell                       (optional)

Records: label/generator "human", domain "academic", subdomain
"student_essay", source "persuade". The prompt, grade and ELL status are kept
in the id, `persuade:<prompt>:g<grade>:ell-<yes|no|na>:<essay id>`, where
<prompt> is the slug used by `skolfie.data.generation.student.PERSUADE_PROMPTS`
when the name is one of the 15 known assignments (else a plain slug of the
name). Fairness checks can parse grade and ELL status from the id.

Cap: `load(..., max_essays=N)` keeps at most N essays (default
`DEFAULT_MAX_ESSAYS`; 0 or None keeps all), as equal a number per prompt as
the prompts allow (a prompt with fewer essays than its share gives the rest
to the others), chosen with a seeded RNG over the sorted essay ids, so the
same files and seed always give the same essays. Equal per-prompt counts
match the AI essays, which are spread evenly over the 15 prompts. The cap
keeps the human student essays from swamping the much smaller set of modern
AI essays. `scripts/build_corpus.py --persuade-max-essays` sets it.

Download (`download=True` and no files): the train csv from Google Drive
(file id `DRIVE_FILE_ID`) with `gdown` (`pip install -e ".[data]"`). Drive
sometimes answers with an HTML page instead of the file (quota or a virus-scan
interstitial); that is detected and raised as a RuntimeError naming the URL.
"""

from __future__ import annotations

import csv
import io
import logging
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from skolfie.data.schema import Record
from skolfie.data.sources._common import first_key, missing_files_error, split_from_name

log = logging.getLogger(__name__)

NAME = "persuade"
DESCRIPTION = (
    "PERSUADE 2.0: ~25k argumentative essays by US students in grades 6-12 on 15 prompts "
    "(human only)."
)
CITATION = (
    "Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Benner, M., Boser, U. (2024). "
    "A large-scale corpus for assessing written argumentation: PERSUADE 2.0. Assessing "
    "Writing 61, 100865."
)
LICENSE_NOTE = "CC BY 4.0 (https://github.com/scrosseye/persuade_corpus_2.0)."

DRIVE_FILE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"
URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
REPO_URL = "https://github.com/scrosseye/persuade_corpus_2.0"

#: Essays kept by default (see the module docstring); 0 or None keeps all.
DEFAULT_MAX_ESSAYS = 4000

MANUAL_DOWNLOAD = f"""\
PERSUADE 2.0 manual download: get the train csv linked from {REPO_URL}
(Google Drive: {URL}; `gdown {DRIVE_FILE_ID}` fetches it from a shell) and put it,
or the .zip holding it, in <raw_dir>/{NAME}/ (e.g. <raw_dir>/{NAME}/persuade_corpus_2.0_train.csv).
"""

ID_KEYS = ("essay_id_comp", "essay_id", "id")
TEXT_KEYS = ("full_text", "text", "essay_text", "essay")
PROMPT_KEYS = ("prompt_name", "prompt", "assignment_name")
GRADE_KEYS = ("grade_level", "grade")
ELL_KEYS = ("ell_status", "ell")

# PERSUADE prompt names whose plain slug differs from student.py's key.
_PROMPT_ALIASES = {
    "a-cowboy-who-rode-the-waves": "cowboy-who-rode-the-waves",
    "the-face-on-mars": "face-on-mars",
    "does-the-electoral-college-work": "electoral-college",
}
_YES = {"yes", "y", "true", "1", "ell"}
_NO = {"no", "n", "false", "0", "non-ell", "not ell"}


def slug(text: str) -> str:
    """Lowercase, runs of anything but [a-z0-9] become "-", trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def prompt_slug(name: str) -> str:
    """The student.py key for a PERSUADE prompt name, or a plain slug of the name."""
    s = slug(name)
    return _PROMPT_ALIASES.get(s, s) or "unknown"


def _grade(value: object) -> str:
    text = str(value if value is not None else "").strip()
    match = re.fullmatch(r"(\d+)(?:\.0+)?", text)
    return f"g{match.group(1)}" if match else "gNA"


def _ell(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in _YES:
        return "ell-yes"
    if text in _NO:
        return "ell-no"
    return "ell-na"


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in (".csv", ".zip")
                  and not p.name.startswith("."))


def _csv_sources(paths: list[Path]) -> Iterator[tuple[str, io.TextIOBase]]:
    """(name for messages, open text stream) for each csv, including csvs inside zips.

    A csv inside a zip is skipped when a csv of the same name sits next to it
    (the unpacked copy of the same file).
    """
    plain = {p.name.lower() for p in paths if p.suffix.lower() == ".csv"}
    for path in paths:
        if path.suffix.lower() == ".csv":
            with open(path, encoding="utf-8-sig", newline="") as fh:
                yield str(path), fh
            continue
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{path}: not a zip file")
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist()
                       if m.lower().endswith(".csv") and not Path(m).name.startswith((".", "_"))]
            if not members:
                raise ValueError(f"{path}: the zip holds no .csv file (has {zf.namelist()[:10]})")
            for member in sorted(members):
                if Path(member).name.lower() in plain:
                    continue
                with zf.open(member) as raw:
                    yield (f"{path}!{member}",
                           io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))


def _read_essays(paths: list[Path]) -> dict[str, dict[str, str]]:
    """essay id -> {text, prompt, grade, ell, split}, first occurrence wins."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    essays: dict[str, dict[str, str]] = {}
    for where, stream in _csv_sources(paths):
        reader = csv.DictReader(stream)
        header = {k: k for k in (reader.fieldnames or [])}
        keys = {}
        for what, candidates, required in (("essay id", ID_KEYS, True),
                                           ("text", TEXT_KEYS, True),
                                           ("prompt", PROMPT_KEYS, True),
                                           ("grade", GRADE_KEYS, False),
                                           ("ell", ELL_KEYS, False)):
            keys[what] = first_key(header, candidates)
            if required and keys[what] is None:
                raise ValueError(f"{where}: no {what} column; looked for {list(candidates)}, "
                                 f"found {sorted(header)}")
        split = split_from_name(Path(where.split("!")[-1]).name)
        rows = 0
        for row in reader:
            rows += 1
            essay_id = str(row.get(keys["essay id"]) or "").strip()
            text = row.get(keys["text"])
            if not essay_id or not isinstance(text, str) or not text.strip():
                continue
            if essay_id in essays:
                continue
            essays[essay_id] = {
                "text": text.strip(),
                "prompt": prompt_slug(str(row.get(keys["prompt"]) or "")),
                "grade": _grade(row.get(keys["grade"]) if keys["grade"] else None),
                "ell": _ell(row.get(keys["ell"]) if keys["ell"] else None),
                "split": split,
            }
        log.info("persuade: %s: %d rows, %d distinct essays so far", where, rows, len(essays))
    return essays


def _download(directory: Path) -> None:
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"gdown is needed to download {URL}; install it (pip install -e \".[data]\") "
            f"or place the file manually.\n{MANUAL_DOWNLOAD}") from exc
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "persuade_download"
    try:
        result = gdown.download(id=DRIVE_FILE_ID, output=str(target), quiet=False)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {URL}: {type(exc).__name__}: {exc}\n"
                           f"{MANUAL_DOWNLOAD}") from exc
    if not result or not target.is_file():
        raise RuntimeError(f"could not download {URL}\n{MANUAL_DOWNLOAD}")
    finalize_download(target)


def finalize_download(target: Path) -> Path:
    """Give a downloaded file its real extension (.zip or .csv); reject an HTML page."""
    if zipfile.is_zipfile(target):
        return target.replace(target.with_suffix(".zip"))
    with open(target, "rb") as fh:
        head = fh.read(512).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        target.unlink()
        raise RuntimeError(
            f"could not download {URL}: Google Drive returned a web page instead of the file "
            f"(quota or access problem)\n{MANUAL_DOWNLOAD}")
    return target.replace(target.with_suffix(".csv"))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def stratified_cap(by_prompt: dict[str, list[str]], max_essays: int | None,
                   seed: int = 0) -> set[str]:
    """Ids to keep: at most `max_essays`, as equal per prompt as the prompts allow.

    Prompts are filled smallest first, each with its fair share of what is
    left, so a small prompt's unused share goes to the larger ones; the
    rounding remainder goes one by one to prompts with room, in name order.
    """
    total = sum(len(v) for v in by_prompt.values())
    if not max_essays or max_essays >= total:
        return {i for ids in by_prompt.values() for i in ids}
    order = sorted(by_prompt, key=lambda name: (len(by_prompt[name]), name))
    quota: dict[str, int] = {}
    left = max_essays
    for k, name in enumerate(order):
        quota[name] = min(len(by_prompt[name]), left // (len(order) - k))
        left -= quota[name]
    for name in sorted(by_prompt):
        if left and quota[name] < len(by_prompt[name]):
            quota[name] += 1
            left -= 1
    rng = random.Random(seed)
    keep: set[str] = set()
    for name in sorted(by_prompt):
        keep.update(rng.sample(sorted(by_prompt[name]), quota[name]))
    return keep


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load(raw_dir: Path, download: bool = True, max_essays: int | None = DEFAULT_MAX_ESSAYS,
         seed: int = 0) -> Iterator[Record]:
    """Yield human student-essay Records from `raw_dir/persuade/` (see the module docstring)."""
    if max_essays is not None and max_essays < 0:
        raise ValueError(f"max_essays must be >= 0 (0 keeps all), got {max_essays}")
    directory = Path(raw_dir) / NAME
    paths = _files(directory)
    if not paths and download:
        _download(directory)
        paths = _files(directory)
    if not paths:
        raise missing_files_error(NAME, directory, "a PERSUADE 2.0 .csv or .zip",
                                  MANUAL_DOWNLOAD)

    essays = _read_essays(paths)
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for essay_id, e in essays.items():
        by_prompt[e["prompt"]].append(essay_id)
    keep = stratified_cap(by_prompt, max_essays, seed)
    if len(keep) < len(essays):
        per_prompt = Counter(essays[i]["prompt"] for i in keep)
        log.info("persuade: kept %d of %d essays (max_essays=%s): %s", len(keep), len(essays),
                 max_essays, dict(sorted(per_prompt.items())))

    for essay_id in sorted(keep):
        e = essays[essay_id]
        yield Record(
            id=f"{NAME}:{e['prompt']}:{e['grade']}:{e['ell']}:{essay_id}",
            text=e["text"], label="human", generator="human", domain="academic",
            subdomain="student_essay", source=NAME, split_hint=e["split"],
        )
