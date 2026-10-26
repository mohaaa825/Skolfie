"""
skolfie/data/sources/hc3.py
---------------------------
Loader for HC3 English (Human ChatGPT Comparison Corpus), Hello-SimpleAI/HC3
on the Hugging Face Hub.

On-disk format: one JSONL file per subset (reddit_eli5, open_qa, wiki_csai,
medicine, finance; `all.jsonl` concatenates them). Each line is one
question with lists of answers:

    {"question": "...", "human_answers": ["..."], "chatgpt_answers": ["..."]}

(some copies also carry "id" and, in all.jsonl, "source"). Every answer
becomes its own Record. The per-subset files are preferred; `all.jsonl` is
read only when none of them is present, so nothing is loaded twice.

Generator: HC3 was collected from the ChatGPT web interface (Dec 2022 to
Jan 2023), before the gpt-3.5-turbo API existed, and the source names no
model version, so AI answers keep generator "chatgpt".

Artefact cleanup (human answers only): the ELI5 / WikiQA human answers in
HC3 come pre-tokenised, with a space before punctuation ("book ." / "it ,")
and split contractions ("do n't", "it 's"). `clean_human_answer` removes
the space before . , ; : ! ? % and closing brackets, after opening brackets,
and re-attaches n't 's 're 've 'll 'd 'm. Spaced quotes (`" Best Seller "`)
and spaced hyphens ("Oscar - winning") are left alone because they cannot be
told apart from legitimate spacing. Without this, the detector would learn
"space before full stop = human".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from skolfie.data.schema import Record
from skolfie.data.sources._common import (
    hf_download_file,
    hf_url,
    iter_jsonl,
    missing_files_error,
    pick,
)

log = logging.getLogger(__name__)

NAME = "hc3"
DESCRIPTION = (
    "HC3 English: ~24k questions from ELI5, WikiQA, Wikipedia CS/AI, medical dialogue and "
    "FiQA, each with human answers and ChatGPT answers (Dec 2022 to Jan 2023)."
)
CITATION = (
    "Guo, B., Zhang, X., Wang, Z., Jiang, M., Nie, J., Ding, Y., Yue, J., Wu, Y. (2023). "
    "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection. "
    "arXiv:2301.07597."
)
LICENSE_NOTE = (
    "CC-BY-SA, except where a source dataset's licence is stricter (ELI5: BSD; WikiQA: "
    "Microsoft/PWC custom; medicine and finance: licence unknown per the HC3 README)."
)

REPO_ID = "Hello-SimpleAI/HC3"
SUBSETS = ("reddit_eli5", "open_qa", "wiki_csai", "medicine", "finance")
ALL_FILE = "all.jsonl"

MANUAL_DOWNLOAD = f"""\
HC3 (English) manual download: from https://huggingface.co/datasets/{REPO_ID}/tree/main
download these files into <raw_dir>/{NAME}/ (file names unchanged):
""" + "".join(f"  {hf_url(REPO_ID, s + '.jsonl')}\n" for s in SUBSETS) + f"""\
(or just {hf_url(REPO_ID, ALL_FILE)}; it is used only when the per-subset files are absent).
The same files are mirrored on ModelScope: https://www.modelscope.cn/datasets/simpleai/HC3
"""

# HC3 subset -> coarse schema domain. Subdomain keeps the subset name.
DOMAIN_MAP = {
    "reddit_eli5": "qa",
    "open_qa": "qa",
    "wiki_csai": "academic",
    "medicine": "qa",
    "finance": "qa",
}

HUMAN_KEYS = ("human_answers",)
AI_KEYS = ("chatgpt_answers",)
ID_KEYS = ("id", "index")

_SPACE_BEFORE = re.compile(r"[ \t]+([.,;:!?%)\]}])")
_SPACE_AFTER = re.compile(r"([(\[{])[ \t]+")
_SPLIT_CONTRACTION = re.compile(r"(\w)[ \t]+(n't|'s|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)


def clean_human_answer(text: str) -> str:
    """Undo HC3's pre-tokenisation artefacts in human answers (see module docstring)."""
    text = _SPLIT_CONTRACTION.sub(r"\1\2", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    return text.strip()


def _local_files(directory: Path) -> list[tuple[str, Path]]:
    subset_files = [(s, directory / f"{s}.jsonl") for s in SUBSETS]
    present = [(s, p) for s, p in subset_files if p.is_file()]
    if present:
        return present
    if (directory / ALL_FILE).is_file():
        return [("all", directory / ALL_FILE)]
    return []


def _download(directory: Path) -> None:
    errors = []
    for subset in SUBSETS:
        try:
            hf_download_file(REPO_ID, f"{subset}.jsonl", directory)
        except RuntimeError as exc:
            errors.append(str(exc))
    if len(errors) == len(SUBSETS):
        raise RuntimeError(errors[0] + "\n" + MANUAL_DOWNLOAD)
    for err in errors:
        log.warning("hc3: %s", err)


def _answers(row: dict, keys: tuple[str, ...], where: str) -> list[str]:
    value = pick(row, keys, where, "answer list", default=None)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [a for a in value if isinstance(a, str) and a.strip()]


def load(raw_dir: Path, download: bool = True) -> Iterator[Record]:
    """Yield one Record per HC3 English answer found in `raw_dir/hc3/`."""
    directory = Path(raw_dir) / NAME
    files = _local_files(directory)
    if not files and download:
        _download(directory)
        files = _local_files(directory)
    if not files:
        raise missing_files_error(
            NAME, directory, ", ".join(f"{s}.jsonl" for s in SUBSETS) + f" or {ALL_FILE}",
            MANUAL_DOWNLOAD,
        )

    for file_subset, path in files:
        for index, row in iter_jsonl(path):
            where = f"{path}:{index + 1}"
            human = _answers(row, HUMAN_KEYS, where)
            ai = _answers(row, AI_KEYS, where)
            if not any(k in row for k in HUMAN_KEYS + AI_KEYS):
                raise ValueError(
                    f"{where}: no answer fields; looked for {list(HUMAN_KEYS + AI_KEYS)}, "
                    f"found keys {sorted(row)}"
                )
            subset = file_subset
            if file_subset == "all":
                subset = str(pick(row, ("source",), where, "source", default="all"))
            local_id = pick(row, ID_KEYS, where, "id", default=index)
            domain = DOMAIN_MAP.get(subset, "qa")
            prefix = f"{NAME}:{file_subset}:{local_id}"

            for i, answer in enumerate(human):
                text = clean_human_answer(answer)
                if not text:
                    continue
                yield Record(
                    id=f"{prefix}:human:{i}", text=text, label="human", generator="human",
                    domain=domain, subdomain=subset, source=NAME,
                )
            for i, answer in enumerate(ai):
                yield Record(
                    id=f"{prefix}:chatgpt:{i}", text=answer.strip(), label="ai",
                    generator="chatgpt", domain=domain, subdomain=subset, source=NAME,
                )
