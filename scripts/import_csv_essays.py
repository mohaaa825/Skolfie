"""
scripts/import_csv_essays.py
-----------------------------
Turn a CSV of AI-written student essays (one row per essay) into fresh-source
JSONL records, one `<out-dir>/<generator>.jsonl` per model.

Made for the Gemini essays the user generated on 2026-10-30
(`student_essays_dataset.csv`), whose columns are:

    Essay Number, Model Used, Grade Level, Essay Type, Target Length,
    Prompt, Generated Essay

- generator: "Model Used" without the "models/" prefix, lower-cased
  (e.g. "gemini-3.1-flash-lite").
- topic: what the prompt is "about" ("Write a book review about the water
  cycle. ..." -> "the-water-cycle"). The prompt id is
  `<prefix>-<topic>.<grade>.<type>.w<N>.n<essay number>`, so the corpus
  groups essays by topic and grade (`fresh:<prefix>-<topic>.<grade>`), as it
  does the Claude PERSUADE essays. (Grouping by topic alone leaves too few
  groups: with 27 topics the 10% validation split got none.)
- text: Markdown emphasis and heading marks are removed first (`**`, `*`,
  `_`, leading `#`; `*` bullets become `-`; inline LaTeX becomes plain
  text): the human student essays have none, so keeping them would teach
  the model "asterisks = AI". Then `generate_ai_texts.py`'s
  `clean_output` strips wrappers and drops refusals, empty and too-short
  texts, and `make_record` builds the record (label "ai", source "fresh").
- personal narratives get domain "creative"; everything else "academic",
  subdomain "student_essay".

Usage:

    python scripts/import_csv_essays.py student_essays_dataset.csv \\
        --prefix gemini --out-dir data/raw/fresh

Overwrites each output file. Prints one row of counts per generator.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/...` without installing
    sys.path.insert(0, str(REPO_ROOT))

from skolfie.data.generation.prompts import PromptSpec, max_tokens_for  # noqa: E402

COLUMNS = {"number": "Essay Number", "model": "Model Used", "grade": "Grade Level",
           "type": "Essay Type", "length": "Target Length", "prompt": "Prompt",
           "text": "Generated Essay"}
_TOPIC_RE = re.compile(r"^Write an? .+? (?:about|on) (.+?)\.\s+Write like", re.IGNORECASE)


def _generate_module():
    name = "skolfie_generate_ai_texts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses look their module up while it executes
    module_ok = False
    try:
        spec.loader.exec_module(module)
        module_ok = True
    finally:
        if not module_ok:
            del sys.modules[name]
    return module


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def _plain_math(m: re.Match) -> str:
    math = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", m.group(1))
    math = math.replace("\\times", "x").replace("\\cdot", "x").replace("\\rightarrow", "->")
    return re.sub(r"[_^{}\\]", "", math)


def strip_markdown(text: str) -> str:
    """Remove Markdown emphasis and heading marks, keep the words; inline LaTeX
    (`$M_1V_1$`, `$\\text{CO}_2$`) becomes plain text (`M1V1`, `CO2`)."""
    # Math only, not prices ("$10 ... $20"): a backslash command, a single letter, or no
    # leading digit/space plus a _ ^ or = inside.
    text = re.sub(r"\$((?=[^$\n]*\\)[^$\n]{1,80}?|[A-Za-z]"
                  r"|(?=[^$\d\s][^$\n]*[_^=])[^$\n]{1,80}?)\$", _plain_math, text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^(\s*)[*+]\s+", r"\1- ", text)  # "* item" bullets -> "- item"
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"[ \t]+\n", "\n", text)  # Markdown hard breaks ("  \n")
    return text.strip()


def topic_of(prompt: str) -> Optional[str]:
    m = _TOPIC_RE.search(prompt.strip())
    return m.group(1) if m else None


def target_words(length: str) -> int:
    m = re.search(r"\d+", length or "")
    return int(m.group(0)) if m else 400


def row_spec(row: dict, prefix: str) -> PromptSpec:
    prompt = row[COLUMNS["prompt"]]
    topic = topic_of(prompt)
    if topic is None:
        raise ValueError(f"essay {row[COLUMNS['number']]}: no topic in prompt {prompt[:80]!r}")
    essay_type = row[COLUMNS["type"]].strip()
    words = target_words(row[COLUMNS["length"]])
    narrative = "narrative" in essay_type.lower()
    prompt_id = (f"{prefix}-{slug(topic)}.{slug(row[COLUMNS['grade']])}."
                 f"{slug(essay_type)}.w{words}.n{int(row[COLUMNS['number']]):04d}")
    return PromptSpec(prompt_id=prompt_id, template_id=f"{prefix}-{slug(topic)}",
                      domain="creative" if narrative else "academic",
                      subdomain="personal_narrative" if narrative else "student_essay",
                      style=f"{slug(row[COLUMNS['grade']])}+{slug(essay_type)}", text=prompt,
                      target_words=words, temperature=1.0, max_tokens=max_tokens_for(words))


def generator_of(model: str) -> str:
    return model.strip().lower().removeprefix("models/")


def import_csv(path: Path, out_dir: Path, prefix: str) -> dict[str, Counter]:
    gen = _generate_module()
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in COLUMNS.values() if rows and c not in rows[0]]
    if not rows or missing:
        raise ValueError(f"{path}: no rows or missing column(s) {missing}")
    records: dict[str, list[dict]] = {}
    counts: dict[str, Counter] = {}
    for row in rows:
        generator = generator_of(row[COLUMNS["model"]])
        c = counts.setdefault(generator, Counter())
        c["rows"] += 1
        text, why = gen.clean_output(strip_markdown(row[COLUMNS["text"]] or ""))
        if text is None:
            c[why] += 1
            continue
        records.setdefault(generator, []).append(
            gen.make_record(generator, row_spec(row, prefix), text))
        c["written"] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for generator, recs in records.items():
        with open(gen.output_path(out_dir, generator), "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--prefix", default="gemini",
                        help="Prompt-id prefix naming the batch (default gemini)")
    parser.add_argument("--out-dir", type=Path, default=Path("data") / "raw" / "fresh")
    args = parser.parse_args(argv)
    try:
        counts = import_csv(args.csv, args.out_dir, slug(args.prefix))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cols = ("rows", "written", "empty", "refusal", "too_short")
    print(f"{'generator':32}" + "".join(f"{c:>10}" for c in cols))
    for generator, c in sorted(counts.items()):
        print(f"{generator:32}" + "".join(f"{c[k]:>10}" for k in cols))
    print(f"output: {args.out_dir}/<generator>.jsonl")
    return 0 if any(c["written"] for c in counts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
