"""
scripts/import_claude_essays.py
--------------------------------
Turn essays written outside `generate_ai_texts.py` (e.g. by Claude
subagents, one text file per prompt) into fresh-source JSONL records.

Inputs:

    --plan PLAN.json     a JSON list of dicts: the fields of a `PromptSpec`
                         (`dataclasses.asdict`, e.g. from
                         `skolfie.data.generation.student.student_prompt_plan`)
                         plus "model" (the essays sub-folder) and "generator"
                         (the generator id written to the records)
    --essays-dir DIR     DIR/<model>/<prompt_id>.txt, one essay per file

Output: `<out-dir>/<generator>.jsonl` (default out-dir data/raw/fresh), one
record per essay, made exactly like `generate_ai_texts.py` makes them:
`clean_output` strips a leading "Sure, here's..." line, a title line, a
closing offer to revise and markdown wrappers, and drops refusals and texts
under its minimum length; `make_record` gives the id
`fresh:<generator>:<prompt_id>:<sample_idx>`, label "ai", the spec's domain
and subdomain, and source "fresh". `--min-words` can raise the minimum.

Idempotent: each output file is rewritten (atomically) from the plan and the
essay files, so running it twice gives the same bytes. Lines of an existing
output file whose id this plan does not produce (e.g. from
`generate_ai_texts.py`) are kept as they are, and so is the existing record
of a planned essay whose file is now missing or unusable; a planned essay
with a usable file replaces its old record.

Prints one row of counts per generator (planned, written, missing, empty,
too short, refusal) and exits 0, or 1 when no essay at all was usable, 2 on
a bad plan.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/...` without installing
    sys.path.insert(0, str(REPO_ROOT))

from skolfie.data.generation.prompts import PromptSpec  # noqa: E402

DEFAULT_OUT_DIR = Path("data") / "raw" / "fresh"
_SPEC_FIELDS = {f.name for f in dataclasses.fields(PromptSpec)}
_REQUIRED = {"prompt_id", "template_id", "domain", "subdomain", "style", "text",
             "target_words", "temperature", "max_tokens"}
COUNT_COLUMNS = ("planned", "written", "missing", "empty", "too_short", "refusal")


def _generate_module():
    """scripts/generate_ai_texts.py (clean_output, make_record, record_id, output_path)."""
    import importlib.util

    name = "skolfie_generate_ai_texts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_plan(path: Path) -> list[tuple[str, str, PromptSpec]]:
    """[(model, generator, spec)] from the plan file; ValueError on a bad entry."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of prompt dicts")
    out = []
    seen: set[tuple[str, str, int]] = set()
    for i, entry in enumerate(data):
        where = f"{path}: entry {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected an object")
        missing = sorted((_REQUIRED | {"model", "generator"}) - set(entry))
        if missing:
            raise ValueError(f"{where}: missing key(s) {missing}")
        model = str(entry["model"]).strip()
        generator = str(entry["generator"]).strip().lower()
        if not model or not generator or "/" in model or "\\" in model:
            raise ValueError(f"{where}: bad model {entry['model']!r} or generator "
                             f"{entry['generator']!r}")
        spec = PromptSpec(**{k: v for k, v in entry.items() if k in _SPEC_FIELDS})
        if "/" in spec.prompt_id or "\\" in spec.prompt_id or spec.prompt_id in ("", ".", ".."):
            raise ValueError(f"{where}: bad prompt_id {spec.prompt_id!r}")
        key = (generator, spec.prompt_id, spec.sample_idx)
        if key in seen:
            raise ValueError(f"{where}: {generator}/{spec.prompt_id} is planned twice")
        seen.add(key)
        out.append((model, generator, spec))
    return out


def _read_existing(path: Path) -> list[tuple[Optional[str], str]]:
    """[(id or None, raw line)] of an existing output file (blank lines dropped)."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rid = json.loads(line).get("id")
            except (json.JSONDecodeError, AttributeError):
                rid = None
            rows.append((rid if isinstance(rid, str) else None, line))
    return rows


def import_essays(plan: list[tuple[str, str, PromptSpec]], essays_dir: Path, out_dir: Path,
                  min_words: int = 0) -> dict[str, Counter]:
    """Write `<out_dir>/<generator>.jsonl` per generator; counts per generator."""
    gen = _generate_module()
    by_generator: dict[str, list[tuple[str, PromptSpec]]] = {}
    for model, generator, spec in plan:
        by_generator.setdefault(generator, []).append((model, spec))

    counts: dict[str, Counter] = {}
    for generator, items in by_generator.items():
        c: Counter = Counter(planned=len(items))
        new_records: dict[str, str] = {}
        for model, spec in items:
            path = essays_dir / model / f"{spec.prompt_id}.txt"
            if not path.is_file():
                c["missing"] += 1
                continue
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            text, reason = gen.clean_output(raw)
            if text is not None and len(text.split()) < min_words:
                text, reason = None, "too_short"
            if text is None:
                c[reason] += 1
                continue
            record = gen.make_record(generator, spec, text)
            new_records[record["id"]] = json.dumps(record, ensure_ascii=False)
            c["written"] += 1
        counts[generator] = c

        out_path = gen.output_path(out_dir, generator)
        planned_ids = {gen.record_id(generator, spec) for _, spec in items}
        lines = []
        for rid, line in _read_existing(out_path):
            if rid is not None and rid in new_records:
                continue  # replaced below, in plan order
            if rid is not None and rid in planned_ids:
                c["kept_old"] += 1
            lines.append(line)
        lines += [new_records[gen.record_id(generator, spec)] for _, spec in items
                  if gen.record_id(generator, spec) in new_records]
        if not lines:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        os.replace(tmp, out_path)
    return counts


def format_counts(counts: dict[str, Counter]) -> str:
    cols = (*COUNT_COLUMNS, "kept_old")
    rows = [("generator", *cols)]
    for generator in sorted(counts):
        rows.append((generator, *(str(counts[generator][k]) for k in cols)))
    rows.append(("total", *(str(sum(c[k] for c in counts.values())) for k in cols)))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(v.ljust(w) if i == 0 else v.rjust(w)
                               for i, (v, w) in enumerate(zip(r, widths))) for r in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn essay text files (<essays-dir>/<model>/<prompt_id>.txt) into "
                    "fresh-source JSONL records.")
    parser.add_argument("--plan", type=Path, required=True,
                        help="JSON list of PromptSpec dicts plus 'model' and 'generator'")
    parser.add_argument("--essays-dir", type=Path, required=True,
                        help="folder with one sub-folder of <prompt_id>.txt files per model")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"where <generator>.jsonl goes (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--min-words", type=int, default=0,
                        help="also skip essays shorter than this (default: only the "
                             "generation post-filter's 30-word minimum)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: cannot read plan {args.plan}: {exc}", file=sys.stderr)
        return 2
    if not args.essays_dir.is_dir():
        print(f"error: essays folder not found: {args.essays_dir}", file=sys.stderr)
        return 2
    counts = import_essays(plan, args.essays_dir, args.out_dir, min_words=args.min_words)
    print(format_counts(counts))
    print(f"output: {args.out_dir}/<generator>.jsonl")
    return 0 if any(c["written"] for c in counts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
