"""
skolfie/data/sources/fresh.py
-----------------------------
Loader for "fresh" text: recent human and AI texts produced locally by the
generation script and written to `raw_dir/fresh/*.jsonl`. Each line is
already a Record dict with the schema COLUMNS as keys; this loader only
validates the lines through `Record` (label, domain, generator rules) and
never downloads anything.

The generation script (scripts/generate_ai_texts.py) names files after the
model id, so the generator is read from each record, never from the file
name. Its ids are "fresh:<generator>:<prompt-id>:<sample-idx>", and since the
generator may itself contain ":" (Ollama ids), use `id.rsplit(":", 2)` to
parse them. Files are append-only and a crashed run can leave one truncated
line, so lines that are not valid JSON objects are skipped with a warning.

Ids without the "fresh:" prefix get it added; an empty `source` becomes
"fresh". Unknown extra keys are ignored. A well-formed line that is missing a
required column or fails validation raises ValueError naming the file and
line. Text is passed through as written (paragraph breaks kept); the corpus
builder does whitespace normalisation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from skolfie.data.schema import COLUMNS, Record
from skolfie.data.sources._common import missing_files_error

log = logging.getLogger(__name__)

NAME = "fresh"
DESCRIPTION = (
    "Locally generated recent human/AI texts written as Record JSONL by the generation script."
)
CITATION = ""
LICENSE_NOTE = (
    "Depends on the generation run: AI outputs per the model provider's terms; "
    "human texts per their sources."
)
MANUAL_DOWNLOAD = f"""\
Nothing to download: run the fresh-text generation script, which writes
<raw_dir>/{NAME}/*.jsonl with one Record dict (keys: {", ".join(COLUMNS)}) per line.
"""

REQUIRED = ("id", "text", "label", "generator", "domain", "subdomain")


def _iter_lines(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield (line number, row), skipping blank, truncated or non-object lines."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("fresh: %s:%d: skipping line that is not valid JSON", path, lineno)
                continue
            if not isinstance(row, dict):
                log.warning("fresh: %s:%d: skipping line that is not a JSON object", path, lineno)
                continue
            yield lineno, row


def load(raw_dir: Path, download: bool = True) -> Iterator[Record]:
    """Yield validated Records from `raw_dir/fresh/*.jsonl`. `download` is ignored."""
    directory = Path(raw_dir) / NAME
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    if not files:
        raise missing_files_error(NAME, directory, "*.jsonl", MANUAL_DOWNLOAD)

    for path in files:
        for lineno, row in _iter_lines(path):
            where = f"{path}:{lineno}"
            missing = [k for k in REQUIRED if k not in row]
            if missing:
                raise ValueError(f"{where}: missing columns {missing}; found keys {sorted(row)}")
            fields = {k: row[k] for k in COLUMNS if k in row}
            fields["id"] = str(fields["id"])
            if not fields["id"].startswith(f"{NAME}:"):
                fields["id"] = f"{NAME}:{fields['id']}"
            fields["source"] = fields.get("source") or NAME
            fields["split_hint"] = fields.get("split_hint") or ""
            fields["n_words"] = int(fields.get("n_words") or 0)
            try:
                yield Record(**fields)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}: invalid record: {exc}") from exc
