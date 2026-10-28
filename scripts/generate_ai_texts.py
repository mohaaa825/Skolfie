"""
scripts/generate_ai_texts.py
-----------------------------
Generate fresh AI-written text from current models, across domains, as
corpus records (label "ai", source "fresh").

Examples:

    # See what would be sent (no API calls, no files written)
    python scripts/generate_ai_texts.py --provider anthropic --n 500 --dry-run

    # Generate 500 texts with Claude (needs ANTHROPIC_API_KEY)
    python scripts/generate_ai_texts.py --provider anthropic --n 500 --seed 0

    # OpenAI / Gemini (OPENAI_API_KEY / GEMINI_API_KEY)
    python scripts/generate_ai_texts.py --provider openai --model gpt-6-astra --n 500
    python scripts/generate_ai_texts.py --provider gemini --n 500

    # Llama / Mistral on any OpenAI-compatible server
    OPENAI_COMPATIBLE_BASE_URL=http://localhost:11434/v1 \\
        python scripts/generate_ai_texts.py --provider openai-compatible --model llama3.3:70b
    OPENAI_COMPATIBLE_API_KEY=... python scripts/generate_ai_texts.py \\
        --provider together --model meta-llama/Llama-3.3-70B-Instruct-Turbo

Output goes to `<out-dir>/<generator>.jsonl` (append-only, one record per
line, exactly `skolfie.data.schema.COLUMNS`), where `<generator>` is the
model id lowercased, with characters outside [a-z0-9._-] replaced by "_"
in the file name only. Record ids are
`fresh:<generator>:<prompt-id>:<sample-idx>`.

The run is resumable: the prompt plan is a pure function of `--n`, `--seed`
and `--domains`, and prompts whose record id is already in the output file
are skipped, so re-running the same command after an interruption only
generates what is missing. Outputs that the post-filter drops (refusals,
fewer than 30 words) are not written and are retried on the next run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/...` without installing
    sys.path.insert(0, str(REPO_ROOT))

from skolfie.data.generation.prompts import (  # noqa: E402
    DOMAIN_BLOCK,
    PromptSpec,
    count_combinations,
    sample_prompts,
)
from skolfie.data.generation.providers import (  # noqa: E402
    DEFAULT_MODELS,
    FATAL_STATUSES,
    PROVIDER_NAMES,
    Provider,
    ProviderConfigError,
    ProviderError,
    approx_tokens,
    generator_id,
    make_provider,
    resolve_model,
)
from skolfie.data.schema import COLUMNS, Record  # noqa: E402

DEFAULT_OUT_DIR = Path("data") / "raw" / "fresh"
MIN_WORDS = 30
SOURCE = "fresh"

# --------------------------------------------------------------------------- #
# Post-filter
# --------------------------------------------------------------------------- #

_META_NOUNS = (
    r"essay|story|poem|post|review|email|e-mail|article|guide|piece|text|version|response|"
    r"abstract|column|narrative|explanation|explainer|documentation|section|readme|reply|"
    r"tutorial|scene|paper|walkthrough|draft|words?"
)
# "Sure! Here's a 300-word essay on X." / "Certainly." / "Of course! Below is ..."
_INTERJECTION_RE = re.compile(
    r"^(?:sure|certainly|of course|absolutely)\s*[!.,]\s*"
    r"(?:$|(?:here(?:'|’)?s|here is|here are|below is|i(?:'|’)ve written|i have written)\b)",
    re.IGNORECASE,
)
# "Here is a short story about a lighthouse keeper:" (must end with a colon)
_HERE_IS_RE = re.compile(
    rf"^(?:here(?:'|’)?s|here is|here are|below is|this is)\b.*\b(?:{_META_NOUNS})\b.*:\s*$",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"^(?:#{1,6}\s+\S.*|\*\*[^*\n]+\*\*:?|__[^_\n]+__|title:\s*\S.*)$",
                       re.IGNORECASE)
_BOLD_LINE_RE = re.compile(r"^\*\*([^*\n]+)\*\*(:?)[ \t]*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(?:#{1,6}\s+\S|\*\*[^*\n]+\*\*:?[ \t]*$)", re.MULTILINE)
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OUTRO_RE = re.compile(
    rf"\b(?:would you like me to|if you(?:'|’)?d like,? i can|let me know if you(?:'|’)?d like "
    rf"me to|i can (?:also )?(?:adjust|revise|expand|shorten|rewrite|tweak))\b"
    rf"|\b(?:this|the above)\s+(?:{_META_NOUNS})\b.*\b(?:adjust|revise|tweak|expand|shorten|"
    rf"rewrite|modify|tone|length)\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"^(?:(?:i(?:'|’)?m|i am) sorry,? but\s+)?"
    r"i(?:\s+(?:can(?:'|’)?t|cannot|won(?:'|’)?t|will not|am (?:not able|unable) to)"
    r"|(?:'|’)m (?:not able|unable) to)\s+"
    r"(?:help(?! but| feeling| thinking| wondering| it\b)|assist|write|create|produce|"
    r"provide|comply|fulfil+|generate|do that)"
    r"|\bas an ai(?: language model)?\b",
    re.IGNORECASE,
)


def _strip_wrappers(text: str) -> str:
    text = _THINK_RE.sub("", text).replace("\r\n", "\n").strip()
    fenced = _FENCE_RE.match(text)
    if fenced and "```" not in fenced.group(1):
        text = fenced.group(1).strip()

    lines = text.split("\n")
    first = lines[0].strip() if lines else ""
    if len(first.split()) <= 40 and (_INTERJECTION_RE.match(first) or _HERE_IS_RE.match(first)):
        lines = lines[1:]
    while lines and (not lines[0].strip() or _RULE_RE.match(lines[0])):
        lines = lines[1:]

    # A short title line at the very top (markdown heading, bold line, "Title: ..."),
    # unless the body has headings of its own (then it is the first section heading).
    if (len(lines) > 1 and _TITLE_RE.match(lines[0].strip()) and len(lines[0].split()) <= 15
            and not _HEADING_RE.search("\n".join(lines[1:]))):
        lines = lines[1:]
    text = "\n".join(lines).strip()

    # A closing offer to revise, in the assistant's own voice.
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        last = paragraphs[-1].strip()
        if len(last.split()) <= 40 and _OUTRO_RE.search(last):
            paragraphs = paragraphs[:-1]
            while paragraphs and _RULE_RE.match(paragraphs[-1]):
                paragraphs = paragraphs[:-1]
            text = "\n\n".join(paragraphs).strip()

    # Whole-line bold used as a heading: keep the words, drop the markers.
    text = _BOLD_LINE_RE.sub(r"\1\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_output(raw: str | None) -> tuple[Optional[str], str]:
    """Post-filter one model output.

    Returns `(text, "ok")`, or `(None, reason)` with reason "empty",
    "refusal" or "too_short". Only clear wrappers are removed: a leading
    "Sure! Here's..." line, a title line at the top, a closing offer to
    revise, a code fence around the whole output, <think> blocks, and bold
    markers around whole lines. Headings and formatting inside the body
    are kept.
    """
    if not raw or not raw.strip():
        return None, "empty"
    text = _strip_wrappers(raw)
    if not text:
        return None, "empty"
    if _REFUSAL_RE.search(text[:300]):
        return None, "refusal"
    if len(text.split()) < MIN_WORDS:
        return None, "too_short"
    return text, "ok"


# --------------------------------------------------------------------------- #
# Records and output files
# --------------------------------------------------------------------------- #


def output_path(out_dir: Path | str, generator: str) -> Path:
    safe = re.sub(r"[^a-z0-9._-]+", "_", generator.lower()).strip("._") or "unknown"
    return Path(out_dir) / f"{safe}.jsonl"


def record_id(generator: str, spec: PromptSpec) -> str:
    return f"{SOURCE}:{generator}:{spec.prompt_id}:{spec.sample_idx}"


def make_record(generator: str, spec: PromptSpec, text: str) -> dict:
    """A validated Record dict with exactly COLUMNS."""
    rec = Record(
        id=record_id(generator, spec),
        text=text,
        label="ai",
        generator=generator,
        domain=spec.domain,
        subdomain=spec.subdomain,
        source=SOURCE,
        split_hint="",
    ).to_dict()
    return {col: rec[col] for col in COLUMNS}


def load_done_ids(path: Path) -> set[str]:
    """Record ids already in an output file; unreadable lines are ignored."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("id")
            except (json.JSONDecodeError, AttributeError):
                continue
            if isinstance(rid, str):
                done.add(rid)
    return done


def _ensure_trailing_newline(path: Path) -> None:
    """After a crash mid-write, start the next record on a fresh line."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


class _Cancelled(Exception):
    """A queued request skipped after a fatal error."""


@dataclass
class RunSummary:
    planned: int = 0
    skipped_existing: int = 0
    attempted: int = 0
    written: int = 0
    failed: int = 0
    dropped: Counter = field(default_factory=Counter)
    aborted: str = ""


def run_generation(
    provider: Provider,
    specs: Sequence[PromptSpec],
    out_path: Path,
    *,
    generator: str | None = None,
    concurrency: int = 4,
    log: Callable[[str], None] = print,
    progress_every: int = 25,
) -> RunSummary:
    """Generate, post-filter and append records for every spec not already done."""
    generator = generator or generator_id(provider.model_name)
    summary = RunSummary(planned=len(specs))
    done = load_done_ids(out_path)
    todo = [s for s in specs if record_id(generator, s) not in done]
    summary.skipped_existing = len(specs) - len(todo)
    if not todo:
        return summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(out_path)
    stop = threading.Event()

    def call(spec: PromptSpec) -> str:
        if stop.is_set():
            raise _Cancelled
        try:
            return provider.generate(spec.text, temperature=spec.temperature,
                                     max_tokens=spec.max_tokens)
        except ProviderError as exc:
            if exc.status in FATAL_STATUSES:
                stop.set()  # bad key or model: don't send the queued requests
            raise

    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = {pool.submit(call, s): s for s in todo}
    try:
        with out_path.open("a", encoding="utf-8") as fh:
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    raw = fut.result()
                except _Cancelled:
                    continue
                except ProviderError as exc:
                    summary.attempted += 1
                    summary.failed += 1
                    if exc.status in FATAL_STATUSES:
                        summary.aborted = str(exc)
                        log(f"[abort] {exc}")
                        break
                    log(f"[fail] {spec.prompt_id}: {exc}")
                    continue
                except Exception as exc:  # a bug or odd payload: skip this one, keep going
                    summary.attempted += 1
                    summary.failed += 1
                    log(f"[fail] {spec.prompt_id}: {type(exc).__name__}: {exc}")
                    continue
                summary.attempted += 1
                text, reason = clean_output(raw)
                if text is None:
                    summary.dropped[reason] += 1
                else:
                    record = make_record(generator, spec, text)
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    summary.written += 1
                if progress_every and summary.attempted % progress_every == 0:
                    log(f"  {summary.attempted}/{len(todo)} done, {summary.written} written")
    except KeyboardInterrupt:
        summary.aborted = "interrupted; re-run the same command to resume"
        log(f"[abort] {summary.aborted}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_domains(value: str | None) -> Optional[list[str]]:
    if not value:
        return None
    return [d.strip() for d in value.split(",") if d.strip()]


def build_parser() -> argparse.ArgumentParser:
    defaults = ", ".join(f"{k}={v}" for k, v in DEFAULT_MODELS.items())
    parser = argparse.ArgumentParser(
        description="Generate fresh AI text from current models as 'fresh' corpus records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, "
            "OPENAI_COMPATIBLE_BASE_URL, OPENAI_COMPATIBLE_API_KEY "
            "(optional: ANTHROPIC_BASE_URL, OPENAI_BASE_URL, GEMINI_BASE_URL).\n"
            f"Default models: {defaults}."
        ),
    )
    parser.add_argument("--provider", required=True, choices=PROVIDER_NAMES)
    parser.add_argument("--model", default=None,
                        help="model id (default per provider; required for OpenAI-compatible)")
    parser.add_argument("--n", type=int, default=500, help="number of prompts (default 500)")
    parser.add_argument("--seed", type=int, default=0, help="prompt-sampling seed (default 0)")
    parser.add_argument("--domains", default=None,
                        help="comma-separated subset of: "
                             + ",".join(dict.fromkeys(DOMAIN_BLOCK)) + " (default: all)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default {DEFAULT_OUT_DIR})")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel requests (default 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print sample prompts and planned counts; call nothing")
    parser.add_argument("--show", type=int, default=5,
                        help="sample prompts to print with --dry-run (default 5)")
    parser.add_argument("--base-url", default=None,
                        help="override the API base URL (e.g. for OpenAI-compatible servers)")
    parser.add_argument("--effort", default=None,
                        choices=("low", "medium", "high", "xhigh", "max"),
                        help="anthropic only: output_config.effort (default: the model's default)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-request timeout in seconds (default 120)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="retries on 408/429/5xx/connection errors (default 5)")
    return parser


def _estimate(specs: Iterable[PromptSpec]) -> tuple[int, int, int]:
    specs = list(specs)
    tokens_in = sum(approx_tokens(s.text) for s in specs)
    tokens_out = sum(int(s.target_words * 1.35) for s in specs)
    cap = sum(s.max_tokens for s in specs)
    return tokens_in, tokens_out, cap


def print_plan(provider: str, model: str, generator: str, out_path: Path,
               specs: Sequence[PromptSpec], show: int) -> None:
    done = load_done_ids(out_path)
    todo = [s for s in specs if record_id(generator, s) not in done]
    tokens_in, tokens_out, cap = _estimate(todo)
    print(f"provider:   {provider}")
    print(f"model:      {model}  (generator {generator!r})")
    print(f"output:     {out_path}")
    print(f"prompt bank: {count_combinations()} distinct combinations")
    print(f"planned:    {len(specs)} prompts, {len(specs) - len(todo)} already in the output, "
          f"{len(todo)} to request")
    by_domain = Counter(s.domain for s in todo)
    print("by domain:  " + ", ".join(f"{d}={c}" for d, c in sorted(by_domain.items())))
    by_style = Counter(s.style for s in todo)
    print("by style:   " + ", ".join(f"{k}={c}" for k, c in sorted(by_style.items())))
    print(f"requests:   {len(todo)} (plus any retries)")
    print(f"tokens:     ~{tokens_in} input, ~{tokens_out} output expected "
          f"(max_tokens budget {cap}; thinking models may add more output)")
    for spec in todo[: max(0, show)]:
        print()
        print(f"[{record_id(generator, spec)}] {spec.domain}/{spec.subdomain} "
              f"temperature={spec.temperature} max_tokens={spec.max_tokens}")
        print(f"  {spec.text}")


def print_summary(summary: RunSummary, provider: Provider, out_path: Path) -> None:
    stats = getattr(provider, "stats", None)
    print()
    print(f"output:     {out_path}")
    print(f"planned {summary.planned}, already done {summary.skipped_existing}, "
          f"attempted {summary.attempted}, written {summary.written}, failed {summary.failed}")
    if summary.dropped:
        print("dropped:    " + ", ".join(f"{k}={v}" for k, v in sorted(summary.dropped.items())))
    if stats is not None:
        print(f"requests:   {stats.requests} sent ({stats.retries} retries, "
              f"{stats.failures} failed)")
        print(f"tokens:     {stats.input_tokens} input, {stats.output_tokens} output "
              "(as reported by the API; estimated for the fake provider)")
    if summary.aborted:
        print(f"aborted:    {summary.aborted}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if not args.timeout > 0:
        parser.error("--timeout must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")

    try:
        model = resolve_model(args.provider, args.model)
        specs = sample_prompts(args.n, seed=args.seed, domains=_parse_domains(args.domains))
    except (ProviderConfigError, ValueError) as exc:
        parser.error(str(exc))
    generator = generator_id(model)
    out_path = output_path(args.out_dir, generator)

    if args.dry_run:
        print_plan(args.provider, model, generator, out_path, specs, args.show)
        return 0

    try:
        provider = make_provider(args.provider, model, base_url=args.base_url,
                                 effort=args.effort, timeout=args.timeout,
                                 max_retries=args.max_retries)
    except ProviderConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"generating with {args.provider}:{model} -> {out_path}")
    summary = run_generation(provider, specs, out_path, generator=generator,
                             concurrency=args.concurrency)
    print_summary(summary, provider, out_path)
    return 1 if summary.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
