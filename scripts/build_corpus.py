"""
scripts/build_corpus.py
------------------------
Build the versioned training corpus from the registered public datasets.

Examples:

    # Everything registered, downloading what is missing into data/raw
    python scripts/build_corpus.py

    # Two sources, offline (files must already be in data/raw), balanced
    python scripts/build_corpus.py --sources hc3,m4gt --no-download --balance downsample

    # Stop on the first source that fails instead of skipping it
    python scripts/build_corpus.py --strict

    # Modern data: PERSUADE student essays (capped at 4,000) + generated essays
    python scripts/build_corpus.py --sources hc3,m4gt,autextification,persuade,fresh \\
        --persuade-max-essays 4000 --out-dir data/processed_modern

Writes data/processed/corpus.parquet, manifest.json and stats.md (see
skolfie/data/corpus.py) and prints the stats table. A source that fails is
reported with its error and skipped unless --strict. The exit code is 1 when
no record survives (or with --strict when a source fails), 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    """Allow `python scripts/build_corpus.py` from a checkout without installing."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _drop_error_records(record: logging.LogRecord) -> bool:
    return record.levelno < logging.ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the skolfie training corpus (Parquet + manifest) from public datasets."
    )
    parser.add_argument("--sources", default=None,
                        help="comma-separated source names (default: all registered)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="where source files are downloaded / looked for (default: data/raw)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"),
                        help="where corpus.parquet, manifest.json, stats.md go "
                             "(default: data/processed)")
    parser.add_argument("--min-words", type=int, default=30,
                        help="drop texts with fewer words (default: 30)")
    parser.add_argument("--max-words", type=int, default=2000,
                        help="drop texts with more words (default: 2000)")
    parser.add_argument("--balance", choices=("none", "downsample"), default="none",
                        help="downsample the majority label per domain (default: none)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for --balance")
    parser.add_argument("--no-download", action="store_true",
                        help="never download; use only files already in --raw-dir")
    parser.add_argument("--strict", action="store_true",
                        help="abort if any source fails instead of skipping it")
    parser.add_argument("--persuade-max-essays", type=int, default=None, metavar="N",
                        help="keep at most N PERSUADE essays, equally many per prompt "
                             "(default: 4000; 0 keeps all 25k)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors and stats")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    from skolfie.data import corpus

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Source errors are printed below, once, in a fixed format.
    corpus_log = logging.getLogger(corpus.__name__)
    if _drop_error_records not in corpus_log.filters:
        corpus_log.addFilter(_drop_error_records)

    try:
        registry = corpus.load_registry()
    except Exception as exc:  # e.g. the sources package is missing or broken
        print(f"error: could not load the source registry: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    if args.sources:
        names = [n.strip() for n in args.sources.split(",") if n.strip()]
    else:
        names = list(registry)
    unknown = [n for n in names if n not in registry]
    if unknown or not names:
        parser.error(
            f"unknown source(s): {', '.join(unknown) or '(none given)'}; "
            f"registered: {', '.join(sorted(registry)) or '(none)'}"
        )

    source_options = {}
    if args.persuade_max_essays is not None:
        if args.persuade_max_essays < 0:
            parser.error("--persuade-max-essays must be >= 0 (0 keeps all)")
        if "persuade" in names:
            source_options["persuade"] = {"max_essays": args.persuade_max_essays,
                                          "seed": args.seed}
        else:
            print("warning: --persuade-max-essays ignored: persuade is not in --sources",
                  file=sys.stderr)
    elif "persuade" in names:
        source_options["persuade"] = {"seed": args.seed}

    try:
        result = corpus.build_corpus(
            names,
            args.raw_dir,
            args.out_dir,
            min_words=args.min_words,
            max_words=args.max_words,
            download=not args.no_download,
            seed=args.seed,
            balance=args.balance,
            strict=args.strict,
            registry=registry,
            source_options=source_options,
        )
    except corpus.SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("aborted (--strict); nothing written", file=sys.stderr)
        return 1
    except corpus.EmptyCorpusError as exc:
        for name, stats in exc.source_stats.items():
            if stats.get("error"):
                print(f"error: {stats['error']}", file=sys.stderr)
            else:
                print(f"{name}: {stats['raw']} raw records, 0 kept "
                      f"(too_short={stats['too_short']}, too_long={stats['too_long']}, "
                      f"invalid={stats['invalid']})", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        parser.error(str(exc))

    for message in result.errors.values():
        print(f"error: {message}", file=sys.stderr)
    if result.errors:
        print(f"warning: built without {', '.join(result.errors)}", file=sys.stderr)

    print(result.stats_markdown)
    print(f"wrote {result.parquet_path} ({result.n_records} records, {result.corpus_version})")
    print(f"      {result.manifest_path}")
    print(f"      {result.stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
