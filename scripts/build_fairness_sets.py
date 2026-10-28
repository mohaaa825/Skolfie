"""
scripts/build_fairness_sets.py
-------------------------------
Build the external fairness test sets (Liang et al. 2023 and ELLIPSE).

Examples:

    # Everything, downloading what is missing into data/raw/fairness
    python scripts/build_fairness_sets.py

    # Drop any text that is also in the training corpus
    python scripts/build_fairness_sets.py --corpus data/processed/corpus.parquet

    # Only the Liang et al. sets, offline (files must already be cached)
    python scripts/build_fairness_sets.py --sources liang2023 --no-download

Writes --out (default data/processed/fairness/fairness_sets.parquet) and
fairness_manifest.json next to it, both atomically (see
skolfie/data/fairness_sets.py for the columns), and prints the count per
set. With --corpus, rows whose normalised text matches a corpus text are
dropped and counted, so nothing trained or evaluated on leaks into the
fairness sets. Exit code 1 if a download fails or no row is left, 2 on bad
arguments or an unreadable corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_NAME = "fairness_manifest.json"


def _ensure_repo_on_path() -> None:
    """Allow `python scripts/build_fairness_sets.py` from a checkout without installing."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fairness test sets (human text by non-native / native writers, "
                    "AI-edited human text, and AI text) as one Parquet file."
    )
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/fairness/fairness_sets.parquet"),
                        help="output Parquet file; the manifest goes next to it "
                             "(default: data/processed/fairness/fairness_sets.parquet)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/fairness"),
                        help="where downloaded files are kept and reused "
                             "(default: data/raw/fairness)")
    parser.add_argument("--sources", default=None,
                        help="comma-separated sources (default: all: liang2023,ellipse)")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="corpus Parquet file; drop fairness texts that also appear in it")
    parser.add_argument("--no-download", action="store_true",
                        help="never download; use only files already in --cache-dir")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="print only the output paths and errors")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    from skolfie.data import fairness_sets as fs
    from skolfie.eval.report import atomic_path

    parser = build_parser()
    args = parser.parse_args(argv)
    sources = fs.FAIRNESS_SOURCES
    if args.sources:
        sources = tuple(s.strip().lower() for s in args.sources.split(",") if s.strip())
    try:
        fs.sets_for(sources)
    except ValueError as exc:
        parser.error(str(exc))  # exits with 2

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    started = time.perf_counter()
    hashes = None
    if args.corpus is not None:
        try:
            hashes = fs.corpus_hashes(args.corpus)
        except Exception as exc:  # missing file, not Parquet, no text column
            print(f"error: cannot read corpus {args.corpus}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 2
        log(f"fairness: {len(hashes)} distinct corpus texts to check against")

    try:
        df = fs.build_fairness_sets(args.cache_dir, sources=sources, log=log,
                                    download=not args.no_download)
    except fs.FairnessDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # a cached file in an unexpected format
        print(f"error: {exc}", file=sys.stderr)
        return 1

    dropped: dict[str, int] = {}
    if hashes is not None:
        df, dropped = fs.drop_corpus_overlap(df, hashes)
        log(f"fairness: dropped {sum(dropped.values())} texts also in the corpus"
            + (f" {dropped}" if dropped else ""))
    if df.empty:
        print("error: no fairness text left; nothing written", file=sys.stderr)
        return 1

    manifest = fs.manifest(df, corpus=args.corpus, corpus_dropped=dropped)
    manifest["out"] = str(args.out)
    manifest_path = args.out.parent / MANIFEST_NAME
    table = df.copy()
    table.attrs = {}  # keep pandas from storing the build stats in the Parquet metadata
    with atomic_path(args.out) as tmp:
        table.to_parquet(tmp, index=False)
    with atomic_path(manifest_path) as tmp:
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started

    if not args.quiet:
        print(f"{'set':<24}{'n':>6}  label  writer      edited  group")
        for name, row in manifest["by_set"].items():
            print(f"{name:<24}{row['n']:>6}  {row['label']:<5}  {row['writer']:<10}  "
                  f"{'yes' if row['edited_by_ai'] else 'no':<6}  {row['group']}")
    print(f"wrote {args.out} ({len(df)} texts, {elapsed:.1f}s)")
    print(f"      {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
