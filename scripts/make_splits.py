"""
scripts/make_splits.py
-----------------------
Make the held-out evaluation splits of a built corpus.

Example:

    python scripts/make_splits.py \\
        --corpus data/processed/corpus.parquet \\
        --config configs/splits.json \\
        --out-dir data/processed

Writes splits.parquet (id, split; one row per membership), splits_manifest.json
and splits.md to --out-dir (see skolfie/eval/splits.py) and prints the
summary. The `text` column is never read. Exit code 1 if a split guarantee
fails (nothing is written), 2 on bad arguments or unreadable inputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    """Allow `python scripts/make_splits.py` from a checkout without installing."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Make deterministic train/val/test and held-out splits of the corpus."
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus.parquet"),
                        help="corpus Parquet file (default: data/processed/corpus.parquet)")
    parser.add_argument("--config", type=Path, default=Path("configs/splits.json"),
                        help="split config JSON (default: configs/splits.json)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"),
                        help="where splits.parquet, splits_manifest.json, splits.md go "
                             "(default: data/processed)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="print only the output paths and warnings, not the summary")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    import pandas as pd

    from skolfie.eval import splits

    args = build_parser().parse_args(argv)
    try:
        config = splits.load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: cannot read config {args.config}: {exc}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        df = pd.read_parquet(args.corpus, columns=list(splits.REQUIRED_COLUMNS))
    except Exception as exc:  # missing file, not Parquet, missing columns
        print(f"error: cannot read corpus {args.corpus}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    try:
        memberships = splits.make_splits(df, config)
    except splits.SplitLeakError as exc:
        print(f"error: split guarantee violated: {exc}; nothing written", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = splits.write_splits(df, memberships, config, args.out_dir, corpus_path=args.corpus)
    elapsed = time.perf_counter() - started

    if args.quiet:
        for w in result.manifest["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
    else:
        print(result.summary_markdown)
    print(f"wrote {result.parquet_path} ({len(memberships)} memberships of {len(df)} rows, "
          f"{result.splits_version}, {elapsed:.1f}s)")
    print(f"      {result.manifest_path}")
    print(f"      {result.summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
