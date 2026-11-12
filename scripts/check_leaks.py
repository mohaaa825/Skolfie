"""
scripts/check_leaks.py
----------------------
Leak report for existing splits: how many evaluation texts have an exact or
near duplicate (MinHash Jaccard >= --threshold, `skolfie.eval.near_dup`) in
`train`. Needs no GPU and no models; about 2 minutes for 270k texts.

Example:

    python scripts/check_leaks.py \\
        --corpus "/content/drive/MyDrive/skolfie/data/processed_modern/corpus.parquet" \\
        --splits "/content/drive/MyDrive/skolfie/data/processed_modern/splits.parquet" \\
        --out-dir "/content/drive/MyDrive/skolfie/reports_modern/leaks"

Writes leaks.md and leaks.json (counts per evaluation split, label and source,
plus up to --examples example pairs) to --out-dir and prints the markdown.
Splits made with `near_duplicate_threshold` in their config should show 0
near-duplicate leaks at that threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Report train/test overlap (exact and near duplicates).")
    p.add_argument("--corpus", type=Path, required=True, help="corpus.parquet")
    p.add_argument("--splits", type=Path, required=True, help="splits.parquet")
    p.add_argument("--out-dir", type=Path, help="write leaks.md and leaks.json here")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="MinHash Jaccard estimate counted as a near duplicate (default 0.8)")
    p.add_argument("--examples", type=int, default=10, help="example pairs to show (default 10)")
    return p


def leak_report(df, memberships, threshold: float, n_examples: int = 10) -> dict:
    """Counts of evaluation rows with an exact / near duplicate in train."""
    import numpy as np
    import pandas as pd

    from skolfie.eval.near_dup import cluster_ids

    df = df.reset_index(drop=True)
    norm = df["text"].astype(str).str.lower().str.split().str.join(" ")
    clusters = cluster_ids(df["text"].astype(str).tolist(), threshold=threshold)
    pos = pd.Index(df["id"]).get_indexer(memberships["id"])
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} split ids are not in the corpus")
    split_of = pd.DataFrame({"pos": pos, "split": memberships["split"].to_numpy()})
    train_pos = split_of.loc[split_of["split"] == "train", "pos"].to_numpy()

    train_texts = set(norm.iloc[train_pos])
    train_clusters = {}
    for p in train_pos:
        c = clusters[p]
        if c >= 0 and c not in train_clusters:
            train_clusters[c] = p

    out: dict = {"threshold": threshold, "n_corpus": int(len(df)),
                 "n_train": int(len(train_pos)), "splits": {}, "examples": []}
    for split, grp in split_of[split_of["split"] != "train"].groupby("split", sort=True):
        p = grp["pos"].to_numpy()
        exact = norm.iloc[p].isin(train_texts).to_numpy()
        near = np.array([clusters[i] >= 0 and clusters[i] in train_clusters for i in p])
        leak = exact | near
        sub = df.iloc[p][["label", "source"]].assign(leak=leak)
        by = (sub[sub["leak"]].groupby(["label", "source"]).size()
              .rename("n").reset_index().to_dict(orient="records"))
        out["splits"][split] = {"n": int(len(p)), "exact": int(exact.sum()),
                                "near": int((near & ~exact).sum()), "leaked": int(leak.sum()),
                                "share": float(leak.mean()) if len(p) else 0.0, "by": by}
        for i in p[leak]:
            if len(out["examples"]) >= n_examples:
                break
            j = train_clusters.get(clusters[i])
            if j is None:
                continue
            out["examples"].append({"split": split, "id": str(df.at[i, "id"]),
                                    "train_id": str(df.at[j, "id"]),
                                    "text": str(df.at[i, "text"])[:160],
                                    "train_text": str(df.at[j, "text"])[:160]})
    return out


def render(report: dict) -> str:
    lines = [f"# Train/test leak check (near duplicate: Jaccard >= {report['threshold']})", "",
             f"- corpus rows: {report['n_corpus']:,}; train rows: {report['n_train']:,}", "",
             "| split | texts | exact dup in train | near dup in train | leaked share |",
             "|---|---|---|---|---|"]
    for split, r in report["splits"].items():
        lines.append(f"| {split} | {r['n']:,} | {r['exact']:,} | {r['near']:,} | "
                     f"{100 * r['share']:.2f}% |")
    lines += ["", "## Leaked texts by label and source", "", "| split | label | source | n |",
              "|---|---|---|---|"]
    for split, r in report["splits"].items():
        for row in r["by"]:
            lines.append(f"| {split} | {row['label']} | {row['source']} | {row['n']:,} |")
    if report["examples"]:
        lines += ["", "## Examples", ""]
        for ex in report["examples"]:
            lines += [f"- **{ex['split']}** `{ex['id']}` ~ train `{ex['train_id']}`",
                      f"  - {ex['text']!r}", f"  - {ex['train_text']!r}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_repo_on_path()
    import pandas as pd

    args = build_parser().parse_args(argv)
    if not 0.0 < args.threshold <= 1.0:
        print("error: --threshold must be in (0, 1]", file=sys.stderr)
        return 2
    try:
        df = pd.read_parquet(args.corpus, columns=["id", "text", "label", "source"])
        memberships = pd.read_parquet(args.splits, columns=["id", "split"])
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"[leaks] {len(df):,} texts, {len(memberships):,} split memberships; "
          "finding near duplicates ...", flush=True)
    report = leak_report(df, memberships, args.threshold, args.examples)
    md = render(report)
    print(md)
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "leaks.md").write_text(md, encoding="utf-8")
        (args.out_dir / "leaks.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"wrote {args.out_dir / 'leaks.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
