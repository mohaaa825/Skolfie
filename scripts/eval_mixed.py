"""
scripts/eval_mixed.py
----------------------
Roadmap step 7: how well a trained model finds Mixed texts (part human,
part AI) and locates the AI parts.

1. Builds synthetic Mixed sets by splicing human and AI passages from the
   same source and domain (`skolfie.mixed.synth`): one from `--fit-split`
   (default `val`) to tune the decision rules, one from `--eval-split`
   (default `test`) to report on. Each set also holds pure human and pure
   AI texts, so the task is three-way: Human / AI / Mixed.
2. Scores every text and every chunk of a few sentences with the model.
3. Tunes two rules on the fit set and compares them on the eval set:
     chunk rule       Mixed when the chunk scores clearly disagree
     whole-text rule  baseline without chunks: Mixed when the whole-text
                      score is in a middle band
4. Reports 3-way macro-F1 for both rules, Mixed recall by splice pattern,
   and how well the AI spans are located (character-level F1 and the
   boundary error in words).

Example:

    python scripts/eval_mixed.py --corpus data/processed/corpus.parquet \\
        --splits data/processed/splits.parquet \\
        --model models/transformer.joblib --out-dir reports/mixed-transformer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_mixed.py",
        description="Evaluate Human / AI / Mixed detection and AI-span localisation on "
                    "synthetic spliced texts.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="A saved model (joblib) with predict_proba(texts), e.g. from "
                             "evaluate.py --save-model or stack.py --save-model.")
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--n-mixed", type=_positive_int, default=2000,
                        help="Spliced texts per set (default: 2000).")
    parser.add_argument("--n-pure", type=_positive_int, default=1000,
                        help="Pure human and pure AI texts per set, each (default: 1000).")
    parser.add_argument("--target-words", type=_positive_int, default=60,
                        help="Chunk size in words (default: 60).")
    parser.add_argument("--min-words", type=_positive_int, default=25,
                        help="Smallest chunk in words (default: 25).")
    parser.add_argument("--batch-size", type=_positive_int, default=4096,
                        help="Texts per predict_proba call (default: 4096).")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{100 * x:.1f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_md(report: dict[str, Any]) -> str:
    rules = report["rules"]
    lines = [
        f"# Mixed detection: {report['model']}",
        "",
        f"- fit set: {report['sets']['fit']['n']:,} texts from `{report['fit_split']}`, "
        f"eval set: {report['sets']['eval']['n']:,} texts from `{report['eval_split']}` "
        f"({report['sets']['eval']['n_mixed']:,} mixed, {report['sets']['eval']['n_human']:,} "
        f"human, {report['sets']['eval']['n_ai']:,} AI)",
        f"- chunks of about {report['chunking']['target_words']} words "
        f"(min {report['chunking']['min_words']}); "
        f"{report['chunking']['mean_chunks']:.1f} chunks per text on average",
        f"- corpus `{report.get('corpus_version') or 'unversioned'}`, git "
        f"`{(report.get('git') or {}).get('commit') or 'unknown'}`"[:200],
        "",
        "## Three-way result (eval set)",
        "",
    ]
    rows = []
    for name, r in rules.items():
        m = r["metrics"]
        rows.append([name, _fmt(m["macro_f1"]), _fmt(m["accuracy"]),
                     _fmt(m["per_class"]["human"]["f1"]), _fmt(m["per_class"]["ai"]["f1"]),
                     _fmt(m["per_class"]["mixed"]["f1"]), _pct(m["mixed_recall"]),
                     _pct(m["human_to_ai_rate"]), json.dumps(r["params"])])
    lines += _table(["rule", "macro-F1", "accuracy", "F1 human", "F1 AI", "F1 mixed",
                     "mixed found", "human → AI", "tuned on fit set"], rows)

    chunk = rules["chunk"]["metrics"]["confusion"]
    lines += ["", "### Confusion, chunk rule (rows: truth, columns: prediction)", ""]
    labels = ("human", "ai", "mixed")
    lines += _table(["truth"] + list(labels),
                    [[t] + [f"{chunk.get(t, {}).get(p, 0):,}" for p in labels] for t in labels])

    lines += ["", "### Mixed found, by splice pattern", "",
              "h = human part, a = AI part, in reading order.", ""]
    lines += _table(["pattern", "n", "chunk rule", "whole-text rule"],
                    [[p, f"{v['n']:,}", _pct(v["chunk"]), _pct(v["whole_text"])]
                     for p, v in report["by_pattern"].items()])

    s = report["spans"]
    lines += ["", "## Locating the AI parts (mixed texts in the eval set)", "",
              "Characters inside chunks scored P(ai) >= 0.5 are predicted AI; whitespace is "
              "ignored.", ""]
    lines += _table(["metric", "value"], [
        ["character precision", _fmt(s.get("char_precision"))],
        ["character recall", _fmt(s.get("char_recall"))],
        ["character F1 (all texts pooled)", _fmt(s.get("char_f1"))],
        ["character F1 (mean per text)", _fmt(s.get("macro_f1"))],
        ["boundary error, words (mean)", _fmt(s.get("boundary_error_mean"), 1)],
        ["boundary error, words (median)", _fmt(s.get("boundary_error_median"), 1)],
        ["texts with one boundary", f"{s.get('n_boundary_texts', 0):,}"],
    ])
    lines += ["", f"Scoring took {report['seconds']['score']:.0f}s; total "
              f"{report['seconds']['total']:.0f}s."]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()
    started = time.time()

    from functools import partial

    import numpy as np
    import pandas as pd

    from skolfie.eval.report import (
        _json_safe,
        git_info,
        load_corpus,
        load_splits,
        load_splits_manifest,
    )
    from skolfie.mixed.chunking import chunk_spans
    from skolfie.mixed.detector import (
        ChunkRule,
        WholeTextRule,
        fit_rule,
        score_texts,
        span_metrics,
        three_way_metrics,
    )
    from skolfie.mixed.synth import make_mixed_set
    from skolfie.models.baselines import JoblibScorer

    if not args.model.is_file():
        parser.error(f"model file not found: {args.model}")
    corpus = load_corpus(args.corpus)
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)
    sets: dict[str, pd.DataFrame] = {}
    for key, split, seed in (("fit", args.fit_split, args.seed),
                             ("eval", args.eval_split, args.seed + 1)):
        ids = splits.loc[splits["split"] == split, "id"]
        ids = ids[ids.isin(corpus.index)].tolist()
        if not ids:
            parser.error(f"no corpus texts in split {split!r}")
        sets[key] = make_mixed_set(corpus, ids, n_mixed=args.n_mixed, n_pure=args.n_pure,
                                   seed=seed).reset_index(drop=True)
        counts = sets[key]["label3"].value_counts()
        print(f"[data] {key} set from {split}: " + ", ".join(
            f"{k} {v:,}" for k, v in counts.items()))

    scorer = JoblibScorer(args.model)
    chunker = partial(chunk_spans, target_words=args.target_words, min_words=args.min_words)
    t0 = time.time()
    scored = {key: score_texts(scorer, df["text"].tolist(), chunker=chunker,
                               batch_size=args.batch_size)
              for key, df in sets.items()}
    score_seconds = time.time() - t0

    fit_df, eval_df = sets["fit"], sets["eval"]
    rules: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for name, cls in (("chunk", ChunkRule), ("whole_text", WholeTextRule)):
        rule, fit_f1 = fit_rule(cls, scored["fit"], fit_df["label3"].tolist())
        preds[name] = rule.predict(scored["eval"])
        rules[name] = {"params": {k: v for k, v in vars(rule).items()},
                       "fit_macro_f1": fit_f1,
                       "metrics": three_way_metrics(eval_df["label3"].tolist(), preds[name])}
        print(f"[rule] {name}: tuned {rules[name]['params']}, eval macro-F1 "
              f"{rules[name]['metrics']['macro_f1']:.3f}")

    mixed = eval_df["label3"] == "mixed"
    by_pattern = {}
    for pattern, idx in eval_df[mixed].groupby("pattern").groups.items():
        rows = np.asarray(list(idx))
        by_pattern[pattern] = {"n": int(len(rows)),
                               "chunk": float((preds["chunk"][rows] == "mixed").mean()),
                               "whole_text": float((preds["whole_text"][rows] == "mixed").mean())}
    mixed_rows = np.flatnonzero(mixed.to_numpy())
    spans = span_metrics([eval_df.at[i, "text"] for i in mixed_rows],
                         [json.loads(eval_df.at[i, "ai_spans"]) for i in mixed_rows],
                         [scored["eval"][i] for i in mixed_rows])

    n_chunks = [len(s["chunks"]) for s in scored["eval"]]
    report = _json_safe({
        "model": str(args.model.name), "model_path": str(args.model),
        "fit_split": args.fit_split, "eval_split": args.eval_split,
        "corpus_version": manifest.get("corpus_version"),
        "splits_version": manifest.get("splits_version"),
        "sets": {key: {"n": len(df), **{f"n_{k}": int((df["label3"] == k).sum())
                                        for k in ("human", "ai", "mixed")}}
                 for key, df in sets.items()},
        "chunking": {"target_words": args.target_words, "min_words": args.min_words,
                     "mean_chunks": float(np.mean(n_chunks))},
        "rules": rules, "by_pattern": by_pattern, "spans": spans, "seed": args.seed,
        "git": git_info(),
        "seconds": {"score": round(score_seconds, 1),
                    "total": round(time.time() - started, 1)},
    })
    out_dir = args.out_dir or Path("reports") / f"mixed-{args.model.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mixed_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write("\n")
    md = render_md(report)
    (out_dir / "mixed_report.md").write_text(md, encoding="utf-8")
    pd.DataFrame({
        "id": eval_df["id"], "label3": eval_df["label3"], "pattern": eval_df["pattern"],
        "pred_chunk": preds["chunk"], "pred_whole_text": preds["whole_text"],
        "p_whole": [s["p"] for s in scored["eval"]],
        "chunks": [json.dumps([[int(a), int(b), round(float(p), 4)] for a, b, p in s["chunks"]])
                   for s in scored["eval"]],
        "ai_spans": eval_df["ai_spans"], "text": eval_df["text"],
    }).to_parquet(out_dir / "mixed_predictions.parquet", index=False)
    print()
    print(md, end="")
    print(f"\n[save] {out_dir / 'mixed_report.md'}, {out_dir / 'mixed_report.json'}, "
          f"{out_dir / 'mixed_predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
