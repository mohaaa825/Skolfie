"""
scripts/stack.py
-----------------
Roadmap step 6: combine models into one calibrated score and set the
Human / Uncertain / AI cut-offs, from predictions the models already saved.

Each member is a `predictions.parquet` written by
`scripts/evaluate.py --save-predictions` (columns id, split, label, p_ai).
The stacker is fitted on `--fit-split` (default `val`, which no member
trained on); that split's own report row uses out-of-fold scores. The
result is a normal report card (report.json / report.md, so `evaluate.py
--compare` includes it) with an extra verdict table per split.

Example:

    python scripts/stack.py --corpus data/processed/corpus.parquet \\
        --splits data/processed/splits.parquet \\
        --member features=reports/features/predictions.parquet \\
        --member transformer=reports/transformer/predictions.parquet \\
        --out-dir reports/stacked

    # Also save a ready-to-use scorer that loads the member models itself
    python scripts/stack.py ... \\
        --model-path features=models/features.joblib \\
        --model-path transformer=models/transformer.joblib \\
        --save-model models/stacked.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _name_path(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def _fraction(value: str) -> float:
    x = float(value)
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stack.py",
        description="Fit a calibrated stack of saved model predictions, set verdict cut-offs "
                    "and write a report card.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--member", type=_name_path, action="append", required=True,
                        metavar="NAME=PREDICTIONS",
                        help="A member's predictions.parquet (repeat for each member).")
    parser.add_argument("--fit-split", default="val",
                        help="Split to fit the stacker and cut-offs on (default: val).")
    parser.add_argument("--calibration", choices=("platt", "isotonic"), default="platt")
    parser.add_argument("--max-fpr", type=_fraction, default=0.01,
                        help="Share of human texts (on the fit split) allowed above the AI "
                             "cut-off (default: 0.01).")
    parser.add_argument("--max-miss", type=_fraction, default=0.05,
                        help="Share of AI texts (on the fit split) allowed below the human "
                             "cut-off (default: 0.05).")
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "stacked")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--model-path", type=_name_path, action="append", default=[],
                        metavar="NAME=JOBLIB",
                        help="A member's saved model (for --save-model).")
    parser.add_argument("--save-model", type=Path, metavar="PATH",
                        help="Save a StackedModel scorer that loads the --model-path files.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


class _LookupScorer:
    """Scores texts from a precomputed {text key: p} table, so run_evaluation can report it."""

    pretrained = True
    name = "stacked"

    def __init__(self, table: dict[str, float], params: dict[str, Any]) -> None:
        self.table = table
        self._params = params

    def params(self) -> dict[str, Any]:
        return self._params

    def predict_proba(self, texts: Sequence[str]) -> Any:
        import numpy as np

        from skolfie.features.cache import text_key

        return np.array([self.table[text_key(t)] for t in texts], dtype=np.float64)


# A text in two splits (test and test_short overlap) is scored twice. GPU
# models in fp16 give it slightly different scores depending on what else was
# in its batch; anything larger than this means the file mixes different runs.
MAX_SCORE_SPREAD = 0.05


def load_member(path: Path) -> Any:
    """One P(ai) per id from a predictions file (the mean when scored more than once)."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["id", "p_ai"])
    per_id = df.groupby("id")["p_ai"]
    spread = per_id.max() - per_id.min()
    if spread.max() > MAX_SCORE_SPREAD:
        raise ValueError(f"{path}: the same id has scores up to {spread.max():.3f} apart in "
                         "different splits; is the file from more than one run?")
    if spread.max() > 0:
        print(f"[note] {path}: {int((spread > 0).sum()):,} ids scored slightly differently in "
              f"different splits (max {spread.max():.1e}); using the mean")
    return per_id.mean()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()

    import pandas as pd

    from skolfie.eval.metrics import to_binary
    from skolfie.eval.report import (
        EVAL_SPLITS,
        load_corpus,
        load_splits,
        load_splits_manifest,
        run_evaluation,
        write_report,
    )
    from skolfie.features.cache import text_key
    from skolfie.models.stacker import StackedModel, Stacker, operating_points, verdict_table

    names = [n for n, _ in args.member]
    if len(set(names)) != len(names):
        parser.error("member names must be unique")
    for _, path in args.member:
        if not path.is_file():
            parser.error(f"predictions file not found: {path}")
    model_paths = dict(args.model_path)
    if args.save_model and set(model_paths) != set(names):
        parser.error("--save-model needs one --model-path per --member (same names)")

    corpus = load_corpus(args.corpus)
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)

    scores = pd.concat({n: load_member(p) for n, p in args.member}, axis=1, join="inner")
    covered = scores.index.intersection(corpus.index)
    scores = scores.loc[covered, names]
    splits = splits[splits["split"] != "train"]  # members trained on it: never scored
    dropped = int((~splits["id"].isin(covered)).sum())
    if dropped:
        print(f"[note] {dropped:,} evaluation rows have no score from every member; left out")
    splits = splits[splits["id"].isin(covered)]
    print(f"[data] {len(covered):,} texts scored by all of: {', '.join(names)}")

    fit_ids = splits.loc[splits["split"] == args.fit_split, "id"].unique()
    if len(fit_ids) == 0:
        parser.error(f"no scored texts in fit split {args.fit_split!r}")
    P_fit = scores.loc[fit_ids].to_numpy()
    y_fit = to_binary(corpus.loc[fit_ids, "label"].tolist())

    stacker = Stacker(calibration=args.calibration, seed=args.seed).fit(P_fit, y_fit)
    p_all = pd.Series(stacker.predict_proba(scores.to_numpy()), index=scores.index)
    oof = stacker.oof_predict(P_fit, y_fit)
    p_all.loc[fit_ids] = oof  # the fit split is reported out-of-fold
    cutoffs = operating_points(y_fit, oof, max_fpr=args.max_fpr, max_miss=args.max_miss)
    weights = stacker.weights(names)
    print("[stack] weights: " + ", ".join(f"{k} {v:+.3f}" for k, v in weights.items()))
    print(f"[stack] cut-offs: AI above {cutoffs['ai_threshold']:.4f}, human below "
          f"{cutoffs['human_threshold']:.4f}")

    texts = corpus.loc[covered, "text"]
    table = {text_key(t): float(p) for t, p in zip(texts, p_all.loc[covered])}
    info = {"members": names, "fit_split": args.fit_split, "n_fit": int(len(fit_ids)),
            "calibration": args.calibration, "weights": weights, "cutoffs": cutoffs,
            "note": f"{args.fit_split} scores are out-of-fold ({stacker.folds}-fold)"}
    eval_splits = [args.fit_split] + [s for s in EVAL_SPLITS if s != args.fit_split]
    report, predictions = run_evaluation(
        _LookupScorer(table, info), corpus, splits, model_spec="stack:" + "+".join(names),
        corpus_path=args.corpus, splits_path=args.splits, splits_manifest=manifest,
        seed=args.seed, eval_splits=eval_splits)
    report["verdicts"] = {"cutoffs": cutoffs, "splits": {
        split: verdict_table(df["label"].tolist(), df["p_ai"].to_numpy(), cutoffs)
        for split, df in predictions.groupby("split", sort=False)}}
    paths = write_report(report, args.out_dir,
                         predictions if args.save_predictions else None)

    if args.save_model:
        model = StackedModel({n: model_paths[n] for n in names}, stacker, cutoffs, info)
        import joblib

        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.save_model)
        paths["model"] = args.save_model

    print()
    print(paths["md"].read_text(encoding="utf-8"), end="")
    print(f"\n[save] {', '.join(str(p) for p in paths.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
