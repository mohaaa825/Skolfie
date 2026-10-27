"""
scripts/explain.py
-------------------
Roadmap step 8: explain the verdicts on a random sample of held-out texts,
for checking by hand.

1. Samples `--n` texts from `--split` (half human, half AI, seeded).
2. For each one: the verdict and P(ai) of `--model`, the chunk heatmap
   (chunks scored by `--chunk-model`, default `--model`; with a chunk rule
   the label can be Mixed) and the feature model's top SHAP features with
   their direction and a plain-language reason (`skolfie.explain`).
3. Reasons compare each value with the human and AI medians of reference
   texts from `--reference-split` (default `val`, which the feature model
   did not train on); `--reference-n 0` skips this.
4. Writes explanations.json (everything, with the texts), review.md (a
   checklist a reviewer fills in by hand: are the reasons and the highlights
   plausible?), review.html (the same cases with the heatmap) and
   reference.json (the medians, reusable by the API).

The chunk rule (Mixed) is the one step 7 tuned, read from
`--mixed-report PATH/mixed_report.json` (rules.chunk.params, plus its chunk
size), or set with `--chunk-lo` / `--chunk-hi`. Without either, no text is
called Mixed, but the heatmap is still drawn.

Example:

    python scripts/explain.py --corpus data/processed/corpus.parquet \\
        --splits data/processed/splits.parquet \\
        --model models/stacked.joblib --features-model models/features.joblib \\
        --chunk-model models/transformer.joblib \\
        --mixed-report reports/mixed-transformer/mixed_report.json \\
        --out-dir reports/explain
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


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


def _share(value: str) -> float:
    x = float(value)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return x


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="explain.py",
        description="Explain verdicts (top SHAP features with reasons, chunk heatmap) on a "
                    "random sample of texts and write a review sheet.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", default="test", help="Split to sample from (default: test).")
    parser.add_argument("--n", type=_positive_int, default=20,
                        help="Texts to explain, half human and half AI (default: 20).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=Path, required=True,
                        help="The verdict model (joblib), e.g. stacked.joblib.")
    parser.add_argument("--features-model", type=Path,
                        help="The LightGBM feature model (features.joblib) to explain with; "
                             "default: the feature model inside --model, if any.")
    parser.add_argument("--chunk-model", type=Path,
                        help="Model that scores the chunks (default: --model).")
    parser.add_argument("--mixed-report", type=Path,
                        help="mixed_report.json from eval_mixed.py: use its tuned chunk rule "
                             "and chunk size.")
    parser.add_argument("--chunk-lo", type=_share,
                        help="Chunk rule: a chunk at or below this is clearly human.")
    parser.add_argument("--chunk-hi", type=_share,
                        help="Chunk rule: a chunk at or above this is clearly AI.")
    parser.add_argument("--target-words", type=_positive_int,
                        help="Chunk size in words (default: from --mixed-report, else 60).")
    parser.add_argument("--min-words", type=_positive_int,
                        help="Smallest chunk in words (default: from --mixed-report, else 25).")
    parser.add_argument("--top-k", type=_positive_int, default=5,
                        help="Features per text (default: 5).")
    parser.add_argument("--reference-split", default="val",
                        help="Split for the typical feature values (default: val).")
    parser.add_argument("--reference-n", type=_non_negative_int, default=400,
                        help="Reference texts, half human and half AI (default: 400; 0 = none).")
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "explain")
    return parser


def stratified_ids(df: Any, n: int, seed: int) -> list[str]:
    """Up to n ids of `df` (index = id, column label), half human and half AI, shuffled.

    A class with too few texts is topped up from the other one.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    pools = {lab: df.index[df["label"] == lab].to_numpy() for lab in ("human", "ai")}
    take = {"human": min(n // 2, len(pools["human"]))}
    take["ai"] = min(n - take["human"], len(pools["ai"]))
    take["human"] = min(n - take["ai"], len(pools["human"]))
    picked = [x for lab in ("human", "ai")
              for x in rng.choice(pools[lab], size=take[lab], replace=False)]
    return [str(x) for x in rng.permutation(np.array(picked, dtype=object))]


def _same_file(a: Any, b: Any) -> bool:
    import os

    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def reuse_or_load(verdict_model: Any, model_path: Path, path: Path) -> Any:
    """The model saved at `path`, reusing the verdict model or one of its members when it is
    the same file (a StackedModel loads features.joblib and transformer.joblib itself), so
    GPT-2 and DistilBERT are not loaded twice."""
    from skolfie.explain.explainer import load_model

    if _same_file(path, model_path):
        return verdict_model
    members = getattr(verdict_model, "members", None)
    if isinstance(members, dict) and callable(getattr(verdict_model, "_member", None)):
        for name, member_path in members.items():
            if _same_file(path, member_path):
                return verdict_model._member(name).obj
    return load_model(path)


def chunk_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Chunk size and chunk rule params from --mixed-report and the --chunk-* flags."""
    params: dict[str, Any] | None = None
    chunking: dict[str, Any] = {}
    if args.mixed_report is not None:
        if not args.mixed_report.is_file():
            parser.error(f"mixed report not found: {args.mixed_report}")
        report = json.loads(args.mixed_report.read_text(encoding="utf-8"))
        try:
            params = dict(report["rules"]["chunk"]["params"])
        except (KeyError, TypeError):
            parser.error(f"{args.mixed_report} has no rules.chunk.params")
        chunking = dict(report.get("chunking") or {})
    if args.chunk_lo is not None or args.chunk_hi is not None:
        params = dict(params or {})
        if args.chunk_lo is not None:
            params["lo"] = args.chunk_lo
        if args.chunk_hi is not None:
            params["hi"] = args.chunk_hi
    if params is not None and params.get("lo", 0.2) >= params.get("hi", 0.8):
        parser.error(f"chunk rule lo must be below hi, got {params}")
    return {
        "target_words": args.target_words or int(chunking.get("target_words") or 60),
        "min_words": args.min_words or int(chunking.get("min_words") or 25),
        "rule": params,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()
    started = time.time()

    from dataclasses import fields
    from functools import partial

    from skolfie.eval.report import (
        _json_safe,
        git_info,
        load_corpus,
        load_splits,
        load_splits_manifest,
    )
    from skolfie.explain.contributions import feature_reference
    from skolfie.explain.explainer import Explainer, find_feature_model, load_model
    from skolfie.explain.render import render_html, render_markdown
    from skolfie.mixed.chunking import chunk_spans
    from skolfie.mixed.detector import ChunkRule

    for flag, path in (("--model", args.model), ("--features-model", args.features_model),
                       ("--chunk-model", args.chunk_model)):
        if path is not None and not path.is_file():
            parser.error(f"{flag} file not found: {path}")
    settings = chunk_settings(args, parser)
    rule = None
    if settings["rule"] is not None:
        known = {f.name for f in fields(ChunkRule)}
        rule = ChunkRule(**{k: v for k, v in settings["rule"].items() if k in known})

    corpus = load_corpus(args.corpus)
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)

    def split_frame(name: str) -> Any:
        ids = splits.loc[splits["split"] == name, "id"]
        return corpus.loc[ids[ids.isin(corpus.index)].unique()]

    pool = split_frame(args.split)
    if pool.empty:
        parser.error(f"no corpus texts in split {args.split!r}")
    ids = stratified_ids(pool, args.n, args.seed)
    cases = corpus.loc[ids]
    print(f"[data] {len(ids)} texts from {args.split}: "
          + ", ".join(f"{k} {v}" for k, v in cases["label"].value_counts().items()))

    verdict_model = load_model(args.model)
    feature_model = (reuse_or_load(verdict_model, args.model, args.features_model)
                     if args.features_model is not None else find_feature_model(verdict_model))
    if feature_model is None:
        parser.error(f"{args.model} has no feature model inside; pass --features-model")

    reference, n_reference = None, 0
    if args.reference_n > 0:
        ref_pool = split_frame(args.reference_split)
        if ref_pool.empty:
            parser.error(f"no corpus texts in reference split {args.reference_split!r}")
        ref = corpus.loc[stratified_ids(ref_pool, args.reference_n, args.seed + 1)]
        print(f"[reference] typical feature values from {len(ref)} {args.reference_split} texts")
        reference = feature_reference(feature_model, ref["text"].tolist(), ref["label"].tolist())
        n_reference = len(ref)

    explainer = Explainer(
        verdict_model, feature_model,
        chunk_scorer=reuse_or_load(verdict_model, args.model, args.chunk_model)
        if args.chunk_model is not None else None,
        chunk_rule=rule, top_k=args.top_k, reference=reference,
        chunker=partial(chunk_spans, target_words=settings["target_words"],
                        min_words=settings["min_words"]),
        batch_size=args.batch_size)
    texts = cases["text"].tolist()
    t0 = time.time()
    explanations = explainer.explain_many(texts)
    explain_seconds = time.time() - t0
    print(f"[explain] {len(texts)} texts in {explain_seconds:.1f}s")

    rows = []
    for (case_id, case), ex in zip(cases.iterrows(), explanations):
        meta = {"id": str(case_id), "true_label": str(case["label"])}
        for col in ("generator", "domain", "source"):
            if col in cases.columns:
                meta[col] = str(case[col])
        rows.append({**meta, **ex, "text": str(case["text"])})

    agreement = {lab: {} for lab in ("human", "ai")}
    for row in rows:
        counts = agreement.setdefault(row["true_label"], {})
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _json_safe({
        "split": args.split, "n": len(rows), "seed": args.seed,
        "models": {"model": str(args.model),
                   "features_model": None if args.features_model is None
                   else str(args.features_model),
                   "chunk_model": None if args.chunk_model is None else str(args.chunk_model)},
        "explainer": explainer.info(),
        "chunking": {k: settings[k] for k in ("target_words", "min_words")},
        "mixed_report": None if args.mixed_report is None else str(args.mixed_report),
        "reference": {"split": args.reference_split, "n": n_reference}
        if reference is not None else None,
        "labels_by_truth": agreement,
        "corpus_version": manifest.get("corpus_version"),
        "splits_version": manifest.get("splits_version"),
        "git": git_info(),
        "seconds": {"explain": round(explain_seconds, 1),
                    "total": round(time.time() - started, 1)},
        "cases": rows,
    })
    with open(out_dir / "explanations.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
        fh.write("\n")
    if reference is not None:
        with open(out_dir / "reference.json", "w", encoding="utf-8") as fh:
            json.dump(_json_safe(reference), fh, indent=2, allow_nan=False)
            fh.write("\n")

    model_note = f"{args.model.name}" + (f", features from {args.features_model.name}"
                                         if args.features_model is not None else "")
    cases_meta = [{"id": r["id"], "true_label": r["true_label"]} for r in rows]
    (out_dir / "review.md").write_text(
        render_markdown(explanations, texts, cases_meta,
                        title=f"Explanation review: {model_note}, {len(rows)} {args.split} texts"),
        encoding="utf-8")
    captions = [f"{i}. {r['id']} (true label: {r['true_label']})"
                for i, r in enumerate(rows, 1)]
    (out_dir / "review.html").write_text(
        render_html(explanations, texts, captions, title=f"Skolfie explanations: {model_note}"),
        encoding="utf-8")

    print("[labels] by true label: " + "; ".join(
        f"{t}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
        for t, c in agreement.items()))
    print(f"[save] {out_dir / 'review.md'}, {out_dir / 'review.html'}, "
          f"{out_dir / 'explanations.json'}"
          + (f", {out_dir / 'reference.json'}" if reference is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
