"""
scripts/evaluate.py
--------------------
Held-out evaluation harness: one command, one report card per model.

Examples:

    # Train a baseline on `train`, score val + every test split, write reports/tfidf/
    python scripts/evaluate.py --corpus data/corpus/corpus.parquet \\
        --splits data/splits/splits.parquet --model tfidf

    # The existing ensemble is slow: train on 5k texts, score at most 2k per split
    python scripts/evaluate.py --corpus ... --splits ... --model ensemble \\
        --max-train 5000 --max-eval 2000

    # Feature model (roadmap step 4): style + GPT-2 + Delta features into LightGBM.
    # GPT-2 needs a GPU for the full corpus; cache features so re-runs are fast.
    python scripts/evaluate.py --corpus ... --splits ... --model features \\
        --cache-dir data/cache --save-model models/features.joblib

    # DistilBERT fine-tuned with the same style + GPT-2 features (step 5; GPU)
    python scripts/evaluate.py --corpus ... --splits ... --model transformer \\
        --cache-dir data/cache --save-model models/transformer.joblib

    # Any pickled object with predict_proba(texts) (or a saved AIDetectorModel)
    python scripts/evaluate.py --corpus ... --splits ... \\
        --model joblib:models/detector_model.joblib --out-dir reports/ensemble-v1

    # Adversarial training (step 9): also train on attacked copies of train texts
    python scripts/evaluate.py --corpus ... --splits ... --model transformer \\
        --extra-train reports/robust/attacks-train/attacks.parquet \\
        --out-dir reports/transformer-adv --save-model models/transformer-adv.joblib

    # Modern data: repeat the few fresh (current-model) train texts 6 times
    python scripts/evaluate.py --corpus ... --splits ... --model features \\
        --oversample fresh=6

    # Feature model plus Binoculars features from a modern Qwen pair (GPU, ~10x
    # slower than GPT-2): train on 40k texts, never sampling away the modern essays
    python scripts/evaluate.py --corpus ... --splits ... --model features \\
        --bino-observer Qwen/Qwen2.5-1.5B --bino-performer Qwen/Qwen2.5-1.5B-Instruct \\
        --max-train 40000 --keep-sources fresh,persuade --cache-dir data/cache

    # Compare existing reports on the headline metrics
    python scripts/evaluate.py --compare reports/*/report.json

The report card (report.md) is printed and written to --out-dir together
with report.json and, with --save-predictions, predictions.parquet.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CHOICES = (
    "length | tfidf | features | features-nolm | features-rf | features-nolm-rf | ensemble | "
    "transformer | transformer-text | transformer-chunks | ensemble:PATH | joblib:PATH"
)
FEATURE_MODELS = ("features", "features-nolm", "features-rf", "features-nolm-rf")
NAMED_MODELS = (*FEATURE_MODELS, "transformer", "transformer-text", "transformer-chunks")
# The ensemble calls predict() per text and extracts NLTK features: keep it bounded.
ENSEMBLE_DEFAULT_MAX_TRAIN = 5000
ENSEMBLE_DEFAULT_MAX_EVAL = 2000


def _ensure_repo_on_path() -> None:
    """Allow `python scripts/evaluate.py` from a checkout without installing."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _model_spec(value: str) -> str:
    if value in ("length", "tfidf", "ensemble", *NAMED_MODELS):
        return value
    if re.match(r"^(joblib|ensemble):.+", value):
        return value
    raise argparse.ArgumentTypeError(f"invalid model {value!r}; use {MODEL_CHOICES}")


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def parse_oversample(values: Sequence[str]) -> dict[str, int]:
    """`["fresh=6", "persuade=1,x=2"]` -> {"fresh": 6, "persuade": 1, "x": 2}."""
    out: dict[str, int] = {}
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            name, sep, factor = item.partition("=")
            name = name.strip()
            try:
                n = int(factor)
            except ValueError:
                n = 0
            if not sep or not name or n < 1:
                raise argparse.ArgumentTypeError(
                    f"invalid --oversample {item!r}; use SOURCE=N with an integer N >= 1, "
                    "e.g. fresh=6")
            if name in out and out[name] != n:
                raise argparse.ArgumentTypeError(f"--oversample gives {name!r} twice")
            out[name] = n
    return out


def parse_keep_sources(values: Sequence[str]) -> list[str]:
    """`["fresh,persuade", "x"]` -> ["fresh", "persuade", "x"] (order kept, no repeats)."""
    out: list[str] = []
    for value in values:
        out += [s for s in (item.strip() for item in value.split(",")) if s and s not in out]
    return out


def default_out_dir(spec: str) -> Path:
    if ":" in spec:
        kind, path = spec.split(":", 1)
        return Path("reports") / f"{kind}-{Path(path).stem}"
    return Path("reports") / spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Train a scorer on the train split, evaluate it on val and every test "
                    "split, and write/print a report card.",
    )
    parser.add_argument("--corpus", type=Path, help="Corpus Parquet file (skolfie schema).")
    parser.add_argument("--splits", type=Path,
                        help="splits.parquet (id, split); splits_manifest.json is read from "
                             "the same directory if present.")
    parser.add_argument("--model", type=_model_spec, metavar="MODEL",
                        help=f"Scorer to evaluate: {MODEL_CHOICES}.")
    parser.add_argument("--out-dir", type=Path,
                        help="Where to write report.json/report.md (default: reports/<model>).")
    parser.add_argument("--max-train", type=_positive_int,
                        help="Sample the train split down to N texts (stratified, seeded). "
                             f"Default: all; {ENSEMBLE_DEFAULT_MAX_TRAIN} for ensemble.")
    parser.add_argument("--max-eval", type=_positive_int,
                        help="Sample each evaluation split down to N texts (stratified by "
                             f"label, seeded). Default: all; {ENSEMBLE_DEFAULT_MAX_EVAL} for "
                             "ensemble.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling/model seed (default: 0).")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write predictions.parquet (id, split, label, p_ai).")
    parser.add_argument("--save-model", type=Path, metavar="PATH",
                        help="Also save the fitted scorer with joblib (reload it with "
                             "--model joblib:PATH).")
    parser.add_argument("--extra-train", type=Path, action="append", default=[],
                        metavar="PARQUET",
                        help="Extra training rows (columns text, label), appended to the train "
                             "split before fitting, e.g. attacked copies from make_attacks.py "
                             "--share for adversarial training. Repeatable. Rows whose text (or "
                             "orig_id) is in an evaluation split are dropped.")
    parser.add_argument("--oversample", action="append", default=[], metavar="SOURCE=N",
                        help="Repeat the train rows of corpus source SOURCE N times in all "
                             "(e.g. fresh=6 for the few texts from current models). "
                             "Comma-separated or repeatable. Evaluation splits are unchanged.")
    parser.add_argument("--keep-sources", action="append", default=[], metavar="SOURCES",
                        help="Corpus sources whose rows --max-train / --max-eval never sample "
                             "away (e.g. fresh,persuade); only the other rows are sampled, "
                             "stratified by label, to fill the rest. Comma-separated or "
                             "repeatable.")
    group = parser.add_argument_group("feature and transformer models (features, transformer, ...)")
    group.add_argument("--cache-dir", type=Path,
                       help="Cache per-text style/GPT-2 features here (parquet keyed by text "
                            "hash), so re-runs skip the slow part.")
    group.add_argument("--lm-model", help="Causal LM for the lm_* features (default: gpt2).")
    group.add_argument("--lm-max-tokens", type=_positive_int,
                       help="Truncate texts to N tokens for the LM (default: 512).")
    group.add_argument("--lm-batch-size", type=_positive_int, help="LM batch size (default: 32).")
    group.add_argument("--lm-device", help="cuda / cpu (default: cuda if available).")
    group.add_argument("--bino-observer", metavar="HF_ID",
                       help="Feature models only: add Binoculars features (bino_*) from this "
                            "observer LM, e.g. Qwen/Qwen2.5-1.5B (default: off). Needs a GPU; "
                            "about 10x slower than GPT-2, so combine with --max-train and "
                            "--keep-sources.")
    group.add_argument("--bino-performer", metavar="HF_ID",
                       help="Performer LM with the observer's tokenizer, e.g. "
                            "Qwen/Qwen2.5-1.5B-Instruct (default: none = observer-only "
                            "features, no cross-perplexity / Binoculars score).")
    group.add_argument("--bino-max-tokens", type=_positive_int,
                       help="Truncate texts to N tokens for Binoculars (default: 256).")
    group.add_argument("--bino-batch-size", type=_positive_int,
                       help="Binoculars batch size (default: 8).")
    group.add_argument("--bino-quantization", choices=("4bit", "8bit"),
                       help="Load the Binoculars models in 4-bit (nf4) or 8-bit with "
                            "bitsandbytes, e.g. for a 27B observer on a 24 GB GPU "
                            "(default: fp16 on CUDA).")
    group.add_argument("--n-jobs", type=int,
                       help="Worker processes for style features and the classifier "
                            "(default: -1 = all cores).")
    group.add_argument("--hf-model",
                       help="Transformer backbone (default: distilbert-base-cased).")
    group.add_argument("--max-length", type=_positive_int,
                       help="Transformer input length in tokens (default: 512).")
    group.add_argument("--epochs", type=_positive_int, help="Transformer epochs (default: 1).")
    group.add_argument("--batch-size", type=_positive_int,
                       help="Transformer batch size (default: 16, x2 gradient accumulation).")
    parser.add_argument("--compare", nargs="+", metavar="REPORT_JSON",
                        help="Only print a comparison table of these report.json files "
                             "(globs are expanded).")
    return parser


def _expand(patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        paths += [m for m in matches if m not in paths]
    return paths


def load_extra_train(paths: Sequence[Path], parser: argparse.ArgumentParser) -> Any:
    """The --extra-train files as one frame: text, label, extra_source (+ orig_id, attack)."""
    import pandas as pd

    frames = []
    for path in paths:
        if not path.is_file():
            parser.error(f"--extra-train file not found: {path}")
        df = pd.read_parquet(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            parser.error(f"--extra-train {path} is missing column(s): "
                         f"{', '.join(sorted(missing))}")
        cols = [c for c in ("text", "label", "orig_id", "attack") if c in df.columns]
        frames.append(df[cols].assign(extra_source=str(path)))
    return pd.concat(frames, ignore_index=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()

    from skolfie.eval.report import compare_reports

    if args.compare:
        paths = _expand(args.compare)
        missing = [p for p in paths if not Path(p).is_file()]
        if missing or not paths:
            parser.error(f"report file(s) not found: {', '.join(missing) or args.compare}")
        try:
            print(compare_reports(paths), end="")
        except ValueError as exc:  # also malformed JSON
            parser.error(str(exc))
        return 0

    for flag in ("corpus", "splits", "model"):
        if getattr(args, flag) is None:
            parser.error(f"--{flag} is required (unless --compare is given)")
    for flag in ("corpus", "splits"):
        if not getattr(args, flag).is_file():
            parser.error(f"--{flag} file not found: {getattr(args, flag)}")

    if args.model == "ensemble":
        if args.max_train is None:
            args.max_train = ENSEMBLE_DEFAULT_MAX_TRAIN
            print(f"[note] ensemble is slow: --max-train defaults to {args.max_train}")
    if args.model == "ensemble" or args.model.startswith("ensemble:"):
        if args.max_eval is None:
            args.max_eval = ENSEMBLE_DEFAULT_MAX_EVAL
            print(f"[note] ensemble is slow: --max-eval defaults to {args.max_eval}")

    from skolfie.eval.report import (
        atomic_path,
        load_corpus,
        load_splits,
        load_splits_manifest,
        run_evaluation,
        write_report,
    )
    from skolfie.models.baselines import make_scorer

    try:
        oversample = parse_oversample(args.oversample)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    keep_sources = parse_keep_sources(args.keep_sources)
    if args.bino_observer and args.model not in FEATURE_MODELS:
        parser.error("--bino-observer is for the feature models only "
                     f"({', '.join(FEATURE_MODELS)}); the transformer keeps its own features")
    if args.bino_performer and not args.bino_observer:
        parser.error("--bino-performer needs --bino-observer")
    out_dir = args.out_dir or default_out_dir(args.model)
    extra_train = None
    if args.extra_train:
        extra_train = load_extra_train(args.extra_train, parser)
        print(f"[data] {len(extra_train):,} extra train rows from "
              f"{len(args.extra_train)} file(s)")
    corpus = load_corpus(args.corpus)
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)
    print(f"[data] {len(corpus):,} corpus rows; splits: "
          + ", ".join(f"{k}={v:,}" for k, v in splits["split"].value_counts().sort_index().items()))

    scorer = make_scorer(args.model, seed=args.seed, cache_dir=args.cache_dir,
                         lm_model=args.lm_model, lm_max_tokens=args.lm_max_tokens,
                         lm_batch_size=args.lm_batch_size, lm_device=args.lm_device,
                         n_jobs=args.n_jobs, model_name=args.hf_model,
                         max_length=args.max_length, epochs=args.epochs,
                         batch_size=args.batch_size, bino_observer=args.bino_observer,
                         bino_performer=args.bino_performer,
                         bino_max_tokens=args.bino_max_tokens,
                         bino_batch_size=args.bino_batch_size,
                         bino_quantization=args.bino_quantization)
    try:
        report, predictions = run_evaluation(
            scorer, corpus, splits, model_spec=args.model, corpus_path=args.corpus,
            splits_path=args.splits, splits_manifest=manifest, max_train=args.max_train,
            max_eval=args.max_eval, seed=args.seed, extra_train=extra_train,
            oversample=oversample or None, keep_sources=keep_sources or None,
        )
    except ValueError as exc:
        if keep_sources and "cannot keep source" in str(exc):
            parser.error(str(exc))
        raise
    if not report["results"]:
        print("error: no evaluation splits found in the splits file", file=sys.stderr)
        return 2
    paths = write_report(report, out_dir, predictions if args.save_predictions else None)
    if args.save_model:
        import joblib

        with atomic_path(args.save_model) as tmp:
            joblib.dump(scorer, tmp)
        paths["model"] = args.save_model
    print()
    print(paths["md"].read_text(encoding="utf-8"), end="")
    print(f"\n[save] {', '.join(str(p) for p in paths.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
