"""
scripts/eval_lofo.py
---------------------
Leave-one-family-out evaluation (`skolfie.eval.lofo`): for each AI model
family, rebuild the splits with that family held out, train a fresh model on
the rest, and measure how much of the unseen family it catches at a 1%
human-FPR cut-off, next to its false positives on PERSUADE student essays.

Examples:

    # Feature model (fast: the feature cache is reused), modern families
    python scripts/eval_lofo.py --corpus data/processed_modern/corpus.parquet \\
        --config configs/splits_modern.json --cache-dir data/cache \\
        --oversample fresh=3 --out-dir reports_modern/lofo

    # Also hold out GPT-3.5 in a run of its own; Binoculars on a capped sample
    python scripts/eval_lofo.py --corpus ... --families gpt-3.5 \\
        --bino-observer Qwen/Qwen2.5-1.5B --bino-performer Qwen/Qwen2.5-1.5B-Instruct \\
        --max-train 40000 --max-eval 5000 --keep-sources fresh,persuade

    # The (slow, GPU) transformer instead
    python scripts/eval_lofo.py --corpus ... --model transformer --epochs 1

Writes `lofo.json` and `lofo.md` (one row per family) to --out-dir, plus each
run's report card and predictions in `<out-dir>/<family>/`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
LOFO_MODELS = ("features", "features-nolm", "features-rf", "features-nolm-rf", "transformer",
               "transformer-text")


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _evaluate_script() -> Any:
    """scripts/evaluate.py as a module (for its option parsers)."""
    name = "skolfie_scripts_evaluate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out += [s for s in (v.strip() for v in value.split(",")) if s and s not in out]
    return out


def build_parser() -> argparse.ArgumentParser:
    ev = _evaluate_script()
    parser = argparse.ArgumentParser(
        prog="eval_lofo.py",
        description="Leave-one-family-out: hold out each AI model family in turn, retrain, and "
                    "report recall at a 1% human-FPR cut-off and PERSUADE false positives.")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus Parquet file.")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "splits_modern.json",
                        help="Base split config (default: configs/splits_modern.json).")
    parser.add_argument("--families", action="append", default=[], metavar="FAMILIES",
                        help="More families to hold out, one run each, on top of the defaults "
                             "(claude, claude-haiku, gemini), e.g. gpt-3.5,bloom. "
                             "Comma-separated or repeatable.")
    parser.add_argument("--no-default-families", action="store_true",
                        help="Run only the --families given.")
    parser.add_argument("--model", choices=LOFO_MODELS, default="features",
                        help="Scorer retrained per family (default: features; transformer is "
                             "slow and needs a GPU).")
    parser.add_argument("--out-dir", type=Path, default=Path("reports_modern") / "lofo",
                        help="Output directory (default: reports_modern/lofo).")
    parser.add_argument("--max-fpr", type=float, default=0.01,
                        help="Human FPR on val that sets the cut-off (default: 0.01).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=ev._positive_int)
    parser.add_argument("--max-eval", type=ev._positive_int,
                        help="Cap on the val texts scored per run (sets the cut-off).")
    parser.add_argument("--oversample", action="append", default=[], metavar="SOURCE=N")
    parser.add_argument("--keep-sources", action="append", default=[], metavar="SOURCES")
    group = parser.add_argument_group("model options (as in evaluate.py)")
    group.add_argument("--cache-dir", type=Path)
    group.add_argument("--lm-model")
    group.add_argument("--lm-batch-size", type=ev._positive_int)
    group.add_argument("--lm-device")
    group.add_argument("--n-jobs", type=int)
    group.add_argument("--bino-observer")
    group.add_argument("--bino-performer")
    group.add_argument("--bino-max-tokens", type=ev._positive_int)
    group.add_argument("--bino-batch-size", type=ev._positive_int)
    group.add_argument("--bino-quantization", choices=("4bit", "8bit"))
    group.add_argument("--backbone", "--hf-model", dest="hf_model")
    group.add_argument("--max-length", type=ev._positive_int)
    group.add_argument("--epochs", type=ev._positive_int)
    group.add_argument("--batch-size", type=ev._positive_int)
    group.add_argument("--grad-accum", type=ev._positive_int)
    group.add_argument("--lr", type=ev._positive_float)
    group.add_argument("--precision", choices=("auto", "fp16", "bf16", "fp32"))
    group.add_argument("--gradient-checkpointing", action="store_true", default=None)
    return parser


def load_lofo_corpus(path: Path) -> Any:
    """`load_corpus` plus the `n_words` column the splits need."""
    import pandas as pd
    import pyarrow.parquet as pq

    from skolfie.eval.report import load_corpus

    corpus = load_corpus(path)
    if "n_words" in pq.read_schema(path).names:
        n_words = pd.read_parquet(path, columns=["id", "n_words"]).set_index("id")["n_words"]
        corpus["n_words"] = n_words.reindex(corpus.index).to_numpy()
    else:
        corpus["n_words"] = corpus["text"].str.split().str.len()
    missing = {"generator", "domain", "source"} - set(corpus.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    return corpus


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()
    ev = _evaluate_script()

    from skolfie.eval import lofo
    from skolfie.eval.report import _json_safe, atomic_path, git_info, write_report
    from skolfie.eval.splits import load_config
    from skolfie.models.baselines import make_scorer

    if not args.corpus.is_file():
        parser.error(f"--corpus file not found: {args.corpus}")
    if not 0 < args.max_fpr < 1:
        parser.error("--max-fpr must be in (0, 1)")
    if args.bino_observer and not args.model.startswith("features"):
        parser.error("--bino-observer is for the feature models only")
    try:
        oversample = ev.parse_oversample(args.oversample)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    keep_sources = ev.parse_keep_sources(args.keep_sources)
    families = [] if args.no_default_families else list(lofo.DEFAULT_FAMILIES)
    families += [f for f in _list(args.families) if f not in families]
    if not families:
        parser.error("no families to hold out (--no-default-families without --families)")

    base = load_config(args.config)
    corpus = load_lofo_corpus(args.corpus)
    if base.near_duplicate_threshold:
        from skolfie.eval.splits import add_near_duplicate_clusters

        print("[lofo] finding near duplicates for the splits ...", flush=True)
        corpus = add_near_duplicate_clusters(corpus, corpus["text"].astype(str).tolist(), base)
    present = lofo.ai_families(corpus, base)
    unknown = [f for f in families if f not in present]
    if unknown:
        parser.error(f"no AI essays of family {', '.join(unknown)} in the corpus "
                     f"(it has: {', '.join(present)})")
    print(f"[lofo] {len(corpus):,} corpus rows; families: {', '.join(families)}")

    options = dict(cache_dir=args.cache_dir, lm_model=args.lm_model,
                   lm_batch_size=args.lm_batch_size, lm_device=args.lm_device,
                   n_jobs=args.n_jobs, bino_observer=args.bino_observer,
                   bino_performer=args.bino_performer, bino_max_tokens=args.bino_max_tokens,
                   bino_batch_size=args.bino_batch_size,
                   bino_quantization=args.bino_quantization, model_name=args.hf_model,
                   max_length=args.max_length, epochs=args.epochs, batch_size=args.batch_size,
                   grad_accum=args.grad_accum, lr=args.lr, precision=args.precision,
                   gradient_checkpointing=args.gradient_checkpointing)
    rows = []
    for family in families:
        row, report, preds = lofo.run_family(
            lambda: make_scorer(args.model, seed=args.seed, **options), corpus, base, family,
            max_fpr=args.max_fpr, oversample=oversample or None, max_train=args.max_train,
            max_eval=args.max_eval, keep_sources=keep_sources or None, seed=args.seed)
        report["lofo"] = row
        write_report(report, args.out_dir / family, preds)
        rows.append(row)

    result = _json_safe({
        "schema": lofo.LOFO_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model, "config": str(args.config),
        "base_held_out_generators": list(base.held_out_generators),
        "max_fpr": args.max_fpr,
        "sampling": {"seed": args.seed, "max_train": args.max_train, "max_eval": args.max_eval,
                     "oversample": oversample, "keep_sources": keep_sources},
        "families": rows, "git": git_info(),
    })
    paths = {"json": args.out_dir / "lofo.json", "md": args.out_dir / "lofo.md"}
    with atomic_path(paths["json"]) as tmp:
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md = lofo.render_lofo_md(result)
    with atomic_path(paths["md"]) as tmp:
        tmp.write_text(md, encoding="utf-8")
    print()
    print(md, end="")
    print(f"\n[save] {paths['json']}, {paths['md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
