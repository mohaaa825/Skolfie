"""
scripts/train.py
-----------------
Standalone training CLI for the skolfie AI-vs-human text detector.

Examples:

    # Train on your own labeled dataset (CSV or Parquet, columns: text, label)
    python scripts/train.py --data data/labeled.csv

    # ...plus every reviewed row from the app's feedback CSV
    python scripts/train.py --data data/labeled.csv --feedback feedback_data.csv

    # Hold out 20% for an honest accuracy / F1 / AUROC estimate
    python scripts/train.py --data data/labeled.parquet --test-size 0.2

    # Quick smoke run on the small synthetic test fixture (NOT real data)
    python scripts/train.py --seed-demo --out /tmp/demo_model.joblib

Labels must be "human" or "ai" (case-insensitive; the app's display labels
"Human-Written" / "AI-Generated" are accepted too). Nothing is trained at
import time; everything happens in `main()`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = Path("models") / "detector_model.joblib"

_LABEL_ALIASES = {
    "human": "human",
    "human-written": "human",
    "human written": "human",
    "ai": "ai",
    "ai-generated": "ai",
    "ai generated": "ai",
}


def _ensure_repo_on_path() -> None:
    """Allow `python scripts/train.py` from a checkout without installing."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def normalize_label(value: object) -> Optional[str]:
    """Map a raw label to "human" / "ai", or None if it is empty/unknown."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    return _LABEL_ALIASES.get(text)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labeled_data(path: str | os.PathLike[str]) -> Tuple[List[str], List[str]]:
    """Load a CSV/Parquet file with `text` and `label` columns."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported data file '{path}'. Use a .csv or .parquet file.")

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")

    texts: List[str] = []
    labels: List[str] = []
    bad = 0
    for text, raw_label in zip(df["text"], df["label"]):
        label = normalize_label(raw_label)
        if label is None:
            bad += 1
            continue
        if not isinstance(text, str) or not text.strip():
            bad += 1
            continue
        texts.append(text)
        labels.append(label)
    if bad:
        print(f"[data] skipped {bad} row(s) with an empty text or unknown label", file=sys.stderr)
    return texts, labels


def load_feedback_data(path: str | os.PathLike[str]) -> Tuple[List[str], List[str]]:
    """
    Pull reviewed rows out of the app's feedback CSV.

    A row counts as reviewed when the user gave an explicit `user_label`
    (a correction or a manually added example), or gave a thumbs-up to the
    model's prediction, in which case `predicted_label` is taken as confirmed.
    Thumbs-down rows without a corrected label are ignored.
    """
    from skolfie.data.feedback import load_feedback

    if not Path(path).exists():
        raise FileNotFoundError(f"Feedback file not found: {path}")

    df = load_feedback(os.fspath(path))
    texts: List[str] = []
    labels: List[str] = []
    for _, row in df.iterrows():
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        label = normalize_label(row.get("user_label"))
        if label is None and str(row.get("thumbs", "")).strip().lower() == "up":
            label = normalize_label(row.get("predicted_label"))
        if label is not None:
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_seed_demo() -> Tuple[List[str], List[str]]:
    """The 25 + 25 synthetic texts from the test fixtures (smoke runs only)."""
    _ensure_repo_on_path()
    from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

    texts = list(HUMAN_TEXTS) + list(AI_TEXTS)
    labels = ["human"] * len(HUMAN_TEXTS) + ["ai"] * len(AI_TEXTS)
    return texts, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, texts: Sequence[str], labels: Sequence[str]) -> dict:
    """Accuracy / F1 (positive class "ai") / AUROC on a held-out split."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    probs = [model.predict(t)["ai_probability"] for t in texts]
    y_true = [1 if lab == "ai" else 0 for lab in labels]
    y_pred = [1 if p >= 0.5 else 0 for p in probs]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float("nan"),
    }
    if len(set(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, probs))
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the skolfie AI-vs-human text detector and save it with joblib.",
    )
    parser.add_argument(
        "--data", type=Path,
        help="CSV or Parquet file with 'text' and 'label' (human/ai) columns.",
    )
    parser.add_argument(
        "--feedback", type=Path,
        help="Also train on reviewed rows from this feedback CSV (written by the app).",
    )
    parser.add_argument(
        "--seed-demo", action="store_true",
        help="Train on the 50 synthetic fixture texts in tests/fixtures/seed_texts.py "
             "(quick smoke run; not a real model).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Where to write the trained model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.0,
        help="Fraction (0-1) held out, stratified, for accuracy/F1/AUROC. "
             "0 (default) trains on everything.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.data is None and args.feedback is None and not args.seed_demo:
        parser.error("nothing to train on: pass --data, --feedback and/or --seed-demo")
    if not 0.0 <= args.test_size < 1.0:
        parser.error("--test-size must be in [0, 1)")

    _ensure_repo_on_path()

    texts: List[str] = []
    labels: List[str] = []
    sources = []
    if args.seed_demo:
        t, lab = load_seed_demo()
        texts += t
        labels += lab
        sources.append(f"seed-demo ({len(t)})")
    if args.data is not None:
        t, lab = load_labeled_data(args.data)
        texts += t
        labels += lab
        sources.append(f"{args.data} ({len(t)})")
    if args.feedback is not None:
        t, lab = load_feedback_data(args.feedback)
        texts += t
        labels += lab
        sources.append(f"{args.feedback} feedback ({len(t)})")

    n_ai = labels.count("ai")
    n_human = labels.count("human")
    print(f"[data] {len(texts)} texts from {', '.join(sources)}: {n_human} human / {n_ai} ai")
    if n_ai == 0 or n_human == 0:
        print("error: need at least one 'human' and one 'ai' example to train.", file=sys.stderr)
        return 2

    from skolfie.models.ensemble import AIDetectorModel

    train_texts, train_labels = texts, labels
    test_texts: List[str] = []
    test_labels: List[str] = []
    if args.test_size > 0:
        from sklearn.model_selection import train_test_split

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size,
            random_state=args.random_state, stratify=labels,
        )
        print(f"[split] train={len(train_texts)} test={len(test_texts)}")

    model = AIDetectorModel()
    stats = model.train(list(train_texts), list(train_labels), random_state=args.random_state)
    print(f"[train] training-set accuracy: {stats.get('train_accuracy', float('nan')):.3f} "
          "(optimistic; use --test-size for a held-out estimate)")

    if test_texts:
        m = evaluate(model, test_texts, test_labels)
        print(f"[test]  accuracy={m['accuracy']:.3f}  f1={m['f1']:.3f}  auroc={m['auroc']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(os.fspath(out))
    print(f"[save]  model written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
