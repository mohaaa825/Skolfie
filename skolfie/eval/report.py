"""
skolfie/eval/report.py
-----------------------
Run one scorer over the held-out splits and write its "report card".

Inputs (the contract with `skolfie/eval/splits.py`):

    corpus.parquet          the corpus, columns as in `skolfie.data.schema.COLUMNS`
    splits.parquet          columns `id`, `split`; one row per (id, split)
                            membership, so an id can sit in several splits
    splits_manifest.json    next to splits.parquet: `splits_version`,
                            `corpus_version` (optional; recorded if present)

`run_evaluation()` fits the scorer on `train` (optionally sampled down),
predicts every evaluation split (`val` plus all test splits, each optionally
sampled down stratified by label), and returns a JSON-able report dict.
`write_report()` writes into an output directory:

    report.json         everything: metrics, breakdowns, versions, timings, git
    report.md           headline table (one row per split) + breakdown tables
    predictions.parquet (optional) id, split, label, p_ai

`compare_reports()` renders several report.json files side by side.

A report may also carry `verdicts` (cut-offs and the share of Human /
Uncertain / AI verdicts per split, added by `scripts/stack.py`); it is
rendered when present.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from skolfie.eval.metrics import BREAKDOWN_BY, binary_metrics, breakdown, to_binary

REPORT_SCHEMA = 1
TRAIN_SPLIT = "train"
# Reported in this order; val is reported only (not used for tuning yet).
EVAL_SPLITS = ("val", "test", "test_ood_generator", "test_ood_domain", "test_short")
HEADLINE = (("auroc", "AUROC"), ("f1", "F1"), ("fpr_at_95tpr", "FPR@95%TPR"), ("ece", "ECE"))
# Higher is better for these; lower for the rest (used to bold the best in comparisons).
HIGHER_IS_BETTER = {"auroc", "f1", "accuracy"}
CORPUS_COLUMNS = ["id", "text", "label", *BREAKDOWN_BY]
MAX_BREAKDOWN_ROWS = 25
MAX_IMPORTANCE_ROWS = 25


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_corpus(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read the columns the harness needs, indexed by id."""
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    missing = {"id", "text", "label"} - available
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    cols = [c for c in CORPUS_COLUMNS if c in available]
    df = pd.read_parquet(path, columns=cols)
    if df["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    df.index = pd.Index(df["id"].to_numpy())
    return df


def load_splits(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["id", "split"])
    return df.drop_duplicates(ignore_index=True)


def load_splits_manifest(splits_path: str | os.PathLike[str]) -> dict[str, Any]:
    """`splits_manifest.json` next to the splits file, or {} if there is none."""
    path = Path(splits_path).with_name("splits_manifest.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_info(repo: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Current commit and whether the work tree has uncommitted changes."""
    cwd = os.fspath(repo or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=cwd, capture_output=True, text=True, timeout=10,
                                check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def stratified_sample(df: pd.DataFrame, max_n: int | None, seed: int) -> pd.DataFrame:
    """
    At most `max_n` rows with the label mix kept (largest-remainder rounding,
    at least one row per label when possible), in a seed-fixed order.
    """
    if max_n is None or len(df) <= max_n:
        return df
    counts = df["label"].value_counts().sort_index()
    quotas = counts / counts.sum() * max_n
    take = np.floor(quotas).astype(int)
    for lab in (quotas - take).sort_values(ascending=False).index[: max_n - int(take.sum())]:
        take[lab] += 1
    if max_n >= len(counts):
        for lab in take.index:
            if take[lab] == 0:
                donor = take.idxmax()
                take[donor] -= 1
                take[lab] += 1
    parts = [df[df["label"] == lab].sample(n=int(k), random_state=seed)
             for lab, k in take.items() if k > 0]
    return pd.concat(parts).sort_index(kind="stable")


def split_frame(corpus: pd.DataFrame, splits: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    """Corpus rows in split `name` (sorted by id) and the number of split ids not in the corpus."""
    ids = splits.loc[splits["split"] == name, "id"]
    present = ids[ids.isin(corpus.index)]
    return corpus.loc[present.sort_values()], int(len(ids) - len(present))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _model_params(scorer: Any) -> dict[str, Any]:
    params = getattr(scorer, "params", None)
    return params() if callable(params) else {}


def _feature_importance(scorer: Any) -> list[dict[str, Any]]:
    """[{feature, importance}] from scorers that expose `feature_importance()`, else []."""
    method = getattr(scorer, "feature_importance", None)
    return list(method()) if callable(method) else []


def run_evaluation(scorer: Any, corpus: pd.DataFrame, splits: pd.DataFrame, *,
                   model_spec: str | None = None,
                   corpus_path: str | os.PathLike[str] | None = None,
                   splits_path: str | os.PathLike[str] | None = None,
                   splits_manifest: dict[str, Any] | None = None,
                   max_train: int | None = None, max_eval: int | None = None,
                   seed: int = 0, eval_splits: Sequence[str] = EVAL_SPLITS,
                   log: Any = print) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit on `train`, score every eval split. Returns (report dict, predictions frame)."""
    started = time.time()
    manifest = splits_manifest or {}
    available = set(splits["split"].unique())

    # --- fit --------------------------------------------------------------- #
    pretrained = bool(getattr(scorer, "pretrained", False))
    train_info: dict[str, Any] = {"pretrained": pretrained}
    if pretrained:
        log(f"[fit] {scorer.name}: pretrained, skipping training")
        train_info.update({"n": 0, "n_human": 0, "n_ai": 0, "n_available": 0,
                           "fit_seconds": 0.0})
    else:
        train_df, missing = split_frame(corpus, splits, TRAIN_SPLIT)
        n_available = len(train_df)
        train_df = stratified_sample(train_df, max_train, seed)
        y = to_binary(train_df["label"].tolist())
        if n_available == 0 or np.unique(y).size < 2:
            raise ValueError("the train split needs both human and ai texts")
        log(f"[fit] {scorer.name} on {len(train_df)} train texts "
            f"({int(y.sum())} ai / {len(y) - int(y.sum())} human)")
        t0 = time.time()
        scorer.fit(train_df["text"].tolist(), train_df["label"].tolist())
        fit_seconds = time.time() - t0
        log(f"[fit] done in {fit_seconds:.1f}s")
        train_info.update({"n": len(train_df), "n_human": len(y) - int(y.sum()),
                           "n_ai": int(y.sum()), "n_available": n_available,
                           "n_missing_from_corpus": missing,
                           "fit_seconds": round(fit_seconds, 3)})
        del train_df

    # --- evaluate ---------------------------------------------------------- #
    results: dict[str, Any] = {}
    pred_frames: list[pd.DataFrame] = []
    for name in eval_splits:
        if name not in available:
            log(f"[eval] {name}: not in the splits file, skipped")
            continue
        df, missing = split_frame(corpus, splits, name)
        n_available = len(df)
        df = stratified_sample(df, max_eval, seed)
        t0 = time.time()
        p = np.asarray(scorer.predict_proba(df["text"].tolist()), dtype=np.float64)
        predict_seconds = time.time() - t0
        if p.shape != (len(df),):
            raise ValueError(f"{scorer.name}.predict_proba returned shape {p.shape} "
                             f"for {len(df)} texts")
        p = np.clip(p, 0.0, 1.0)
        metrics = binary_metrics(df["label"].tolist(), p)
        tables = breakdown(df, p)
        results[name] = {
            "n_available": n_available,
            "n_missing_from_corpus": missing,
            "predict_seconds": round(predict_seconds, 3),
            "metrics": metrics,
            "breakdown": {col: t.to_dict(orient="records") for col, t in tables.items()},
        }
        log(f"[eval] {name}: n={metrics['n']} auroc={_fmt(metrics['auroc'])} "
            f"f1={_fmt(metrics['f1'])} fpr@95={_fmt(metrics['fpr_at_95tpr'])} "
            f"ece={_fmt(metrics['ece'])} ({predict_seconds:.1f}s)")
        pred_frames.append(pd.DataFrame({"id": df["id"].to_numpy(), "split": name,
                                         "label": df["label"].to_numpy(), "p_ai": p}))

    import sklearn

    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"name": scorer.name, "spec": model_spec or scorer.name,
                  "class": type(scorer).__name__, "params": _model_params(scorer),
                  "feature_importance": _feature_importance(scorer)},
        "corpus": {"path": None if corpus_path is None else os.fspath(corpus_path),
                   "corpus_version": manifest.get("corpus_version"), "n_rows": len(corpus)},
        "splits": {"path": None if splits_path is None else os.fspath(splits_path),
                   "splits_version": manifest.get("splits_version"),
                   "corpus_version": manifest.get("corpus_version")},
        "sampling": {"seed": seed, "max_train": max_train, "max_eval": max_eval},
        "train": train_info,
        "results": results,
        "git": git_info(),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "platform": platform.platform()},
        "total_seconds": round(time.time() - started, 3),
    }
    predictions = (pd.concat(pred_frames, ignore_index=True) if pred_frames else
                   pd.DataFrame(columns=["id", "split", "label", "p_ai"]))
    return _json_safe(report), predictions


def _json_safe(obj: Any) -> Any:
    """NaN/inf -> None and numpy scalars -> Python, so report.json is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# Writing / rendering
# --------------------------------------------------------------------------- #
def write_report(report: dict[str, Any], out_dir: str | os.PathLike[str],
                 predictions: pd.DataFrame | None = None) -> dict[str, Path]:
    """Write report.json, report.md and (if given) predictions.parquet into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "report.json", "md": out / "report.md"}
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)
        fh.write("\n")
    paths["md"].write_text(render_report_md(report), encoding="utf-8")
    if predictions is not None:
        paths["predictions"] = out / "predictions.parquet"
        predictions.to_parquet(paths["predictions"], index=False)
    return paths


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.1f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return lines


def render_report_md(report: dict[str, Any]) -> str:
    model = report["model"]
    train = report["train"]
    git = report.get("git") or {}
    commit = (git.get("commit") or "unknown")[:12] + (" (dirty)" if git.get("dirty") else "")
    sampling = report.get("sampling", {})
    if train.get("pretrained"):
        train_line = "pretrained (not trained here)"
    else:
        train_line = (f"{train['n']:,} texts ({train['n_ai']:,} ai / {train['n_human']:,} human) "
                      f"of {train['n_available']:,} in `train`, fit {train['fit_seconds']:.1f}s")
    lines = [
        f"# Report card: {model['name']}",
        "",
        f"- model: `{model['spec']}` ({model['class']})",
        f"- corpus: `{report['corpus'].get('corpus_version') or 'unversioned'}`, "
        f"splits: `{report['splits'].get('splits_version') or 'unversioned'}`",
        f"- train: {train_line}",
        f"- sampling: seed {sampling.get('seed')}, max_train {sampling.get('max_train')}, "
        f"max_eval {sampling.get('max_eval')}",
        f"- git: `{commit}`, created {report.get('created_at')}, "
        f"total {report.get('total_seconds', 0):.1f}s",
        "",
        "## Headline",
        "",
        "Threshold 0.5 for F1; FPR on human text at the threshold reaching 95% AI recall; "
        "ECE with 15 equal-width bins of P(ai).",
        "",
    ]
    rows = []
    notes = []
    for split, res in report["results"].items():
        m = res["metrics"]
        n = f"{m['n']:,} ({m['n_ai']:,} ai / {m['n_human']:,} h)"
        if res.get("n_available", m["n"]) > m["n"]:
            n += f" of {res['n_available']:,}"
        rows.append([split, n] + [_fmt(m.get(key)) for key, _ in HEADLINE]
                    + [f"{res.get('predict_seconds', 0):.1f}s"])
        notes += [f"- {split}: {note}" for note in m.get("notes", [])]
    lines += _table(["split", "n"] + [label for _, label in HEADLINE] + ["predict"], rows)
    if notes:
        lines += ["", "Notes:", ""] + notes

    verdict_info = report.get("verdicts") or {}
    if verdict_info.get("splits"):
        cut = verdict_info["cutoffs"]
        lines += ["", "## Verdicts", "",
                  f"AI if P(ai) > {cut['ai_threshold']:.4f} (set so that at most "
                  f"{cut['max_fpr']:.0%} of human texts in the fit split score above it); "
                  f"Human if P(ai) < {cut['human_threshold']:.4f} (at most "
                  f"{cut['max_miss']:.0%} of AI texts below it); Uncertain in between.", ""]
        rows = []
        for split, t in verdict_info["splits"].items():
            h, a = t["human_texts"], t["ai_texts"]
            rows.append([split, _pct(h["ai"]), _pct(h["uncertain"]), _pct(h["human"]),
                         _pct(a["ai"]), _pct(a["uncertain"]), _pct(a["human"])])
        lines += _table(["split", "human → AI", "human → uncertain", "human → human",
                         "AI → AI", "AI → uncertain", "AI → human"], rows)

    importance = model.get("feature_importance") or []
    if importance:
        shown = importance[:MAX_IMPORTANCE_ROWS]
        lines += ["", f"## Feature importance (top {len(shown)} of {len(importance)})", "",
                  "Share of the model's total importance (sums to 1 over all features).", ""]
        lines += _table(["feature", "importance"],
                        [[r["feature"], _fmt(r["importance"])] for r in shown])

    for split, res in report["results"].items():
        lines += ["", f"## Breakdown: {split}"]
        for col, records in res.get("breakdown", {}).items():
            lines += ["", f"### by {col}", ""]
            shown = records[:MAX_BREAKDOWN_ROWS]
            lines += _table(
                [col, "n", "ai", "human", "acc", "recall (ai)", "FPR (human)", "AUROC"],
                [[r["group"], f"{r['n']:,}", f"{r['n_ai']:,}", f"{r['n_human']:,}",
                  _fmt(r["accuracy"]), _fmt(r["recall_ai"]), _fmt(r["fpr_human"]),
                  _fmt(r["auroc"])] for r in shown])
            if len(records) > len(shown):
                lines.append(f"\n... {len(records) - len(shown)} smaller groups in report.json")
    return "\n".join(lines) + "\n"


def compare_reports(paths: Sequence[str | os.PathLike[str]]) -> str:
    """Markdown table of the headline metrics per split for several report.json files."""
    reports = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            reports.append((Path(path), json.load(fh)))
    if not reports:
        return "No reports to compare.\n"
    names = [r["model"]["name"] for _, r in reports]
    labels = [f"{name} ({path.parent.name})" if names.count(name) > 1 else name
              for name, (path, _) in zip(names, reports)]

    splits: list[str] = []
    for _, r in reports:
        splits += [s for s in r["results"] if s not in splits]
    order = {s: i for i, s in enumerate(EVAL_SPLITS)}
    splits.sort(key=lambda s: order.get(s, len(order)))

    rows = []
    for split in splits:
        present = [(lab, r["results"][split]["metrics"]) for lab, (_, r) in zip(labels, reports)
                   if split in r["results"]]
        best = {}
        for key, _ in HEADLINE:
            vals = [m.get(key) for _, m in present if m.get(key) is not None]
            if vals:
                best[key] = max(vals) if key in HIGHER_IS_BETTER else min(vals)
        for lab, m in present:
            cells = []
            for key, _ in HEADLINE:
                cell = _fmt(m.get(key))
                if len(present) > 1 and m.get(key) is not None and m.get(key) == best.get(key):
                    cell = f"**{cell}**"
                cells.append(cell)
            rows.append([split, lab, f"{m['n']:,}"] + cells)
    lines = ["# Model comparison", "",
             "Best value per split in bold (higher AUROC/F1, lower FPR@95%TPR/ECE).", ""]
    lines += _table(["split", "model", "n"] + [label for _, label in HEADLINE], rows)
    meta = [f"- {lab}: corpus `{r['corpus'].get('corpus_version') or 'unversioned'}`, "
            f"splits `{r['splits'].get('splits_version') or 'unversioned'}`, "
            f"train n={r['train'].get('n', 0):,}, commit "
            f"`{((r.get('git') or {}).get('commit') or 'unknown')[:12]}`"
            for lab, (_, r) in zip(labels, reports)]
    versions = {(r["corpus"].get("corpus_version"), r["splits"].get("splits_version"))
                for _, r in reports}
    lines += ["", "Reports:", ""] + meta
    if len(versions) > 1:
        lines += ["", "**Warning:** these reports use different corpus/splits versions; "
                      "their numbers are not directly comparable."]
    uneven = [split for split in splits
              if len({r["results"][split]["metrics"]["n"] for _, r in reports
                      if split in r["results"]}) > 1]
    if uneven:
        lines += ["", f"**Note:** different sample sizes (--max-eval) on: {', '.join(uneven)}."]
    return "\n".join(lines) + "\n"
