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
Every output file is written atomically (`atomic_path`: temporary file,
then rename), so an interrupted run never leaves a truncated report or
predictions file for `--compare` or `scripts/stack.py` to read.

Oversampling (`oversample={"fresh": 6}`, `evaluate.py --oversample fresh=6`):
the train rows of a source are repeated that many times (whole copies, after
`max_train` sampling), so a few hundred texts from current models are not
drowned out by 176k older ones. It works for every scorer, since it only
changes the (texts, labels) passed to `fit()`, and a transformer-chunks model
cuts a different random chunk from each copy. Copies are of training texts
only, so no evaluation split is affected; but a scorer's *own* internal
hold-outs (LightGBM's early-stopping rows, the transformer's eval rows, the
Delta out-of-fold folds) can now hold a copy of a text that is also trained
on, which makes those internal numbers, and the number of trees or the
checkpoint they pick, slightly optimistic. The stacker (`scripts/stack.py`)
is not oversampled: it calibrates on `val` as it is.

Kept sources (`keep_sources=["fresh", "persuade"]`, `evaluate.py
--keep-sources fresh,persuade`): when `max_train` / `max_eval` sample a
split down, the rows of these sources are always kept and only the other
rows are sampled (stratified by label) to fill the rest, so a slow scorer
(e.g. the feature model with Binoculars features) trained on 40k of 176k
texts never loses the few modern student and AI essays. If the kept rows
alone exceed the limit, all of them are kept and nothing else. The
report's `sampling` block records the sources and how many rows each
split kept.

A report may also carry `verdicts` (cut-offs and the share of Human /
Uncertain / AI verdicts per split, added by `scripts/stack.py`); it is
rendered when present.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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


def sample_keeping(df: pd.DataFrame, max_n: int | None, seed: int,
                   keep: Sequence[str] | None = None,
                   column: str = "source") -> tuple[pd.DataFrame, dict[str, int]]:
    """`stratified_sample`, except that rows whose `column` is in `keep` are never dropped.

    Returns (sample, {kept value: rows kept}); the dict is empty when nothing
    was sampled. The other rows fill the remaining `max_n - n_kept` places,
    stratified by label (none if the kept rows already reach `max_n`).
    """
    if not keep or max_n is None or len(df) <= max_n:
        return stratified_sample(df, max_n, seed), {}
    if column not in df.columns:
        raise ValueError(f"cannot keep sources: the corpus has no {column!r} column")
    values = df[column].astype(str)
    is_kept = values.isin(set(keep)).to_numpy()
    kept, rest = df[is_kept], df[~is_kept]
    room = max_n - len(kept)
    rest = stratified_sample(rest, room, seed) if room > 0 else rest.iloc[:0]
    counts = values[is_kept].value_counts()
    info = {str(v): int(counts.get(v, 0)) for v in sorted(set(keep))}
    return pd.concat([kept, rest]).sort_index(kind="stable"), info


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


def _text_key(text: Any) -> bytes:
    """The corpus's own duplicate key: case-folded, punctuation and whitespace removed."""
    from skolfie.data.corpus import normalized_hash

    return normalized_hash(text if isinstance(text, str) else "")


def oversample_rows(df: pd.DataFrame, factors: dict[str, int] | None,
                    column: str = "source") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repeat the rows whose `column` value is in `factors` that many times in all.

    Returns (frame with the extra copies appended, info). A factor must be an
    integer >= 1 (1 changes nothing); a value with no rows in `df` is an
    error, since a typo would otherwise silently train the old model.
    """
    if not factors:
        return df, {}
    if column not in df.columns:
        raise ValueError(f"cannot oversample: the corpus has no {column!r} column")
    present = df[column].astype(str)
    info: dict[str, Any] = {"column": column, "factors": {}, "rows": {}, "n_added": 0}
    parts = [df]
    for value, factor in sorted(factors.items()):
        if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
            raise ValueError(f"oversample factor for {value!r} must be an integer >= 1, "
                             f"got {factor!r}")
        rows = df[present == value]
        if rows.empty:
            seen = ", ".join(sorted(present.unique())) or "(none)"
            raise ValueError(f"cannot oversample {column} {value!r}: no train rows have it "
                             f"(train has: {seen})")
        info["factors"][value] = int(factor)
        info["rows"][value] = int(len(rows))
        if factor > 1:
            parts += [rows] * (int(factor) - 1)
            info["n_added"] += int(len(rows)) * (int(factor) - 1)
    out = pd.concat(parts) if len(parts) > 1 else df
    return out, info


def _filter_extra_train(extra: pd.DataFrame, corpus: pd.DataFrame, splits: pd.DataFrame,
                        log: Any) -> tuple[pd.DataFrame, dict]:
    """Drop extra train rows that overlap any non-train split; summarise the rest.

    A row is dropped when its text is a duplicate of an evaluation text in the
    corpus's sense (`normalized_hash`: case, punctuation and whitespace are
    ignored) or its `orig_id` is an evaluation id. Every split except `train`
    counts, not only the ones reported here, so a saved model never trained
    on a text that a later evaluation might use.
    """
    missing = {"text", "label"} - set(extra.columns)
    if missing:
        raise ValueError(f"extra train data is missing column(s): {', '.join(sorted(missing))}")
    eval_ids = splits.loc[splits["split"] != TRAIN_SPLIT, "id"]
    eval_ids = eval_ids[eval_ids.isin(corpus.index)]
    eval_keys = {_text_key(t) for t in corpus.loc[eval_ids.unique(), "text"]}
    overlap = np.array([_text_key(t) in eval_keys for t in extra["text"]], dtype=bool)
    if "orig_id" in extra:
        overlap |= extra["orig_id"].astype(str).isin(set(eval_ids.astype(str))).to_numpy()
    else:
        log("[fit] warning: extra train data has no orig_id column; only texts that "
            "duplicate an evaluation text can be dropped, not edited copies of one")
    kept = extra[~overlap]
    y = to_binary(kept["label"].tolist())
    info: dict[str, Any] = {"n_given": len(extra), "n_dropped_eval_overlap": int(overlap.sum()),
                            "n": len(kept), "n_ai": int(y.sum()), "n_human": len(y) - int(y.sum())}
    if "extra_source" in extra:
        info["files"] = sorted(map(str, pd.unique(extra["extra_source"])))
    for col, key in (("extra_source", "by_file"), ("attack", "by_attack")):
        if col in kept:
            info[key] = {str(k): int(v)
                         for k, v in kept[col].value_counts().sort_index().items()}
    if overlap.any():
        log(f"[fit] dropped {int(overlap.sum())} extra train texts that are (or come from) "
            "evaluation-split texts")
    log(f"[fit] + {len(kept)} extra train texts ({info['n_ai']} ai / {info['n_human']} human)")
    return kept, info


def run_evaluation(scorer: Any, corpus: pd.DataFrame, splits: pd.DataFrame, *,
                   model_spec: str | None = None,
                   corpus_path: str | os.PathLike[str] | None = None,
                   splits_path: str | os.PathLike[str] | None = None,
                   splits_manifest: dict[str, Any] | None = None,
                   max_train: int | None = None, max_eval: int | None = None,
                   seed: int = 0, eval_splits: Sequence[str] = EVAL_SPLITS,
                   extra_train: pd.DataFrame | None = None,
                   oversample: dict[str, int] | None = None,
                   keep_sources: Sequence[str] | None = None,
                   log: Any = print) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit on `train`, score every eval split. Returns (report dict, predictions frame).

    `extra_train` (columns text, label; optional orig_id, attack, extra_source)
    is appended to the train texts after sampling, e.g. attacked copies for
    adversarial training. Rows whose text, or `orig_id`, is in any non-train
    split are dropped first (see `_filter_extra_train`).

    `oversample` ({source: factor}) repeats the sampled train rows of those
    sources (see `oversample_rows` and the module docstring).

    `keep_sources` (corpus `source` values) are never sampled away by
    `max_train` / `max_eval` (see `sample_keeping`).
    """
    started = time.time()
    manifest = splits_manifest or {}
    available = set(splits["split"].unique())
    keep = sorted(set(keep_sources or ()))
    kept_rows: dict[str, dict[str, int]] = {}
    if keep:
        if "source" not in corpus.columns:
            raise ValueError("cannot keep sources: the corpus has no 'source' column")
        present = set(corpus["source"].astype(str).unique())
        unknown = [k for k in keep if k not in present]
        if unknown:  # a typo would otherwise silently sample the modern essays away
            raise ValueError(f"cannot keep source(s) {', '.join(unknown)}: not in the corpus "
                             f"(it has: {', '.join(sorted(present)) or '(none)'})")

    # --- fit --------------------------------------------------------------- #
    pretrained = bool(getattr(scorer, "pretrained", False))
    train_info: dict[str, Any] = {"pretrained": pretrained}
    if pretrained:
        log(f"[fit] {scorer.name}: pretrained, skipping training"
            + (" (extra train texts ignored)" if extra_train is not None else ""))
        train_info.update({"n": 0, "n_human": 0, "n_ai": 0, "n_available": 0,
                           "fit_seconds": 0.0})
    else:
        train_df, missing = split_frame(corpus, splits, TRAIN_SPLIT)
        n_available = len(train_df)
        train_df, kept = sample_keeping(train_df, max_train, seed, keep)
        if kept:
            kept_rows[TRAIN_SPLIT] = kept
            log("[fit] kept every train row of "
                + ", ".join(f"{k} ({n})" for k, n in kept.items()))
        y = to_binary(train_df["label"].tolist())
        if n_available == 0 or np.unique(y).size < 2:
            raise ValueError("the train split needs both human and ai texts")
        log(f"[fit] {scorer.name} on {len(train_df)} train texts "
            f"({int(y.sum())} ai / {len(y) - int(y.sum())} human)")
        fit_df, over_info = oversample_rows(train_df, oversample)
        if over_info:
            train_info["oversample"] = over_info
            log(f"[fit] oversampling: + {over_info['n_added']} copies ("
                + ", ".join(f"{k} {n} rows x{over_info['factors'][k]}"
                            for k, n in over_info["rows"].items()) + ")")
        texts, labels = fit_df["text"].tolist(), fit_df["label"].tolist()
        del fit_df
        if extra_train is not None:
            extra, train_info["extra_train"] = _filter_extra_train(
                extra_train, corpus, splits, log)
            texts += extra["text"].tolist()
            labels += extra["label"].tolist()
        t0 = time.time()
        scorer.fit(texts, labels)
        fit_seconds = time.time() - t0
        log(f"[fit] done in {fit_seconds:.1f}s")
        train_info.update({"n": len(train_df), "n_human": len(y) - int(y.sum()),
                           "n_ai": int(y.sum()), "n_available": n_available,
                           "n_missing_from_corpus": missing,
                           "fit_seconds": round(fit_seconds, 3)})
        del train_df, texts, labels

    # --- evaluate ---------------------------------------------------------- #
    results: dict[str, Any] = {}
    pred_frames: list[pd.DataFrame] = []
    for name in eval_splits:
        if name not in available:
            log(f"[eval] {name}: not in the splits file, skipped")
            continue
        df, missing = split_frame(corpus, splits, name)
        n_available = len(df)
        df, kept = sample_keeping(df, max_eval, seed, keep)
        if kept:
            kept_rows[name] = kept
        t0 = time.time()
        p = np.asarray(scorer.predict_proba(df["text"].tolist()), dtype=np.float64)
        predict_seconds = time.time() - t0
        if p.shape != (len(df),):
            raise ValueError(f"{scorer.name}.predict_proba returned shape {p.shape} "
                             f"for {len(df)} texts")
        if np.isnan(p).any():
            raise ValueError(f"{scorer.name}.predict_proba returned NaN for "
                             f"{int(np.isnan(p).sum())} of {len(p)} texts in {name}")
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
        "sampling": {"seed": seed, "max_train": max_train, "max_eval": max_eval,
                     **({"oversample": dict(oversample)} if oversample else {}),
                     **({"keep_sources": keep, "kept_rows": kept_rows} if keep else {})},
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
@contextlib.contextmanager
def atomic_path(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a temporary path next to `path`; it replaces `path` only if the block succeeds.

    The temporary name is unique per call and ends with the target's own
    name, so writers that pick a format from the extension (joblib
    compression) behave the same.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{secrets.token_hex(6)}-{path.name}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_report(report: dict[str, Any], out_dir: str | os.PathLike[str],
                 predictions: pd.DataFrame | None = None) -> dict[str, Path]:
    """Write report.json, report.md and (if given) predictions.parquet into out_dir."""
    out = Path(out_dir)
    paths = {"json": out / "report.json", "md": out / "report.md"}
    if predictions is not None:
        paths["predictions"] = out / "predictions.parquet"
        with atomic_path(paths["predictions"]) as tmp:
            predictions.to_parquet(tmp, index=False)
    text = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    with atomic_path(paths["json"]) as tmp:
        tmp.write_text(text, encoding="utf-8")
    with atomic_path(paths["md"]) as tmp:
        tmp.write_text(render_report_md(report), encoding="utf-8")
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


def _cell(value: Any) -> str:
    """A table cell: no line breaks, `|` escaped, and no raw HTML (group names come from data)."""
    text = " ".join(str(value).split())
    return text.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
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
        extra = train.get("extra_train")
        if extra:
            train_line += (f"; plus {extra['n']:,} extra texts ({extra['n_ai']:,} ai / "
                           f"{extra['n_human']:,} human, e.g. attacked copies) from --extra-train")
        over = train.get("oversample")
        if over:
            train_line += ("; oversampled by " + over.get("column", "source") + ": "
                           + ", ".join(f"{k} {n:,} x{over['factors'][k]}"
                                       for k, n in over["rows"].items())
                           + f" (+{over['n_added']:,} copies)")
    lines = [
        f"# Report card: {model['name']}",
        "",
        f"- model: `{model['spec']}` ({model['class']})",
        f"- corpus: `{report['corpus'].get('corpus_version') or 'unversioned'}`, "
        f"splits: `{report['splits'].get('splits_version') or 'unversioned'}`",
        f"- train: {train_line}",
        f"- sampling: seed {sampling.get('seed')}, max_train {sampling.get('max_train')}, "
        f"max_eval {sampling.get('max_eval')}"
        + (f", always kept: {', '.join(sampling['keep_sources'])}"
           if sampling.get("keep_sources") else ""),
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
            data = json.load(fh)
        if not (isinstance(data, dict) and isinstance(data.get("model"), dict)
                and isinstance(data.get("results"), dict)):
            raise ValueError(f"{path} is not a report.json written by evaluate.py or stack.py")
        for key in ("corpus", "splits", "train"):
            data[key] = data.get(key) or {}
        reports.append((Path(path), data))
    if not reports:
        return "No reports to compare.\n"
    names = [str(r["model"].get("name") or path.parent.name) for path, r in reports]
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
