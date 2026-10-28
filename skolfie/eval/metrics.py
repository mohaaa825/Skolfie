"""
skolfie/eval/metrics.py
------------------------
Pure metric functions for a binary AI-vs-human detector.

Conventions used everywhere in `skolfie.eval`:

    y_true  1 = AI-generated text (the positive class), 0 = human text
    p_ai    the model's probability that the text is AI-generated, in [0, 1]

`binary_metrics()` returns one flat dict per evaluation set:

    n, n_human, n_ai
    accuracy, f1        at the fixed threshold 0.5 (predict AI iff p_ai >= 0.5)
    auroc               sklearn's roc_auc_score; NaN (+ a note) if one class only
    fpr_at_95tpr        see `fpr_at_tpr()`: the share of human texts flagged as AI
    threshold_at_95tpr  at the strictest threshold that still catches >= 95% of AI
    tpr_at_threshold    the AI recall actually reached at that threshold (>= 0.95)
    ece                 see `expected_calibration_error()`: 15 equal-width bins
    brier               mean((p_ai - y_true)^2)
    notes               human-readable caveats (e.g. why a value is NaN)

`breakdown()` gives the same view per generator / domain / source group.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.5
TARGET_TPR = 0.95
ECE_BINS = 15
BREAKDOWN_BY = ("generator", "domain", "source")


def to_binary(labels: Iterable[Any]) -> np.ndarray:
    """Map labels ("ai"/"human", case-insensitive, or 1/0, or bools) to 1 = ai, 0 = human."""
    out = []
    for lab in labels:
        if isinstance(lab, str):
            low = lab.strip().lower()
            if low in ("ai", "ai-generated", "1"):
                out.append(1)
            elif low in ("human", "human-written", "0"):
                out.append(0)
            else:
                raise ValueError(f"unknown label {lab!r}; expected 'human' or 'ai'")
        else:
            try:
                val = int(lab)
            except (TypeError, ValueError):
                val = -1
            if val not in (0, 1) or val != lab:  # 0.7 must not silently become 0
                raise ValueError(f"unknown label {lab!r}; expected 0 (human) or 1 (ai)")
            out.append(val)
    return np.asarray(out, dtype=np.int64)


def _as_arrays(y_true: Sequence[Any], p_ai: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(y_true) and isinstance(next(iter(y_true)), str):
        y = to_binary(y_true)
    else:
        raw = np.asarray(y_true)
        # Floats such as 0.7 or NaN must not be truncated into a valid label.
        if raw.dtype.kind not in "biu" and not np.isin(raw.astype(np.float64), (0, 1)).all():
            raise ValueError("y_true must contain only 0 (human) and 1 (ai)")
        y = raw.astype(np.int64)
    p = np.asarray(p_ai, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y_true and p_ai must be 1-D and equal length, got {y.shape} / {p.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 (human) and 1 (ai)")
    if p.size and (np.isnan(p).any() or p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("p_ai must be finite probabilities in [0, 1]")
    return y, p


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #
def fpr_at_tpr(y_true: Sequence[Any], p_ai: Sequence[float],
               target_tpr: float = TARGET_TPR) -> dict[str, float]:
    """
    False-positive rate on human text at the threshold that keeps AI recall >= target_tpr.

    Decision rule: predict AI iff p_ai >= t. The threshold t is chosen on the
    given data as the *largest* t with TPR(t) = mean(p_ai[ai] >= t) >= target_tpr,
    which is the k-th highest AI score, k = ceil(target_tpr * n_ai). Taking the
    largest such t gives the lowest FPR among all thresholds that meet the
    recall target. Ties at t all count as AI, so the TPR reached can be a bit
    above the target. Then FPR = mean(p_ai[human] >= t).

    Returns {"fpr", "threshold", "tpr"}; fpr is NaN without human texts, all
    three are NaN without AI texts.
    """
    y, p = _as_arrays(y_true, p_ai)
    ai_scores = np.sort(p[y == 1])[::-1]
    human_scores = p[y == 0]
    if ai_scores.size == 0:
        return {"fpr": math.nan, "threshold": math.nan, "tpr": math.nan}
    # Small epsilon so that e.g. 0.95 * 20 = 18.999999... still gives k = 19.
    k = max(1, math.ceil(target_tpr * ai_scores.size - 1e-9))
    threshold = float(ai_scores[k - 1])
    tpr = float(np.mean(ai_scores >= threshold))
    fpr = float(np.mean(human_scores >= threshold)) if human_scores.size else math.nan
    return {"fpr": fpr, "threshold": threshold, "tpr": tpr}


def expected_calibration_error(y_true: Sequence[Any], p_ai: Sequence[float],
                               n_bins: int = ECE_BINS) -> float:
    """
    ECE of P(ai) with `n_bins` equal-width bins over [0, 1].

    Bin b holds the texts with b/n_bins <= p_ai < (b+1)/n_bins (the last bin
    also includes p_ai = 1). With n texts in total:

        ECE = sum over non-empty bins b of  |B_b| / n * | mean(y_true in B_b) - mean(p_ai in B_b) |

    i.e. the weighted gap between the predicted AI probability and the
    observed AI fraction (calibration of the positive class, not of the
    top-label confidence). NaN for an empty input.
    """
    y, p = _as_arrays(y_true, p_ai)
    if y.size == 0:
        return math.nan
    bins = np.clip(np.floor(p * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    nonempty = counts > 0
    gaps = np.abs(sum_y[nonempty] - sum_p[nonempty])  # = count * |mean_y - mean_p|
    return float(gaps.sum() / y.size)


def brier_score(y_true: Sequence[Any], p_ai: Sequence[float]) -> float:
    """Mean squared error between p_ai and the 0/1 label. NaN for an empty input."""
    y, p = _as_arrays(y_true, p_ai)
    if y.size == 0:
        return math.nan
    return float(np.mean((p - y) ** 2))


def auroc(y_true: Sequence[Any], p_ai: Sequence[float]) -> float:
    """sklearn ROC AUC, or NaN when only one class is present."""
    from sklearn.metrics import roc_auc_score

    y, p = _as_arrays(y_true, p_ai)
    if np.unique(y).size < 2:
        return math.nan
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------- #
# All headline metrics at once
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: Sequence[Any], p_ai: Sequence[float],
                   threshold: float = DEFAULT_THRESHOLD,
                   target_tpr: float = TARGET_TPR,
                   n_bins: int = ECE_BINS) -> dict[str, Any]:
    """Every headline metric for one evaluation set (see the module docstring)."""
    y, p = _as_arrays(y_true, p_ai)
    n = int(y.size)
    n_ai = int(y.sum())
    n_human = n - n_ai
    notes: list[str] = []

    if n == 0:
        notes.append("empty evaluation set")
        nan = math.nan
        return {"n": 0, "n_human": 0, "n_ai": 0, "accuracy": nan, "f1": nan, "auroc": nan,
                "fpr_at_95tpr": nan, "threshold_at_95tpr": nan, "tpr_at_threshold": nan,
                "ece": nan, "brier": nan, "notes": notes}

    pred = (p >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    accuracy = float(np.mean(pred == y))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    roc = auroc(y, p)
    if math.isnan(roc):
        only = "AI" if n_ai else "human"
        notes.append(f"AUROC undefined: only {only} texts present")

    at = fpr_at_tpr(y, p, target_tpr)
    if n_ai == 0:
        notes.append("FPR@95%TPR undefined: no AI texts to set the threshold")
    elif n_human == 0:
        notes.append("FPR@95%TPR undefined: no human texts")

    return {
        "n": n,
        "n_human": n_human,
        "n_ai": n_ai,
        "accuracy": accuracy,
        "f1": float(f1),
        "auroc": roc,
        "fpr_at_95tpr": at["fpr"],
        "threshold_at_95tpr": at["threshold"],
        "tpr_at_threshold": at["tpr"],
        "ece": expected_calibration_error(y, p, n_bins),
        "brier": brier_score(y, p),
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Per-group breakdown
# --------------------------------------------------------------------------- #
BREAKDOWN_COLUMNS = ["group", "n", "n_human", "n_ai", "accuracy", "recall_ai", "fpr_human",
                     "auroc"]


def breakdown(df: pd.DataFrame, p_ai: Sequence[float],
              by: Sequence[str] = BREAKDOWN_BY,
              threshold: float = DEFAULT_THRESHOLD) -> dict[str, pd.DataFrame]:
    """
    Per-group metrics at a fixed threshold, one table per column in `by`.

    `df` needs a `label` column ("human"/"ai" or 1/0) plus the `by` columns
    (missing ones are skipped); `p_ai` is aligned with df's rows. Each table
    has `BREAKDOWN_COLUMNS`, one row per group value, largest group first:

        accuracy   share of the group classified correctly at `threshold`
        recall_ai  share of the group's AI texts flagged as AI (NaN if none)
        fpr_human  share of the group's human texts flagged as AI (NaN if none)
        auroc      only when the group has both classes, else NaN

    So an all-AI generator group shows accuracy + recall, the human group
    shows accuracy + FPR, and mixed domain/source groups show everything.
    """
    p = np.asarray(p_ai, dtype=np.float64)
    if len(df) != p.size:
        raise ValueError(f"df has {len(df)} rows but p_ai has {p.size} values")
    y = to_binary(df["label"].tolist())
    pred = (p >= threshold).astype(np.int64)

    tables: dict[str, pd.DataFrame] = {}
    for col in by:
        if col not in df.columns:
            continue
        keys = df[col].astype(str).to_numpy()
        rows = []
        for value in pd.unique(keys):
            mask = keys == value
            yg, pg, predg = y[mask], p[mask], pred[mask]
            n_ai = int(yg.sum())
            n_human = int(mask.sum()) - n_ai
            rows.append({
                "group": str(value),
                "n": int(mask.sum()),
                "n_human": n_human,
                "n_ai": n_ai,
                "accuracy": float(np.mean(predg == yg)),
                "recall_ai": float(np.mean(predg[yg == 1])) if n_ai else math.nan,
                "fpr_human": float(np.mean(predg[yg == 0])) if n_human else math.nan,
                "auroc": auroc(yg, pg) if n_ai and n_human else math.nan,
            })
        table = pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)
        tables[col] = table.sort_values(["n", "group"], ascending=[False, True],
                                        ignore_index=True)
    return tables
