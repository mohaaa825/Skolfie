"""
skolfie/explain/contributions.py
---------------------------------
Per-text SHAP values for the feature model (Model B,
`skolfie.models.feature_model.FeatureModel`).

LightGBM computes exact TreeSHAP itself (`Booster.predict(X,
pred_contrib=True)`), so the `shap` package is not needed. The values are in
log-odds: for every text, the bias (`base_value`, the model's average
log-odds) plus the sum of its feature contributions is the model's raw score,
and P(ai) = sigmoid(raw score). A positive contribution pushes the text
towards AI, a negative one towards human. The random-forest variant of the
feature model is not supported (it is only a comparison model).

`explain_features()` keeps the top-k features per text by absolute
contribution, each with its raw value and a plain-language reason from
`skolfie.explain.reasons`.

`feature_reference()` gives each feature's median on reference texts (all,
human only, AI only). With it, a reason says whether the text's value is
low or high: compared with the midpoint of the human and the AI median (so
the answer does not depend on the class mix of the reference texts), else
with the overall median. A reviewer can check the value against what human
and AI texts usually have. Use texts the feature model did not train on
(e.g. `val`): its Delta features are fitted on the training split, so
training texts score closer to their own class than new texts do.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from skolfie.explain.reasons import lookup, reason

Reference = dict[str, dict[str, Optional[float]]]


def _booster(feature_model: Any) -> Any:
    if getattr(feature_model, "clf", None) is None:
        raise RuntimeError("the feature model is not fitted")
    if getattr(feature_model, "classifier", "lgbm") != "lgbm":
        raise NotImplementedError(
            "SHAP explanations need the LightGBM feature model; the random-forest variant "
            f"({getattr(feature_model, 'name', 'rf')}) is not supported")
    return feature_model.clf.booster_


def shap_values(feature_model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (contributions (n, n_features), base values (n,)) in log-odds for the rows of X.

    X is a feature matrix in `feature_model.feature_names_` order, as
    `feature_model.featurize(texts)` returns it.
    """
    booster = _booster(feature_model)
    X = np.asarray(X, dtype=np.float64)
    n_features = len(feature_model.feature_names_)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"X must have shape (n, {n_features}), got {X.shape}")
    if len(X) == 0:
        return np.zeros((0, n_features)), np.zeros(0)
    # Same trees as predict_proba: the early-stopping iteration when there is one.
    best = getattr(feature_model.clf, "best_iteration_", None) or None
    out = np.asarray(booster.predict(X, pred_contrib=True, num_iteration=best),
                     dtype=np.float64)
    if out.shape != (len(X), n_features + 1):
        raise ValueError(f"unexpected SHAP output shape {out.shape}")
    return out[:, :-1], out[:, -1]


def _num(x: Any) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def feature_reference(feature_model: Any, texts: Sequence[str],
                      labels: Sequence[Any]) -> Reference:
    """{feature: {"median", "human", "ai"}}: medians on reference texts (NaN -> None)."""
    from skolfie.eval.metrics import to_binary

    y = to_binary(labels)
    if len(y) != len(texts):
        raise ValueError("texts and labels must have the same length")
    X = np.asarray(feature_model.featurize(list(texts)), dtype=np.float64)
    out: Reference = {}
    for j, name in enumerate(feature_model.feature_names_):
        col = X[:, j]
        row: dict[str, Optional[float]] = {}
        for key, mask in (("median", np.ones(len(y), dtype=bool)), ("human", y == 0),
                          ("ai", y == 1)):
            vals = col[mask & np.isfinite(col)]
            row[key] = _num(np.median(vals)) if vals.size else None
        out[name] = row
    return out


def _typical(ref: dict[str, Optional[float]]) -> Optional[float]:
    """Where a value turns from "low" to "high" for the reasons (see the module docstring)."""
    human, ai = ref.get("human"), ref.get("ai")
    if human is not None and ai is not None:
        return (human + ai) / 2
    return ref.get("median")


def top_features(names: Sequence[str], values: np.ndarray, contributions: np.ndarray,
                 top_k: int = 5, reference: Optional[Reference] = None) -> list[dict[str, Any]]:
    """The top_k features of one text by |contribution|, largest first (ties: input order)."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    order = np.argsort(-np.abs(contributions), kind="stable")[:top_k]
    rows = []
    for j in order:
        name = str(names[j])
        contribution = float(contributions[j])
        direction = "ai" if contribution > 0 else "human"
        value = _num(values[j])
        ref = (reference or {}).get(name) or {}
        rows.append({
            "feature": name,
            "group": lookup(name).group,
            "value": value,
            "typical_human": ref.get("human"),
            "typical_ai": ref.get("ai"),
            "contribution": contribution,
            "direction": direction,
            "reason": reason(name, direction, float("nan") if value is None else value,
                             _typical(ref)),
        })
    return rows


def explain_features(feature_model: Any, texts: Sequence[str] | None = None,
                     X: np.ndarray | None = None, top_k: int = 5,
                     reference: Optional[Reference] = None) -> list[dict[str, Any]]:
    """
    Per text: {"p_ai", "raw_score", "base_value", "top_features": [...]}.

    Pass the texts, or a feature matrix X already computed with
    `feature_model.featurize`. p_ai and raw_score are the feature model's own
    P(ai) and log-odds (base_value + sum of all contributions). Each top
    feature is {"feature", "group", "value", "typical_human", "typical_ai",
    "contribution", "direction", "reason"}: contribution in log-odds,
    direction "ai" if it pushes towards AI else "human", value None when
    the feature could not be computed, typical_* None without a reference.
    """
    if X is None:
        if texts is None:
            raise ValueError("pass texts or X")
        X = feature_model.featurize(list(texts))
    X = np.asarray(X, dtype=np.float64)
    contrib, base = shap_values(feature_model, X)
    raw = base + contrib.sum(axis=1)
    names = feature_model.feature_names_
    out = []
    for i in range(len(X)):
        out.append({
            "p_ai": _sigmoid(float(raw[i])),
            "raw_score": float(raw[i]),
            "base_value": float(base[i]),
            "top_features": top_features(names, X[i], contrib[i], top_k, reference),
        })
    return out
