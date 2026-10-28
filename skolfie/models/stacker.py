"""
skolfie/models/stacker.py
--------------------------
Roadmap step 6: one calibrated score from several models, and the cut-offs
that turn it into a Human / Uncertain / AI verdict.

Stacker
    Logistic regression on the logits of the member models' P(ai), fitted on
    a split none of the members trained on (`val`). Logistic regression is
    itself a Platt-style calibration; `calibration="isotonic"` adds an
    isotonic map on top, fitted on out-of-fold scores. `oof_predict()` gives
    cross-validated scores for the fitting split, so its own report row is
    not in-sample.

Cut-offs (`operating_points`)
    ai_threshold     P(ai) above it is called AI. Chosen on the fitting split
                     so that at most `max_fpr` of human texts score above it.
    human_threshold  P(ai) below it is called Human. Chosen so that at most
                     `max_miss` of AI texts score below it.
    Between the two the verdict is Uncertain. A false accusation is the
    costlier mistake, so `max_fpr` is the stricter target by default.

StackedModel
    The scorer used after training: loads the member models (joblib files,
    lazily), stacks their scores and applies the cut-offs. `predict_proba()`
    follows the common scorer contract; `predict()` returns the verdicts.
    Texts are normalised once (`scoring_texts`) before the members
    see them, so members of any type get the Unicode defence.
    The member paths are stored as given to `stack.py --model-path` (on
    Colab, absolute Drive paths); on another machine, copy the member files
    there or point `model.members[name]` at their new location. Loading a
    member unpickles it, so the same trust rule as for `JoblibScorer`
    applies: a StackedModel is only as trustworthy as the files it names.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from skolfie.eval.metrics import to_binary
from skolfie.models.feature_model import scoring_texts

CALIBRATIONS = ("platt", "isotonic")
VERDICTS = ("human", "uncertain", "ai")
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Stacker:
    """Calibrated logistic stack over member P(ai) columns."""

    def __init__(self, calibration: str = "platt", C: float = 1.0, folds: int = 5,
                 seed: int = 0) -> None:
        if calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}, got {calibration!r}")
        if folds < 2:
            raise ValueError("folds must be at least 2")
        self.calibration = calibration
        self.C = C
        self.folds = folds
        self.seed = seed
        self.lr_: Any = None
        self.iso_: Any = None

    def _check(self, P: np.ndarray, y: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
        P = np.asarray(P, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] == 0:
            raise ValueError(f"P must be (n, n_members), got shape {P.shape}")
        if not np.isfinite(P).all():
            raise ValueError("P contains NaN or inf")
        if y is None:
            return P, None
        y = to_binary(y)
        if len(y) != len(P):
            raise ValueError("P and y must have the same length")
        if np.unique(y).size < 2:
            raise ValueError("need both human and ai texts to fit the stacker")
        return P, y

    def _new_lr(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=self.C, max_iter=1000)

    def _folds(self, y: np.ndarray) -> Any:
        from sklearn.model_selection import StratifiedKFold

        n_splits = int(min(self.folds, np.bincount(y).min()))
        if n_splits < 2:
            raise ValueError("need at least 2 texts of each label for cross-fitting")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

    def _oof_lr(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(len(y))
        for fit_idx, score_idx in self._folds(y).split(X, y):
            lr = self._new_lr().fit(X[fit_idx], y[fit_idx])
            out[score_idx] = lr.predict_proba(X[score_idx])[:, 1]
        return out

    def fit(self, P: np.ndarray, y: Sequence[Any]) -> "Stacker":
        P, y = self._check(P, y)
        X = logit(P)
        self.lr_ = self._new_lr().fit(X, y)
        self.iso_ = None
        if self.calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.iso_.fit(self._oof_lr(X, y), y)
        return self

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        if self.lr_ is None:
            raise RuntimeError("Stacker is not fitted")
        P, _ = self._check(P)
        p = self.lr_.predict_proba(logit(P))[:, 1]
        return self.iso_.predict(p) if self.iso_ is not None else p

    def oof_predict(self, P: np.ndarray, y: Sequence[Any]) -> np.ndarray:
        """Cross-validated scores for the fitting rows (each from a stack that never saw it)."""
        P, y = self._check(P, y)
        out = np.empty(len(y))
        for fit_idx, score_idx in self._folds(y).split(P, y):
            part = Stacker(self.calibration, self.C, self.folds, self.seed)
            out[score_idx] = part.fit(P[fit_idx], y[fit_idx]).predict_proba(P[score_idx])
        return out

    def weights(self, names: Sequence[str]) -> dict[str, float]:
        """Logistic weight per member logit, plus the intercept."""
        if self.lr_ is None:
            return {}
        out = {str(n): float(w) for n, w in zip(names, self.lr_.coef_[0])}
        out["intercept"] = float(self.lr_.intercept_[0])
        return out


# --------------------------------------------------------------------------- #
# Cut-offs and verdicts
# --------------------------------------------------------------------------- #
def operating_points(y: Sequence[Any], p_ai: Sequence[float], max_fpr: float = 0.01,
                     max_miss: float = 0.05) -> dict[str, float]:
    """Cut-offs from a fitting split; see the module docstring."""
    y = to_binary(y)
    p = np.asarray(p_ai, dtype=np.float64)
    if not (0 < max_fpr < 1 and 0 < max_miss < 1):
        raise ValueError("max_fpr and max_miss must be in (0, 1)")
    if np.unique(y).size < 2:
        raise ValueError("need both human and ai texts to set the cut-offs")
    ai_threshold = float(np.quantile(p[y == 0], 1 - max_fpr, method="higher"))
    human_threshold = float(np.quantile(p[y == 1], max_miss, method="lower"))
    # Well-separated scores: no Uncertain band rather than an inverted one.
    human_threshold = min(human_threshold, ai_threshold)
    return {"ai_threshold": ai_threshold, "human_threshold": human_threshold,
            "max_fpr": max_fpr, "max_miss": max_miss}


def verdicts(p_ai: Sequence[float], cutoffs: dict[str, float]) -> np.ndarray:
    """"ai" above the AI cut-off, "human" below the human cut-off, else "uncertain"."""
    p = np.asarray(p_ai, dtype=np.float64)
    out = np.full(p.shape, "uncertain", dtype=object)
    out[p < cutoffs["human_threshold"]] = "human"
    out[p > cutoffs["ai_threshold"]] = "ai"
    return out


def verdict_table(y: Sequence[Any], p_ai: Sequence[float],
                  cutoffs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Share of each verdict among human texts and among AI texts."""
    y = to_binary(y)
    v = verdicts(p_ai, cutoffs)
    table: dict[str, dict[str, float]] = {}
    for name, cls in (("human_texts", 0), ("ai_texts", 1)):
        mask = y == cls
        n = int(mask.sum())
        row: dict[str, float] = {"n": n}
        for verdict in VERDICTS:
            row[verdict] = float((v[mask] == verdict).mean()) if n else float("nan")
        table[name] = row
    return table


# --------------------------------------------------------------------------- #
# The trained scorer
# --------------------------------------------------------------------------- #
class StackedModel:
    """Member models + Stacker + cut-offs, behind the common scorer interface."""

    pretrained = True
    name = "stacked"

    def __init__(self, members: dict[str, str | os.PathLike[str]], stacker: Stacker,
                 cutoffs: dict[str, float], info: dict[str, Any] | None = None) -> None:
        if not members:
            raise ValueError("need at least one member model")
        self.members = {k: os.fspath(v) for k, v in members.items()}
        self.stacker = stacker
        self.cutoffs = dict(cutoffs)
        self.info = dict(info or {})
        self._loaded: dict[str, Any] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_loaded"] = {}
        return state

    def _member(self, name: str) -> Any:
        if name not in self._loaded:
            from skolfie.models.baselines import JoblibScorer

            path = self.members[name]
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"StackedModel member {name!r}: model file not found: {path} (paths are "
                    f"recorded by stack.py --model-path; copy the file there or set "
                    f"model.members[{name!r}] to its new path)")
            self._loaded[name] = JoblibScorer(path)
        return self._loaded[name]

    def params(self) -> dict[str, Any]:
        return {"members": self.members, "calibration": self.stacker.calibration,
                "weights": self.stacker.weights(list(self.members)), "cutoffs": self.cutoffs,
                **self.info}

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "StackedModel":
        return self

    def member_scores(self, texts: Sequence[str]) -> np.ndarray:
        texts = scoring_texts(texts)
        return np.column_stack([self._member(n).predict_proba(texts) for n in self.members])

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros(0)
        return self.stacker.predict_proba(self.member_scores(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Verdicts: "human", "uncertain" or "ai" per text."""
        return verdicts(self.predict_proba(texts), self.cutoffs)
