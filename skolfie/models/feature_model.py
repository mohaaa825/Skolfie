"""
skolfie/models/feature_model.py
--------------------------------
Model B of the roadmap: a fast, explainable detector on hand-made features.

Features (one row per text):

    sty_*     compact stylometric features (`skolfie.features.style`)
    lm_*      GPT-2 perplexity, burstiness and token-rank features
              (`skolfie.features.lm`; optional, needs torch + transformers,
              practical only on a GPU for a full corpus)
    delta_*   Cosine/Burrows Delta distances to the human and AI word
              profiles (`skolfie.features.delta_features`)

Classifier: LightGBM (default) or a random forest for comparison.

Leakage: the Delta profiles are learned from labelled texts, so a text's
own label would leak into its Delta features if the profiles were fitted on
the whole training split. `fit()` therefore computes the training rows'
Delta features out-of-fold (`delta_folds` folds, each scored by profiles
fitted on the other folds) and fits the final profiles on the whole split
only for scoring new texts. The style and LM features do not use labels.

Speed: style and LM features are cached on disk by text hash when
`cache_dir` is given (see `skolfie.features.cache`), so a second run over
the same corpus only refits the Delta step and the classifier.

The scorer interface matches `skolfie.models.baselines` (fit /
predict_proba / params), plus `feature_importance()`, which the evaluation
report prints.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import numpy as np

from skolfie.eval.metrics import to_binary

# Bump when a feature formula changes without its name changing, so the
# on-disk caches are rebuilt.
STYLE_CACHE_VERSION = 1
LM_CACHE_VERSION = 1
CLASSIFIERS = ("lgbm", "rf")


class FeatureModel:
    pretrained = False

    def __init__(self, classifier: str = "lgbm", use_lm: bool = True,
                 lm_model: str = "gpt2", lm_max_tokens: int = 512, lm_batch_size: int = 32,
                 lm_device: str | None = None, delta_words: int = 300, delta_folds: int = 5,
                 cache_dir: str | os.PathLike[str] | None = None, n_jobs: int = -1,
                 seed: int = 0, lgbm_params: dict[str, Any] | None = None,
                 log: Callable[[str], None] = print) -> None:
        if classifier not in CLASSIFIERS:
            raise ValueError(f"classifier must be one of {CLASSIFIERS}, got {classifier!r}")
        if delta_folds < 2:
            raise ValueError("delta_folds must be at least 2")
        self.classifier = classifier
        self.use_lm = use_lm
        self.lm_model = lm_model
        self.lm_max_tokens = lm_max_tokens
        self.lm_batch_size = lm_batch_size
        self.lm_device = lm_device
        self.delta_words = delta_words
        self.delta_folds = delta_folds
        self.cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        self.n_jobs = n_jobs
        self.seed = seed
        self.lgbm_params = dict(lgbm_params or {})
        self.log = log
        self.name = "features" + ("" if use_lm else "-nolm") + ("-rf" if classifier == "rf" else "")
        self.feature_names_: list[str] = []
        self.delta_: Any = None
        self.clf: Any = None
        self.best_iteration_: int | None = None
        self._lm: Any = None
        self._caches: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Pickling: drop the GPT-2 weights and the caches (reloaded lazily).
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_lm"] = None
        state["_caches"] = {}
        state["log"] = print
        return state

    def params(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in (
            "classifier", "use_lm", "delta_words", "delta_folds", "seed")}
        if self.use_lm:
            out.update(lm_model=self.lm_model, lm_max_tokens=self.lm_max_tokens)
        if self.classifier == "lgbm":
            out.update(lgbm=self._lgbm_kwargs(), best_iteration=self.best_iteration_)
        out["n_features"] = len(self.feature_names_)
        return out

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #
    def _cache(self, kind: str, name: str, names: Sequence[str]) -> Any:
        from skolfie.features.cache import FeatureCache

        if kind not in self._caches:
            self._caches[kind] = FeatureCache(self.cache_dir, name, names, log=self.log)
        return self._caches[kind]

    def _style(self, texts: list[str]) -> np.ndarray:
        from skolfie.features.style import STYLE_FEATURES, style_matrix

        cache = self._cache("style", f"style-v{STYLE_CACHE_VERSION}", STYLE_FEATURES)
        return cache.compute(texts, lambda chunk: style_matrix(chunk, n_jobs=self.n_jobs))

    def _lm_featurizer(self) -> Any:
        if self._lm is None:
            from skolfie.features.lm import LMFeaturizer

            self._lm = LMFeaturizer(model_name=self.lm_model, device=self.lm_device,
                                    max_tokens=self.lm_max_tokens,
                                    batch_size=self.lm_batch_size)
        return self._lm

    def _lm_features(self, texts: list[str]) -> np.ndarray:
        from skolfie.features.lm import LM_FEATURES

        lm = self._lm_featurizer()
        name = f"lm-v{LM_CACHE_VERSION}-{lm.cache_key}".replace("/", "_")
        cache = self._cache("lm", name, LM_FEATURES)
        return cache.compute(texts, lambda chunk: lm.features(chunk, progress=True))

    def base_features(self, texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """(style [+ LM] feature matrix, names): the label-free features, cached.

        Needs no fitting; the transformer model (step 5) reuses it.
        """
        return self._base_features(list(texts))

    def _base_features(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        from skolfie.features.style import STYLE_FEATURES

        blocks, names = [self._style(texts)], list(STYLE_FEATURES)
        if self.use_lm:
            from skolfie.features.lm import LM_FEATURES

            blocks.append(self._lm_features(texts))
            names += list(LM_FEATURES)
        return np.hstack(blocks).astype(np.float32), names

    def _new_delta(self) -> Any:
        from skolfie.features.delta_features import DeltaFeaturizer

        return DeltaFeaturizer(n_words=self.delta_words)

    def _oof_delta(self, texts: list[str], y: np.ndarray) -> np.ndarray:
        from sklearn.model_selection import StratifiedKFold

        from skolfie.features.delta_features import DELTA_FEATURES

        out = np.full((len(texts), len(DELTA_FEATURES)), np.nan, dtype=np.float32)
        folds = StratifiedKFold(n_splits=self.delta_folds, shuffle=True, random_state=self.seed)
        for fit_idx, score_idx in folds.split(np.zeros(len(y)), y):
            delta = self._new_delta().fit([texts[i] for i in fit_idx], y[fit_idx])
            out[score_idx] = delta.transform([texts[i] for i in score_idx])
        return out

    def featurize(self, texts: Sequence[str]) -> np.ndarray:
        """Feature matrix for new texts (columns in `feature_names_` order)."""
        if self.delta_ is None:
            raise RuntimeError("FeatureModel is not fitted")
        texts = list(texts)
        base, _ = self._base_features(texts)
        return np.hstack([base, self.delta_.transform(texts)])

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def _lgbm_kwargs(self) -> dict[str, Any]:
        kwargs = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63,
                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, reg_lambda=1.0)
        kwargs.update(self.lgbm_params)
        return kwargs

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray) -> Any:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        # A small internal hold-out picks the number of trees; the harness's
        # own val split stays untouched.
        X_tr, X_es, y_tr, y_es = train_test_split(X, y, test_size=0.05, stratify=y,
                                                  random_state=self.seed)
        kwargs = self._lgbm_kwargs()
        # Small training sets (tests, quick runs) could not split at all otherwise.
        kwargs["min_child_samples"] = min(kwargs["min_child_samples"], max(2, len(y_tr) // 20))
        clf = lgb.LGBMClassifier(**kwargs, random_state=self.seed,
                                 n_jobs=self.n_jobs, verbose=-1)
        clf.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        self.best_iteration_ = int(clf.best_iteration_ or clf.n_estimators)
        self.log(f"[fit] lightgbm: {self.best_iteration_} trees")
        return clf

    def _fit_rf(self, X: np.ndarray, y: np.ndarray) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                   n_jobs=self.n_jobs, random_state=self.seed),
        ).fit(X, y)

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "FeatureModel":
        from skolfie.features.delta_features import DELTA_FEATURES

        texts = list(texts)
        y = to_binary(labels)
        if len(texts) != len(y):
            raise ValueError("texts and labels must have the same length")
        if np.unique(y).size < 2:
            raise ValueError("need both human and ai training texts")
        base, names = self._base_features(texts)
        self.log(f"[fit] Delta features, {self.delta_folds}-fold out-of-fold")
        oof = self._oof_delta(texts, y)
        self.delta_ = self._new_delta().fit(texts, y)
        X = np.hstack([base, oof])
        self.feature_names_ = names + list(DELTA_FEATURES)
        self.clf = self._fit_lgbm(X, y) if self.classifier == "lgbm" else self._fit_rf(X, y)
        for cache in self._caches.values():
            cache.flush()
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("FeatureModel is not fitted")
        if len(texts) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(self.featurize(texts))[:, 1].astype(np.float64)

    def feature_importance(self) -> list[dict[str, Any]]:
        """[{feature, importance}] sorted descending; importances sum to 1.

        LightGBM: total split gain per feature. Random forest: mean impurity
        decrease. Both are global, model-internal measures; per-text
        explanations (SHAP) come in roadmap step 8.
        """
        if self.clf is None:
            return []
        if self.classifier == "lgbm":
            raw = self.clf.booster_.feature_importance(importance_type="gain")
        else:
            raw = self.clf[-1].feature_importances_
        raw = np.asarray(raw, dtype=np.float64)
        total = raw.sum() or 1.0
        rows = [{"feature": f, "importance": float(v / total)}
                for f, v in zip(self.feature_names_, raw)]
        return sorted(rows, key=lambda r: -r["importance"])
