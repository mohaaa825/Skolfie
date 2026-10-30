"""
skolfie/models/baselines.py
----------------------------
Reference scorers for the held-out evaluation harness (`scripts/evaluate.py`).

Every scorer has the same small interface:

    scorer.name                          short id used in reports
    scorer.pretrained                    True if fit() is a no-op (already trained)
    scorer.fit(texts, labels)            labels: "human"/"ai" or 0/1 (1 = ai)
    scorer.predict_proba(texts)          -> np.ndarray of P(ai), shape (n,), in [0, 1]

Scorers:

    LengthBaseline     logistic regression on 3 shallow features; a deliberately
                       weak floor that any real model must beat by a wide margin
    TfidfLRBaseline    word 1-2 gram + char 3-5 gram TF-IDF -> LogisticRegression,
                       sized so that ~200k training texts fit on a Colab CPU
    AIDetectorAdapter  wraps the existing `skolfie.models.ensemble.AIDetectorModel`
                       (slow: one predict() call per text)
    JoblibScorer       any pickled object with predict_proba(texts), so later
                       models plug in without touching the harness

The feature model (roadmap step 4) lives in `skolfie.models.feature_model`
and the DistilBERT + features model (step 5) in
`skolfie.models.transformer_model`; `make_scorer` builds them for the
`features*` and `transformer*` specs.

LengthBaseline and TfidfLRBaseline map their input through
`skolfie.robust.normalize.normalize_for_scoring` like the real models, so
look-alike letters and invisible characters do not move them either.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from skolfie.eval.metrics import to_binary
from skolfie.robust.normalize import normalize_for_scoring


@runtime_checkable
class Scorer(Protocol):
    name: str
    pretrained: bool

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "Scorer": ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Length / shallow-statistics floor
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_WORD = re.compile(r"\w+", re.UNICODE)


def shallow_features(text: str) -> list[float]:
    """[log(1 + word count), mean words per sentence, type/token ratio]."""
    words = text.split()
    n_words = len(words)
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    mean_sentence = n_words / max(1, len(sentences))
    tokens = _WORD.findall(text.lower())
    ttr = len(set(tokens)) / len(tokens) if tokens else 0.0
    return [math.log1p(n_words), mean_sentence, ttr]


class LengthBaseline:
    """Logistic regression on log word count, mean sentence length and type/token ratio."""

    name = "length"
    pretrained = False

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.model: Any = None

    @staticmethod
    def featurize(texts: Sequence[str]) -> np.ndarray:
        return np.asarray([shallow_features(normalize_for_scoring(t)) for t in texts],
                          dtype=np.float64).reshape(-1, 3)

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "LengthBaseline":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = to_binary(labels)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=1000))
        self.model.fit(self.featurize(texts), y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LengthBaseline is not fitted")
        if len(texts) == 0:
            return np.zeros(0)
        return self.model.predict_proba(self.featurize(texts))[:, 1]

    def params(self) -> dict[str, Any]:
        return {"C": self.C, "features": ["log1p_words", "mean_sentence_words", "type_token_ratio"]}


# --------------------------------------------------------------------------- #
# TF-IDF + logistic regression
# --------------------------------------------------------------------------- #
class TfidfLRBaseline:
    """
    Word 1-2 gram + char 3-5 gram TF-IDF features into one LogisticRegression.

    Sized for ~200k training texts in ~8 GB of RAM on a 2-core Colab CPU:

    * each text is cut to its first `max_chars` characters before vectorising
      (char n-grams per text grow with its length, and they dominate memory);
    * the vocabularies and IDF weights are learned on a random subsample of
      `vocab_sample` training texts (the full-vocabulary count matrix that
      sklearn builds before pruning to `max_features` is the peak-memory step,
      and a 30k sample already ranks the frequent n-grams reliably);
    * all texts are then transformed in chunks of `chunk_size` into float32
      CSR, so the peak stays near the final matrix size;
    * LogisticRegression (lbfgs) trains directly on that float32 matrix.

    `n_jobs` != 1 transforms chunks in parallel worker processes (-1 = all
    cores; each worker holds one chunk's intermediates on top).

    Scale check (synthetic English-like texts, mean 1,440 chars, 4-core
    container, 2 workers): fit on 200k texts took 5.7 min with a 4.4 GB peak
    RSS in the main process (2,117 nonzeros per text, 384k features, float32
    matrix ~3.4 GB); single-process it extrapolates to ~10 min. Real text may
    need more lbfgs iterations than the synthetic run's 24 (~1-2 s each).
    """

    name = "tfidf"
    pretrained = False

    def __init__(self, word_max_features: int = 200_000, char_max_features: int = 300_000,
                 char_analyzer: str = "char_wb", max_chars: int = 2000,
                 vocab_sample: int = 30_000, chunk_size: int = 5_000,
                 C: float = 4.0, max_iter: int = 1000, seed: int = 0, n_jobs: int = -1) -> None:
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.char_analyzer = char_analyzer
        self.max_chars = max_chars
        self.vocab_sample = vocab_sample
        self.chunk_size = chunk_size
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.n_jobs = n_jobs
        self.word_vec: Any = None
        self.char_vec: Any = None
        self.clf: Any = None

    def params(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "word_max_features", "char_max_features", "char_analyzer", "max_chars", "vocab_sample",
            "chunk_size", "C", "max_iter", "seed")}

    def _clip(self, texts: Sequence[str]) -> list[str]:
        return [normalize_for_scoring(t[: self.max_chars]) for t in texts]

    def _transform(self, texts: Sequence[str]) -> Any:
        texts = self._clip(texts)
        chunks = [texts[i:i + self.chunk_size] for i in range(0, len(texts), self.chunk_size)]
        if self.n_jobs != 1 and len(chunks) > 1:
            from joblib import Parallel, delayed

            parts = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_chunk)(self.word_vec, self.char_vec, c) for c in chunks)
        else:
            parts = [_transform_chunk(self.word_vec, self.char_vec, c) for c in chunks]
        n_features = len(self.word_vec.vocabulary_) + len(self.char_vec.vocabulary_)
        return _stack_rows(parts, n_features)

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "TfidfLRBaseline":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        y = to_binary(labels)
        if np.unique(y).size < 2:
            raise ValueError("need both human and ai training texts")
        texts = list(texts)
        rng = np.random.default_rng(self.seed)
        if len(texts) > self.vocab_sample:
            idx = rng.choice(len(texts), size=self.vocab_sample, replace=False)
            sample = [texts[i] for i in np.sort(idx)]
        else:
            sample = texts
        sample = self._clip(sample)

        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=2,
            max_features=self.word_max_features, sublinear_tf=True, dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",  # keep 1-letter words and punctuation
        )
        self.char_vec = TfidfVectorizer(
            analyzer=self.char_analyzer, ngram_range=(3, 5), lowercase=False, min_df=2,
            max_features=self.char_max_features, sublinear_tf=True, dtype=np.float32,
        )
        for vec in (self.word_vec, self.char_vec):
            vec.fit(sample)
            # The pruned-term set is huge, unused after fit, and would bloat the pickle.
            if hasattr(vec, "stop_words_"):
                vec.stop_words_ = None
        del sample

        X = self._transform(texts)
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("TfidfLRBaseline is not fitted")
        if len(texts) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(self._transform(list(texts)))[:, 1].astype(np.float64)


def _transform_chunk(word_vec: Any, char_vec: Any, texts: Sequence[str]) -> Any:
    import scipy.sparse as sp

    return sp.hstack([word_vec.transform(texts), char_vec.transform(texts)],
                     format="csr", dtype=np.float32)


def _stack_rows(parts: list[Any], n_features: int) -> Any:
    """
    Stack CSR row blocks into one float32 CSR matrix, freeing each block as it
    is copied, so the peak is the result plus one block (sp.vstack holds every
    block and the result at once). Uses int32 indices while nnz allows it.
    """
    import scipy.sparse as sp

    n_rows = sum(p.shape[0] for p in parts)
    nnz = sum(p.nnz for p in parts)
    index_dtype = np.int32 if max(nnz, n_features) < np.iinfo(np.int32).max else np.int64
    data = np.empty(nnz, dtype=np.float32)
    indices = np.empty(nnz, dtype=index_dtype)
    indptr = np.empty(n_rows + 1, dtype=index_dtype)
    indptr[0] = 0
    row = pos = 0
    for i in range(len(parts)):
        part = parts[i]
        parts[i] = None
        k = part.nnz
        data[pos:pos + k] = part.data
        indices[pos:pos + k] = part.indices
        indptr[row + 1:row + 1 + part.shape[0]] = part.indptr[1:] + pos
        row += part.shape[0]
        pos += k
        del part
    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_features), copy=False)


# --------------------------------------------------------------------------- #
# Existing ensemble model
# --------------------------------------------------------------------------- #
class AIDetectorAdapter:
    """
    The existing `AIDetectorModel` ensemble behind the common scorer interface.

    Pass `model_path` to load a trained model with joblib (fit() is then a
    no-op), or nothing to train a fresh model in fit(). predict_proba() calls
    `predict()` once per text, so it is slow: cap eval sizes (--max-eval).
    """

    name = "ensemble"

    def __init__(self, model_path: str | os.PathLike[str] | None = None,
                 model: Any = None, random_state: int = 42) -> None:
        self.model_path = None if model_path is None else os.fspath(model_path)
        self.random_state = random_state
        self.model = model
        if self.model is None and self.model_path is not None:
            from skolfie.models.ensemble import AIDetectorModel

            self.model = AIDetectorModel.load(self.model_path)
        self.pretrained = self.model is not None

    def params(self) -> dict[str, Any]:
        return {"model_path": self.model_path, "random_state": self.random_state}

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "AIDetectorAdapter":
        if self.pretrained:
            return self
        from skolfie.models.ensemble import AIDetectorModel

        str_labels = ["ai" if v else "human" for v in to_binary(labels)]
        self.model = AIDetectorModel()
        self.model.train(list(texts), str_labels, random_state=self.random_state)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("AIDetectorAdapter has no model: call fit() or pass model_path")
        return np.asarray([self.model.predict(t)["ai_probability"] for t in texts],
                          dtype=np.float64)


# --------------------------------------------------------------------------- #
# Any pickled model
# --------------------------------------------------------------------------- #
class JoblibScorer:
    """
    Load any joblib/pickle file holding an object with `predict_proba(texts)`.

    The object may return P(ai) with shape (n,), or an (n, 2) sklearn-style
    array (the "ai"/1 column is picked via `classes_` if present, else column
    1). A pickled `AIDetectorModel` is wrapped in `AIDetectorAdapter`. The
    object is treated as already trained: fit() is a no-op.

    Trust boundary: unpickling runs arbitrary code, so `path` must be a model
    file you (or your own training run) wrote, never one derived from user
    input or downloaded from an untrusted place. The callers only pass paths
    from the command line, an environment variable or a `StackedModel` that
    was itself loaded from such a path.
    """

    pretrained = True

    def __init__(self, path: str | os.PathLike[str]) -> None:
        import joblib

        self.path = os.fspath(path)
        obj = joblib.load(self.path)
        from skolfie.models.ensemble import AIDetectorModel

        if isinstance(obj, AIDetectorModel):
            obj = AIDetectorAdapter(model=obj)
            obj.model_path = self.path
        if not callable(getattr(obj, "predict_proba", None)):
            raise TypeError(f"{self.path}: loaded {type(obj).__name__} has no predict_proba()")
        self.obj = obj
        self.name = str(getattr(obj, "name", None) or f"joblib:{Path(self.path).stem}")

    def params(self) -> dict[str, Any]:
        return {"path": self.path, "type": type(self.obj).__name__}

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "JoblibScorer":
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        out = np.asarray(self.obj.predict_proba(list(texts)), dtype=np.float64)
        if out.ndim == 2:
            if out.shape[1] != 2:
                raise ValueError(
                    f"predict_proba returned shape {out.shape}; expected (n,) or (n, 2)")
            col = 1
            classes = [str(c).lower() for c in getattr(self.obj, "classes_", [])]
            if "ai" in classes:
                col = classes.index("ai")
            out = out[:, col]
        if out.shape != (len(texts),):
            raise ValueError(f"predict_proba returned shape {out.shape} for {len(texts)} texts")
        return out


FEATURE_SPECS = {
    "features": {"classifier": "lgbm", "use_lm": True},
    "features-nolm": {"classifier": "lgbm", "use_lm": False},
    "features-rf": {"classifier": "rf", "use_lm": True},
    "features-nolm-rf": {"classifier": "rf", "use_lm": False},
}
TRANSFORMER_SPECS = {
    "transformer": {"use_features": True},
    "transformer-text": {"use_features": False},
    "transformer-chunks": {"use_features": True, "chunk_augment": 1},
}
# make_scorer options that only the transformer model takes.
_TRANSFORMER_ONLY = ("model_name", "max_length", "epochs", "batch_size")
# make_scorer options that only the feature model takes (the transformer keeps
# its own style + GPT-2 features; Binoculars is too slow for its training set).
_FEATURES_ONLY = ("bino_observer", "bino_performer", "bino_max_tokens", "bino_batch_size",
                  "bino_quantization")


def make_scorer(spec: str, seed: int = 0, **options: Any) -> Any:
    """
    Build a scorer from a CLI spec: length | tfidf | ensemble | ensemble:PATH |
    joblib:PATH | features | features-nolm | features-rf | features-nolm-rf |
    transformer | transformer-text | transformer-chunks.

    `options` go to `FeatureModel` (cache_dir, lm_model, lm_batch_size, ...,
    also bino_observer, bino_performer, bino_max_tokens, bino_batch_size,
    bino_quantization) and `TransformerModel` (also model_name, max_length,
    epochs, batch_size) and are ignored by the other scorers; None values are
    dropped.
    """
    kwargs = {k: v for k, v in options.items() if v is not None}
    if spec == "length":
        return LengthBaseline()
    if spec == "tfidf":
        return TfidfLRBaseline(seed=seed)
    if spec in FEATURE_SPECS:
        from skolfie.models.feature_model import FeatureModel

        feature_kwargs = {k: v for k, v in kwargs.items() if k not in _TRANSFORMER_ONLY}
        return FeatureModel(**FEATURE_SPECS[spec], seed=seed, **feature_kwargs)
    if spec in TRANSFORMER_SPECS:
        from skolfie.models.transformer_model import TransformerModel

        transformer_kwargs = {k: v for k, v in kwargs.items() if k not in _FEATURES_ONLY}
        return TransformerModel(**TRANSFORMER_SPECS[spec], seed=seed, **transformer_kwargs)
    if spec == "ensemble":
        return AIDetectorAdapter(random_state=seed)
    if spec.startswith("ensemble:"):
        return AIDetectorAdapter(model_path=spec.split(":", 1)[1])
    if spec.startswith("joblib:"):
        return JoblibScorer(spec.split(":", 1)[1])
    raise ValueError(f"unknown model spec {spec!r}; use length, tfidf, ensemble, "
                     f"{', '.join([*FEATURE_SPECS, *TRANSFORMER_SPECS])}, ensemble:PATH or "
                     "joblib:PATH")
