"""
burrows_delta.py
-----------------
A from-scratch implementation of Burrows' Delta (Burrows, 2002) used as a
distributional "stylistic distance" signal for AI-vs-human detection.

How it works:
    1. Pick the N most frequent words (MFW) in the training corpus,
       optionally restricted to a candidate vocabulary (e.g. function words).
    2. For every training document compute the relative frequency of each
       MFW, then the corpus-wide mean and standard deviation per word.
    3. Convert each document's frequencies to z-scores with those stats.
    4. Build two profiles -- the centroid (mean z-vector) of the "human"
       documents and of the "ai" documents.
    5. A new text is z-scored the same way and compared to both profiles:
         * "burrows": classic Delta = mean absolute z-score difference
         * "cosine" : Cosine Delta (Jannidis et al., 2015) = 1 - cosine
                      similarity between the z-vectors
    6. The difference (delta_human - delta_ai) is squashed through a
       logistic function, calibrated on the training corpus, to give an
       AI probability (closer to the AI profile -> higher probability).

Words with zero standard deviation in the training corpus get a std of
1.0 so they never cause a division by zero (their z-score is then simply
the centred frequency, which is ~0 for typical texts). Empty texts, or
texts containing none of the MFW tokens, return a neutral 0.5 probability.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

_VALID_METHODS = ("burrows", "cosine")
_HUMAN_LABEL = "human"
_AI_LABEL = "ai"


def _normalise_tokens(tokens: Iterable[str]) -> List[str]:
    """Lower-case tokens and drop pure punctuation / numeric tokens."""
    return [t.lower() for t in tokens if isinstance(t, str) and any(c.isalpha() for c in t)]


def _sigmoid(x: float) -> float:
    # Numerically stable logistic function.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class BurrowsDeltaModel:
    """Burrows' Delta / Cosine Delta classifier over most-frequent-word z-scores."""

    def __init__(self, method: str = "burrows"):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.method = method
        self.vocabulary: List[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.human_profile_: Optional[np.ndarray] = None
        self.ai_profile_: Optional[np.ndarray] = None
        # Scale for the logistic mapping of (delta_human - delta_ai).
        self.scale_: float = 1.0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return bool(self.vocabulary) and self.human_profile_ is not None

    def _relative_frequencies(self, tokens: Sequence[str]) -> Optional[np.ndarray]:
        """Relative frequency of each vocabulary word, or None for empty input."""
        words = _normalise_tokens(tokens)
        if not words:
            return None
        counts = Counter(words)
        total = float(len(words))
        return np.array([counts.get(w, 0) / total for w in self.vocabulary], dtype=float)

    def _zscores(self, freqs: np.ndarray) -> np.ndarray:
        return (freqs - self.mean_) / self.std_

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.method == "cosine":
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                # Cosine is undefined for a zero vector: identical -> 0,
                # otherwise treat as orthogonal.
                return 0.0 if np.allclose(a, b) else 1.0
            cos = float(np.dot(a, b) / (na * nb))
            return float(1.0 - np.clip(cos, -1.0, 1.0))
        return float(np.mean(np.abs(a - b)))

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def fit(
        self,
        tokens_list: Sequence[Sequence[str]],
        labels: Sequence[str],
        n_words: int = 100,
        candidate_vocab: Optional[Iterable[str]] = None,
    ) -> "BurrowsDeltaModel":
        """
        Fit the model.

        tokens_list     : one token list per document
        labels          : matching "human" / "ai" labels (case-insensitive)
        n_words         : number of most frequent words to keep
        candidate_vocab : if given, MFW are chosen only from these words
        """
        if len(tokens_list) != len(labels):
            raise ValueError("tokens_list and labels must have the same length")
        norm_labels = [str(lab).strip().lower() for lab in labels]
        unknown = set(norm_labels) - {_HUMAN_LABEL, _AI_LABEL}
        if unknown:
            raise ValueError(f"labels must be 'human' or 'ai', got {sorted(unknown)}")
        if _HUMAN_LABEL not in norm_labels or _AI_LABEL not in norm_labels:
            raise ValueError("training data must contain both 'human' and 'ai' examples")

        docs = [_normalise_tokens(toks) for toks in tokens_list]
        keep = [i for i, d in enumerate(docs) if d]
        docs = [docs[i] for i in keep]
        norm_labels = [norm_labels[i] for i in keep]
        if _HUMAN_LABEL not in norm_labels or _AI_LABEL not in norm_labels:
            raise ValueError("training data must contain non-empty 'human' and 'ai' examples")

        corpus_counts: Counter = Counter()
        for d in docs:
            corpus_counts.update(d)
        if candidate_vocab is not None:
            allowed = {w.lower() for w in candidate_vocab}
            corpus_counts = Counter({w: c for w, c in corpus_counts.items() if w in allowed})
        # Sort by frequency, then alphabetically for determinism.
        ranked = sorted(corpus_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.vocabulary = [w for w, _ in ranked[: max(int(n_words), 0)]]
        if not self.vocabulary:
            raise ValueError("no vocabulary words found in the training corpus")

        freqs = np.vstack([self._relative_frequencies(d) for d in docs])
        self.mean_ = freqs.mean(axis=0)
        std = freqs.std(axis=0, ddof=1) if len(docs) > 1 else np.zeros(len(self.vocabulary))
        std = np.where(std > 1e-12, std, 1.0)
        self.std_ = std

        z = self._zscores(freqs)
        labels_arr = np.array(norm_labels)
        self.human_profile_ = z[labels_arr == _HUMAN_LABEL].mean(axis=0)
        self.ai_profile_ = z[labels_arr == _AI_LABEL].mean(axis=0)

        # Calibrate the logistic scale on the training distances so the
        # probability is not saturated (or flat) regardless of method.
        diffs = np.array([
            self._distance(row, self.human_profile_) - self._distance(row, self.ai_profile_)
            for row in z
        ])
        spread = float(np.std(diffs))
        self.scale_ = spread if spread > 1e-12 else 1.0
        return self

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------
    def _neutral(self) -> Dict:
        return {
            "delta_human": 0.0,
            "delta_ai": 0.0,
            "ai_probability": 0.5,
            "per_word_diff": {},
        }

    def score(self, tokens: Sequence[str]) -> Dict:
        """
        Score one tokenised text.

        Returns a dict with:
            delta_human    : distance to the human profile
            delta_ai       : distance to the AI profile
            ai_probability : P(AI) in [0, 1] (0.5 when undecidable)
            per_word_diff  : {word: |z - human| - |z - ai|}; positive values
                             mean the word pulls the text toward the AI profile
        """
        if not self.is_fitted:
            return self._neutral()
        freqs = self._relative_frequencies(tokens)
        if freqs is None:
            return self._neutral()

        z = self._zscores(freqs)
        delta_human = self._distance(z, self.human_profile_)
        delta_ai = self._distance(z, self.ai_profile_)
        ai_probability = _sigmoid((delta_human - delta_ai) / self.scale_)

        word_diff = np.abs(z - self.human_profile_) - np.abs(z - self.ai_profile_)
        per_word_diff = {w: float(d) for w, d in zip(self.vocabulary, word_diff)}

        return {
            "delta_human": float(delta_human),
            "delta_ai": float(delta_ai),
            "ai_probability": float(np.clip(ai_probability, 0.0, 1.0)),
            "per_word_diff": per_word_diff,
        }

    def delta(self, tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
        """Delta distance between two texts in the fitted z-score space."""
        if not self.is_fitted:
            raise RuntimeError("BurrowsDeltaModel must be fitted before computing delta")
        fa = self._relative_frequencies(tokens_a)
        fb = self._relative_frequencies(tokens_b)
        if fa is None or fb is None:
            return 0.0 if fa is None and fb is None else float("nan")
        return self._distance(self._zscores(fa), self._zscores(fb))
