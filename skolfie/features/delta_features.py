"""
skolfie/features/delta_features.py
-----------------------------------
Burrows-style Delta distances as features for a classifier, vectorised so
they scale to a few hundred thousand texts.

`burrows_delta.BurrowsDeltaModel` scores one token list at a time and was
fitted on the same texts the old ensemble trained on, so its score leaked
the training labels. Here the Delta step is a featurizer with an explicit
fit/transform split: `FeatureModel` fits it out-of-fold on the training
split and refits it on the whole split only for scoring new texts.

How it works (same maths as `BurrowsDeltaModel`):

    1. The `n_words` most frequent words (MFW) of the training texts.
    2. Per text: relative frequency of each MFW (count / all words).
    3. Z-scores with the training mean and standard deviation per word.
    4. Profiles: the mean z-vector of the human and of the AI texts.
    5. Features per text:
         delta_cos_human  Cosine Delta (1 - cosine similarity) to the human profile
         delta_cos_ai     Cosine Delta to the AI profile
         delta_cos_diff   delta_cos_human - delta_cos_ai (> 0: closer to AI)
         delta_bur_diff   the same difference with classic Burrows' Delta
                          (mean absolute z-score difference)

Texts with no words at all get NaN features (tree models handle NaN).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from skolfie.eval.metrics import to_binary

DELTA_FEATURES = ("delta_cos_human", "delta_cos_ai", "delta_cos_diff", "delta_bur_diff")
FEATURE_DESCRIPTIONS = {
    "delta_cos_human": "How far the text's common-word profile is from typical human text "
                       "(Cosine Delta).",
    "delta_cos_ai": "How far the text's common-word profile is from typical AI text "
                    "(Cosine Delta).",
    "delta_cos_diff": "Cosine Delta to human text minus to AI text; positive means the text "
                      "uses common words more like AI text.",
    "delta_bur_diff": "Burrows' Delta to human text minus to AI text; positive means the text "
                      "uses common words more like AI text.",
}

# Words: letters, optionally with one inner apostrophe ("don't"); no digits.
WORD_PATTERN = r"(?u)\b[^\W\d_]+(?:'[^\W\d_]+)?\b"
_WORD_RE = re.compile(WORD_PATTERN)


class DeltaFeaturizer:
    """Fit MFW statistics and class profiles on training texts; emit Delta distances."""

    def __init__(self, n_words: int = 300, chunk_size: int = 20_000) -> None:
        self.n_words = n_words
        self.chunk_size = chunk_size
        self.vectorizer: Any = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.human_profile_: np.ndarray | None = None
        self.ai_profile_: np.ndarray | None = None

    @property
    def vocabulary(self) -> list[str]:
        if self.vectorizer is None:
            return []
        return list(self.vectorizer.get_feature_names_out())

    def _rel_freqs(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(relative MFW frequencies float64 (n, k), has_words bool (n,))."""
        counts = self.vectorizer.transform(texts).toarray().astype(np.float64)
        totals = np.array([len(_WORD_RE.findall(t.lower())) for t in texts], dtype=np.float64)
        has_words = totals > 0
        counts[has_words] /= totals[has_words, None]
        return counts, has_words

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "DeltaFeaturizer":
        from sklearn.feature_extraction.text import CountVectorizer

        y = to_binary(labels)
        texts = list(texts)
        if len(texts) != len(y):
            raise ValueError("texts and labels must have the same length")
        self.vectorizer = CountVectorizer(token_pattern=WORD_PATTERN, lowercase=True,
                                          max_features=self.n_words, dtype=np.int32)
        try:
            self.vectorizer.fit(texts)
        except ValueError as exc:  # empty vocabulary
            raise ValueError("no words found in the training texts") from exc

        k = len(self.vectorizer.vocabulary_)
        n = 0
        total = np.zeros(k)
        total_sq = np.zeros(k)
        class_sum = {0: np.zeros(k), 1: np.zeros(k)}
        class_n = {0: 0, 1: 0}
        freqs_by_chunk = []
        for start in range(0, len(texts), self.chunk_size):
            freqs, has_words = self._rel_freqs(texts[start:start + self.chunk_size])
            freqs, yc = freqs[has_words], y[start:start + self.chunk_size][has_words]
            n += len(freqs)
            total += freqs.sum(axis=0)
            total_sq += (freqs ** 2).sum(axis=0)
            freqs_by_chunk.append((freqs.astype(np.float32), yc))
        if n == 0:
            raise ValueError("no words found in the training texts")
        self.mean_ = total / n
        var = (total_sq / n - self.mean_ ** 2) * (n / max(n - 1, 1))
        std = np.sqrt(np.clip(var, 0.0, None))
        self.std_ = np.where(std > 1e-12, std, 1.0)

        for freqs, yc in freqs_by_chunk:
            z = (freqs - self.mean_) / self.std_
            for cls in (0, 1):
                class_sum[cls] += z[yc == cls].sum(axis=0)
                class_n[cls] += int((yc == cls).sum())
        if not class_n[0] or not class_n[1]:
            raise ValueError("training data must contain non-empty human and ai texts")
        self.human_profile_ = class_sum[0] / class_n[0]
        self.ai_profile_ = class_sum[1] / class_n[1]
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """(n, len(DELTA_FEATURES)) float32."""
        if self.vectorizer is None:
            raise RuntimeError("DeltaFeaturizer is not fitted")
        texts = list(texts)
        out = np.full((len(texts), len(DELTA_FEATURES)), np.nan, dtype=np.float32)
        for start in range(0, len(texts), self.chunk_size):
            freqs, has_words = self._rel_freqs(texts[start:start + self.chunk_size])
            z = (freqs - self.mean_) / self.std_
            cos_h = _cosine_distance(z, self.human_profile_)
            cos_a = _cosine_distance(z, self.ai_profile_)
            bur = (np.abs(z - self.human_profile_).mean(axis=1)
                   - np.abs(z - self.ai_profile_).mean(axis=1))
            block = np.column_stack([cos_h, cos_a, cos_h - cos_a, bur])
            block[~has_words] = np.nan
            out[start:start + len(block)] = block
        return out


def _cosine_distance(z: np.ndarray, profile: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1) * np.linalg.norm(profile)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (z @ profile) / norms
    return np.where(norms > 0, 1.0 - sim, 1.0)
