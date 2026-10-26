"""
skolfie/features/stylometric.py
--------------------------------
Extracts a rich set of stylometric / linguistic features from a text.

Feature groups:
    1. Basic counts & ratios        (word count, sentence count, avg word length...)
    2. Sentence structure           (mean/std sentence length, burstiness...)
    3. Vocabulary richness          (TTR, hapax legomena ratio, Yule's K...)
    4. Punctuation patterns         (comma/period/exclaim/question density...)
    5. Readability                  (avg syllables per word, Flesch reading ease)
    6. Part-of-speech distribution  (noun/verb/adj/adv ratios)
    7. Function-word frequencies    (also reused by Burrows' Delta)

The extractor is deliberately "flat": `extract_features` returns a plain
dict of {feature_name: float}. That makes it trivial to turn into a numpy
vector for the ML models, and easy to show to users as "top contributing
features" without any extra translation layer.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------
# NLTK setup
# --------------------------------------------------------------------------
# We prefer NLTK for tokenization/POS tagging because it's far more accurate
# than naive regex splitting. If the required NLTK data isn't available
# (e.g. no internet on first run), we quietly fall back to regex-based
# approximations so the app never crashes because of it.
# --------------------------------------------------------------------------
_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def download_nltk_data() -> None:
    """Download the NLTK resources this module uses. Run once at setup time;
    importing the module never touches the network."""
    import nltk

    for path, pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_NLTK_READY = False
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag

    word_tokenize("Smoke test.")  # confirm data actually loaded
    _NLTK_READY = True
except Exception:
    _NLTK_READY = False


# Common English function words, used both as extra stylistic signal here
# and as the vocabulary for Burrows' Delta (see burrows_delta.py). Kept as
# a local constant (rather than nltk.corpus.stopwords) so the app still
# works even if NLTK corpora aren't installed.
FUNCTION_WORDS = [
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it",
    "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
    "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
    "there", "use", "an", "each", "which", "she", "do", "how", "their", "if",
    "will", "up", "other", "about", "out", "many", "then", "them", "these", "so",
    "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
    "two", "more", "write", "go", "see", "number", "no", "way", "could", "people",
    "my", "than", "first", "been", "call", "who", "its", "now", "find", "long",
    "down", "day", "did", "get", "come", "made", "may", "part",
]


def _simple_sent_tokenize(text: str) -> List[str]:
    """Regex fallback sentence splitter (used only if NLTK data is missing)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _simple_word_tokenize(text: str) -> List[str]:
    """Regex fallback word tokenizer (used only if NLTK data is missing)."""
    return re.findall(r"[A-Za-z']+", text)


def tokenize_sentences(text: str) -> List[str]:
    if _NLTK_READY:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    return _simple_sent_tokenize(text)


def tokenize_words(text: str) -> List[str]:
    if _NLTK_READY:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return _simple_word_tokenize(text)


def _syllable_count(word: str) -> int:
    """Crude heuristic syllable counter (vowel-group counting)."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _yules_k(word_counts: Counter, n_words: int) -> float:
    """
    Yule's K measure of vocabulary richness (lower = richer/more varied
    vocabulary). A classic stylometric measure that's fairly robust to
    differences in text length.
    """
    if n_words == 0:
        return 0.0
    freq_of_freq = Counter(word_counts.values())
    m1 = n_words
    m2 = sum((freq ** 2) * count for freq, count in freq_of_freq.items())
    if m1 == 0:
        return 0.0
    return 10000.0 * (m2 - m1) / (m1 * m1)


class FeatureExtractor:
    """Extracts a dictionary of stylometric features from raw text."""

    def __init__(self, function_words: List[str] = None):
        self.function_words = function_words or FUNCTION_WORDS

    # ---------------------------------------------------------------
    def extract_features(self, text: str) -> Dict[str, float]:
        text = text or ""
        sentences = tokenize_sentences(text)
        words_raw = tokenize_words(text)
        words = [w for w in words_raw if any(c.isalpha() for c in w)]
        words_lower = [w.lower() for w in words]

        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        n_chars = len(text)

        features: Dict[str, float] = {}

        # ---- 1. Basic counts & ratios ---------------------------------
        features["word_count"] = float(n_words)
        features["sentence_count"] = float(len(sentences))
        features["char_count"] = float(n_chars)
        features["avg_word_length"] = (
            sum(len(w) for w in words) / n_words if n_words else 0.0
        )
        features["avg_sentence_length"] = n_words / n_sentences

        # ---- 2. Sentence structure --------------------------------------
        sent_lengths = [len(tokenize_words(s)) for s in sentences] or [0]
        features["sentence_length_std"] = float(np.std(sent_lengths))
        mean_sl = float(np.mean(sent_lengths)) if sent_lengths else 0.0
        features["sentence_length_burstiness"] = (
            features["sentence_length_std"] / mean_sl if mean_sl else 0.0
        )
        features["max_sentence_length"] = float(max(sent_lengths))
        features["min_sentence_length"] = float(min(sent_lengths))

        # ---- 3. Vocabulary richness -------------------------------------
        unique_words = set(words_lower)
        word_counts = Counter(words_lower)
        features["type_token_ratio"] = (
            len(unique_words) / n_words if n_words else 0.0
        )
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        features["hapax_legomena_ratio"] = hapax / n_words if n_words else 0.0
        features["yules_k"] = _yules_k(word_counts, n_words)

        # ---- 4. Punctuation patterns ------------------------------------
        punct_counts = Counter(ch for ch in text if ch in string.punctuation)
        total_punct = sum(punct_counts.values())
        features["punct_density"] = total_punct / n_chars if n_chars else 0.0
        for p, name in [
            (",", "comma"), (".", "period"), ("!", "exclaim"),
            ("?", "question"), (";", "semicolon"), (":", "colon"),
            ("-", "dash"), ('"', "quote"),
        ]:
            features[f"{name}_ratio"] = (
                punct_counts.get(p, 0) / n_words if n_words else 0.0
            )

        # ---- 5. Readability ------------------------------------------
        syllables = sum(_syllable_count(w) for w in words)
        features["avg_syllables_per_word"] = syllables / n_words if n_words else 0.0
        if n_words and n_sentences:
            features["flesch_reading_ease"] = (
                206.835
                - 1.015 * (n_words / n_sentences)
                - 84.6 * (syllables / n_words)
            )
        else:
            features["flesch_reading_ease"] = 0.0

        # ---- 6. Part-of-speech distribution -----------------------------
        features.update(self._pos_distribution(words))

        # ---- 7. Function word frequencies (also used by Burrows' Delta) -
        for fw in self.function_words:
            features[f"fw_{fw}"] = word_counts.get(fw, 0) / n_words if n_words else 0.0

        # ---- 8. Stopword ratio & repetition ------------------------------
        stop_set = set(self.function_words)
        stopword_hits = sum(1 for w in words_lower if w in stop_set)
        features["stopword_ratio"] = stopword_hits / n_words if n_words else 0.0

        # Repeated-bigram ratio: a rough proxy for formulaic / repetitive
        # phrasing.
        bigrams = list(zip(words_lower, words_lower[1:]))
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
        features["repeated_bigram_ratio"] = (
            repeated_bigrams / len(bigrams) if bigrams else 0.0
        )

        return features

    # ---------------------------------------------------------------
    def _pos_distribution(self, words: List[str]) -> Dict[str, float]:
        """Return proportion of noun/verb/adj/adv/pronoun tags among words."""
        result = {
            "noun_ratio": 0.0, "verb_ratio": 0.0,
            "adj_ratio": 0.0, "adv_ratio": 0.0,
            "pronoun_ratio": 0.0,
        }
        if not words:
            return result

        if _NLTK_READY:
            try:
                tags = pos_tag(words)
                total = len(tags)
                counts = Counter(tag[:2] for _, tag in tags)
                result["noun_ratio"] = counts.get("NN", 0) / total
                result["verb_ratio"] = counts.get("VB", 0) / total
                result["adj_ratio"] = counts.get("JJ", 0) / total
                result["adv_ratio"] = counts.get("RB", 0) / total
                result["pronoun_ratio"] = counts.get("PR", 0) / total
                return result
            except Exception:
                pass

        # Fallback heuristic (crude suffix rules) if POS tagging unavailable.
        total = len(words)
        nouns = sum(1 for w in words if w.lower().endswith(("tion", "ness", "ity", "ment")))
        verbs = sum(1 for w in words if w.lower().endswith(("ing", "ed")))
        adjs = sum(1 for w in words if w.lower().endswith(("ous", "ive", "able", "ful")))
        advs = sum(1 for w in words if w.lower().endswith("ly"))
        result["noun_ratio"] = nouns / total
        result["verb_ratio"] = verbs / total
        result["adj_ratio"] = adjs / total
        result["adv_ratio"] = advs / total
        return result


def features_to_vector(features: Dict[str, float], feature_order: List[str]) -> np.ndarray:
    """Convert a feature dict into a fixed-order numpy vector (0.0 for missing keys)."""
    return np.array([features.get(k, 0.0) for k in feature_order], dtype=float)
