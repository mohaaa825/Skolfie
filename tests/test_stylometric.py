"""Unit tests for skolfie.features.stylometric.

These tests must pass both with NLTK data installed (CI downloads it) and
without it (the module falls back to regex tokenization), so they only
assert on values that are identical in both modes.
"""

import math

import numpy as np
import pytest

from skolfie.features import stylometric
from skolfie.features.stylometric import (
    FUNCTION_WORDS,
    FeatureExtractor,
    features_to_vector,
)

SHORT_TEXTS = [
    "The cat sat on mats.",
    "Honestly? I have no idea what happened, but it was fun!",
    "Artificial intelligence is transforming industries. It offers many benefits; "
    "however, it also raises important ethical questions.",
    "one",
    "!!! ??? ...",
]

CORE_FEATURES = {
    "word_count",
    "sentence_count",
    "char_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_std",
    "sentence_length_burstiness",
    "max_sentence_length",
    "min_sentence_length",
    "type_token_ratio",
    "hapax_legomena_ratio",
    "yules_k",
    "punct_density",
    "comma_ratio",
    "period_ratio",
    "exclaim_ratio",
    "question_ratio",
    "semicolon_ratio",
    "colon_ratio",
    "dash_ratio",
    "quote_ratio",
    "avg_syllables_per_word",
    "flesch_reading_ease",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pronoun_ratio",
    "stopword_ratio",
    "repeated_bigram_ratio",
}


def _expected_names() -> set[str]:
    return CORE_FEATURES | {f"fw_{w}" for w in FUNCTION_WORDS}


def _assert_all_finite(features: dict[str, float]) -> None:
    bad = {k: v for k, v in features.items() if not math.isfinite(v)}
    assert not bad, f"non-finite feature values: {bad}"


# ---------------------------------------------------------------------------
# Feature-name set
# ---------------------------------------------------------------------------
def test_feature_names_match_expected_set(extractor):
    names = set(extractor.extract_features(SHORT_TEXTS[0]))
    assert names == _expected_names()


def test_feature_names_stable_across_texts(extractor):
    key_lists = [list(extractor.extract_features(t)) for t in SHORT_TEXTS]
    for keys in key_lists[1:]:
        assert keys == key_lists[0], "feature names or their order differ between texts"


def test_feature_count_is_about_130(extractor):
    n = len(extractor.extract_features(SHORT_TEXTS[0]))
    assert n == len(CORE_FEATURES) + len(FUNCTION_WORDS)
    assert 100 <= n <= 160


def test_feature_names_same_with_regex_fallback(extractor, monkeypatch):
    """Missing NLTK data must not change the feature schema."""
    with_default = list(extractor.extract_features(SHORT_TEXTS[2]))
    monkeypatch.setattr(stylometric, "_NLTK_READY", False)
    with_fallback = list(extractor.extract_features(SHORT_TEXTS[2]))
    assert with_fallback == with_default


def test_custom_function_words_change_fw_features():
    fe = FeatureExtractor(function_words=["cat", "mats"])
    feats = fe.extract_features("The cat sat on mats.")
    fw_keys = {k for k in feats if k.startswith("fw_")}
    assert fw_keys == {"fw_cat", "fw_mats"}
    assert feats["fw_cat"] == pytest.approx(1 / 5)


# ---------------------------------------------------------------------------
# Value sanity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", SHORT_TEXTS)
def test_all_values_finite_floats(extractor, text):
    feats = extractor.extract_features(text)
    assert all(isinstance(v, float) for v in feats.values())
    _assert_all_finite(feats)


def test_seed_texts_all_finite(extractor, seed_texts):
    for label, texts in seed_texts.items():
        for text in texts:
            feats = extractor.extract_features(text)
            assert set(feats) == _expected_names(), label
            _assert_all_finite(feats)


@pytest.mark.parametrize("text", ["", "   \n\t ", None])
def test_empty_input_does_not_crash(extractor, text):
    # Current behaviour: empty / whitespace / None input returns the full
    # feature dict with zero counts rather than raising.
    feats = extractor.extract_features(text)
    assert set(feats) == _expected_names()
    _assert_all_finite(feats)
    assert feats["word_count"] == 0.0
    assert feats["type_token_ratio"] == 0.0
    assert feats["flesch_reading_ease"] == 0.0
    assert all(feats[f"fw_{w}"] == 0.0 for w in FUNCTION_WORDS)


def test_ratios_are_bounded(extractor, seed_texts):
    bounded = [
        "type_token_ratio",
        "hapax_legomena_ratio",
        "punct_density",
        "stopword_ratio",
        "repeated_bigram_ratio",
        "noun_ratio",
        "verb_ratio",
        "adj_ratio",
        "adv_ratio",
        "pronoun_ratio",
    ]
    for text in seed_texts["human"][:5] + seed_texts["ai"][:5]:
        feats = extractor.extract_features(text)
        for name in bounded:
            assert 0.0 <= feats[name] <= 1.0, (name, feats[name])


# ---------------------------------------------------------------------------
# Known simple values
# ---------------------------------------------------------------------------
def test_known_values_five_word_sentence(extractor):
    text = "The cat sat on mats."
    feats = extractor.extract_features(text)
    assert feats["word_count"] == 5.0
    assert feats["sentence_count"] == 1.0
    assert feats["char_count"] == float(len(text))
    assert feats["avg_word_length"] == pytest.approx((3 + 3 + 3 + 2 + 4) / 5)
    assert feats["avg_sentence_length"] == pytest.approx(5.0)
    assert feats["type_token_ratio"] == pytest.approx(1.0)
    assert feats["hapax_legomena_ratio"] == pytest.approx(1.0)
    assert feats["period_ratio"] == pytest.approx(1 / 5)
    assert feats["comma_ratio"] == 0.0
    assert feats["fw_the"] == pytest.approx(1 / 5)
    assert feats["fw_on"] == pytest.approx(1 / 5)
    assert feats["stopword_ratio"] == pytest.approx(2 / 5)  # "the", "on"
    assert feats["repeated_bigram_ratio"] == 0.0


def test_repetition_lowers_type_token_ratio(extractor):
    feats = extractor.extract_features("the the the the")
    assert feats["word_count"] == 4.0
    assert feats["type_token_ratio"] == pytest.approx(0.25)
    assert feats["hapax_legomena_ratio"] == 0.0
    assert feats["fw_the"] == pytest.approx(1.0)
    # 3 bigrams, all ("the", "the"): one distinct bigram that repeats.
    assert feats["repeated_bigram_ratio"] == pytest.approx(1 / 3)


def test_sentence_count_and_punctuation(extractor):
    feats = extractor.extract_features("Is it raining? Yes! Take a coat.")
    assert feats["sentence_count"] == 3.0
    assert feats["question_ratio"] == pytest.approx(1 / 7)
    assert feats["exclaim_ratio"] == pytest.approx(1 / 7)


def test_extraction_is_deterministic(extractor):
    text = SHORT_TEXTS[2]
    assert extractor.extract_features(text) == extractor.extract_features(text)


# ---------------------------------------------------------------------------
# features_to_vector
# ---------------------------------------------------------------------------
def test_features_to_vector_follows_given_order(extractor):
    feats = extractor.extract_features(SHORT_TEXTS[2])
    order = sorted(feats)
    vec = features_to_vector(feats, order)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == float
    assert vec.shape == (len(order),)
    for i, name in enumerate(order):
        assert vec[i] == feats[name]


def test_features_to_vector_consistent_across_texts(extractor):
    order = list(extractor.extract_features(SHORT_TEXTS[0]))
    vecs = [features_to_vector(extractor.extract_features(t), order) for t in SHORT_TEXTS]
    assert {v.shape for v in vecs} == {(len(order),)}
    idx = order.index("word_count")
    assert vecs[0][idx] == 5.0


def test_features_to_vector_missing_keys_are_zero():
    vec = features_to_vector({"a": 1.5}, ["b", "a", "c"])
    np.testing.assert_array_equal(vec, np.array([0.0, 1.5, 0.0]))


def test_features_to_vector_reordering_permutes_values(extractor):
    feats = extractor.extract_features(SHORT_TEXTS[1])
    order = list(feats)
    forward = features_to_vector(feats, order)
    backward = features_to_vector(feats, order[::-1])
    np.testing.assert_array_equal(forward, backward[::-1])
