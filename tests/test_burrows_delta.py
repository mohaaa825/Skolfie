"""
test_burrows_delta.py
----------------------
Unit tests for the from-scratch Burrows' Delta / Cosine Delta model.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from skolfie.features.burrows_delta import BurrowsDeltaModel
from skolfie.features.stylometric import FUNCTION_WORDS, tokenize_words
from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

TEXTS = HUMAN_TEXTS + AI_TEXTS
LABELS = ["human"] * len(HUMAN_TEXTS) + ["ai"] * len(AI_TEXTS)
TOKENS = [tokenize_words(t) for t in TEXTS]


@pytest.fixture(scope="module", params=["burrows", "cosine"])
def fitted(request) -> BurrowsDeltaModel:
    model = BurrowsDeltaModel(method=request.param)
    model.fit(TOKENS, LABELS, n_words=60, candidate_vocab=FUNCTION_WORDS)
    return model


def test_default_method_is_burrows():
    assert BurrowsDeltaModel().method == "burrows"


def test_invalid_method_rejected():
    with pytest.raises(ValueError):
        BurrowsDeltaModel(method="manhattan")


def test_unfitted_model_has_empty_vocabulary_and_neutral_score():
    model = BurrowsDeltaModel()
    assert not model.vocabulary
    assert model.score(["hello", "world"])["ai_probability"] == 0.5


def test_fit_builds_vocabulary(fitted):
    assert 0 < len(fitted.vocabulary) <= 60
    assert set(fitted.vocabulary) <= set(FUNCTION_WORDS)
    assert fitted.std_.shape == (len(fitted.vocabulary),)
    assert np.all(fitted.std_ > 0)


def test_score_result_shape(fitted):
    result = fitted.score(TOKENS[0])
    assert set(result) >= {"delta_human", "delta_ai", "ai_probability", "per_word_diff"}
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert set(result["per_word_diff"]) == set(fitted.vocabulary)


def test_training_texts_lean_to_their_class(fitted):
    human_p = np.mean([fitted.score(t)["ai_probability"] for t in TOKENS[: len(HUMAN_TEXTS)]])
    ai_p = np.mean([fitted.score(t)["ai_probability"] for t in TOKENS[len(HUMAN_TEXTS):]])
    assert ai_p > human_p


def test_identical_text_has_near_zero_delta(fitted):
    assert fitted.delta(TOKENS[3], TOKENS[3]) == pytest.approx(0.0, abs=1e-9)
    assert fitted.delta(TOKENS[3], TOKENS[-1]) > 0.0


def test_text_equal_to_profile_has_zero_delta():
    # With one document per class, each profile IS that document's z-vector.
    model = BurrowsDeltaModel()
    model.fit([TOKENS[0], TOKENS[-1]], ["human", "ai"], n_words=30)
    result = model.score(TOKENS[0])
    assert result["delta_human"] == pytest.approx(0.0, abs=1e-9)
    assert result["delta_ai"] > 0.0
    assert result["ai_probability"] < 0.5


@pytest.mark.parametrize("tokens", [[], ["", "...", "!"], ["42"]])
def test_empty_or_punctuation_only_text_is_neutral(fitted, tokens):
    result = fitted.score(tokens)
    assert result["ai_probability"] == 0.5
    assert result["per_word_diff"] == {}


def test_short_text_does_not_crash(fitted):
    result = fitted.score(["the"])
    assert 0.0 <= result["ai_probability"] <= 1.0
    assert np.isfinite(result["delta_human"]) and np.isfinite(result["delta_ai"])


def test_zero_std_word_does_not_divide_by_zero():
    # "the" appears at exactly the same rate in every doc -> std == 0.
    docs = [["the", "cat"], ["the", "dog"], ["the", "cat"], ["the", "fox"]]
    model = BurrowsDeltaModel().fit(docs, ["human", "human", "ai", "ai"], n_words=10)
    assert np.all(np.isfinite(model.std_)) and np.all(model.std_ > 0)
    result = model.score(["the", "bird"])
    assert np.isfinite(result["delta_human"]) and np.isfinite(result["ai_probability"])


def test_fit_requires_both_classes():
    with pytest.raises(ValueError):
        BurrowsDeltaModel().fit(TOKENS[:3], ["human"] * 3)


def test_methods_give_different_distances():
    burrows = BurrowsDeltaModel("burrows").fit(TOKENS, LABELS, n_words=60)
    cosine = BurrowsDeltaModel("cosine").fit(TOKENS, LABELS, n_words=60)
    rb, rc = burrows.score(TOKENS[5]), cosine.score(TOKENS[5])
    assert rb["delta_human"] != pytest.approx(rc["delta_human"])
    assert 0.0 <= rc["delta_human"] <= 2.0


def test_pickle_round_trip(fitted, tmp_path):
    path = tmp_path / "burrows.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    assert loaded.method == fitted.method
    assert loaded.vocabulary == fitted.vocabulary
    assert loaded.score(TOKENS[7]) == fitted.score(TOKENS[7])
