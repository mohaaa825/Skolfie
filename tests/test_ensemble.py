"""
test_ensemble.py
-----------------
End-to-end tests for AIDetectorModel: train on the synthetic seed texts,
predict, and save/load round-trip.
"""

from __future__ import annotations

import pytest

from skolfie.models.ensemble import AIDetectorModel
from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

EXPECTED_KEYS = {
    "label", "ai_probability", "human_probability", "confidence",
    "ml_ai_probability", "burrows_ai_probability", "burrows_delta_human",
    "burrows_delta_ai", "burrows_top_words", "top_features", "raw_features",
}


@pytest.fixture(scope="module")
def trained_model() -> AIDetectorModel:
    model = AIDetectorModel()
    texts = HUMAN_TEXTS + AI_TEXTS
    labels = ["human"] * len(HUMAN_TEXTS) + ["ai"] * len(AI_TEXTS)
    stats = model.train(texts, labels, n_estimators=20)
    assert stats["n_samples"] == len(texts)
    assert 0.0 <= stats["train_accuracy"] <= 1.0
    return model


def test_predict_before_training_raises():
    with pytest.raises(RuntimeError):
        AIDetectorModel().predict("Some text.")


def test_burrows_component_is_fitted(trained_model):
    assert trained_model.burrows_model.vocabulary
    assert trained_model.is_trained


def test_predict_result_shape(trained_model):
    result = trained_model.predict(AI_TEXTS[0])
    assert EXPECTED_KEYS <= set(result)
    assert result["label"] in {"AI-Generated", "Human-Written"}
    assert result["confidence"] in {"High", "Medium", "Low"}
    for key in ("ai_probability", "human_probability", "ml_ai_probability",
                "burrows_ai_probability"):
        assert 0.0 <= result[key] <= 1.0, key
    assert result["ai_probability"] + result["human_probability"] == pytest.approx(1.0)
    assert len(result["burrows_top_words"]) <= 5
    assert isinstance(result["top_features"], list)


def test_predict_short_text_does_not_crash(trained_model):
    result = trained_model.predict("Hi.")
    assert 0.0 <= result["ai_probability"] <= 1.0


def test_save_load_round_trip(trained_model, tmp_path):
    path = tmp_path / "model.joblib"
    trained_model.save(str(path))
    loaded = AIDetectorModel.load(str(path))
    text = HUMAN_TEXTS[0]
    original, restored = trained_model.predict(text), loaded.predict(text)
    assert restored["ai_probability"] == pytest.approx(original["ai_probability"])
    assert restored["label"] == original["label"]
