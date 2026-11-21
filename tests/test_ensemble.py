"""
test_ensemble.py
-----------------
End-to-end tests for AIDetectorModel: train on the synthetic seed texts,
predict, and save/load round-trip; the out-of-fold Burrows' Delta rows that
keep training labels from leaking into the ML features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skolfie.models.ensemble import AIDetectorModel
from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

EXPECTED_KEYS = {
    "label", "ai_probability", "human_probability", "ml_ai_probability",
    "burrows_ai_probability", "burrows_delta_human", "burrows_delta_ai", "obfuscation",
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
    for key in ("ai_probability", "human_probability", "ml_ai_probability",
                "burrows_ai_probability"):
        assert 0.0 <= result[key] <= 1.0, key
    assert result["ai_probability"] + result["human_probability"] == pytest.approx(1.0)


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


def test_predict_undoes_unicode_tricks(trained_model):
    from skolfie.robust.attacks import homoglyph_text, text_rng, zero_width_text

    text = AI_TEXTS[0]
    attacked = zero_width_text(homoglyph_text(text, 0.3, text_rng(0, "h", text)), 0.3,
                               text_rng(0, "z", text))
    assert attacked != text
    clean, result = trained_model.predict(text), trained_model.predict(attacked)
    assert result["ai_probability"] == pytest.approx(clean["ai_probability"])
    assert clean["obfuscation"] == {"invisible": 0, "lookalikes": 0}
    assert result["obfuscation"]["invisible"] > 0 and result["obfuscation"]["lookalikes"] > 0


def test_burrows_training_features_do_not_leak_labels() -> None:
    from sklearn.metrics import roc_auc_score

    from skolfie.features.stylometric import tokenize_words

    texts = HUMAN_TEXTS + AI_TEXTS
    tokens = [tokenize_words(t) for t in texts]
    rng = np.random.default_rng(0)
    aucs = []
    for seed in range(3):
        noise = list(rng.permutation(["human"] * 25 + ["ai"] * 25))
        rows = AIDetectorModel._oof_burrows_rows(tokens, noise, random_state=seed)
        aucs.append(roc_auc_score([lab == "ai" for lab in noise], rows[:, 2]))
    # In-sample scoring gave AUROC ~0.85 on pure-noise labels.
    assert np.mean(aucs) < 0.7

    real = ["human"] * 25 + ["ai"] * 25
    rows = AIDetectorModel._oof_burrows_rows(tokens, real)
    assert roc_auc_score([lab == "ai" for lab in real], rows[:, 2]) > 0.9


def test_oof_rows_neutral_with_tiny_class() -> None:
    rows = AIDetectorModel._oof_burrows_rows([["the", "a"]] * 3, ["human", "human", "ai"])
    assert rows.tolist() == [[0.0, 0.0, 0.5]] * 3


def test_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skolfie.models import ensemble

    path = tmp_path / "model.joblib"
    path.write_bytes(b"previous good model")

    def broken_dump(obj, target):
        Path(target).write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(ensemble.joblib, "dump", broken_dump)
    with pytest.raises(KeyboardInterrupt):
        AIDetectorModel().save(str(path))
    assert path.read_bytes() == b"previous good model"
    assert [p.name for p in tmp_path.iterdir()] == ["model.joblib"]
