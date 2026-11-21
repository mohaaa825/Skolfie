"""
tests/test_train_script.py
---------------------------
Smoke test for scripts/train.py (trains on the synthetic seed fixture).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import load_script

pytest.importorskip("skolfie.features.burrows_delta")



@pytest.fixture(scope="module")
def train():
    return load_script("train")


def test_seed_demo_writes_model(train, tmp_path: Path) -> None:
    out = tmp_path / "models" / "demo.joblib"
    assert train.main(["--seed-demo", "--out", str(out), "--test-size", "0.2"]) == 0
    assert out.exists()

    from skolfie.models.ensemble import AIDetectorModel

    model = AIDetectorModel.load(str(out))
    result = model.predict("This is a short text that we just want a prediction for, okay?")
    assert 0.0 <= result["ai_probability"] <= 1.0


def test_data_and_feedback_inputs(train, tmp_path: Path) -> None:
    from skolfie.data.feedback import add_manual_label, save_feedback
    from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

    data = tmp_path / "data.parquet"
    pd.DataFrame({
        "text": HUMAN_TEXTS[:10] + AI_TEXTS[:10],
        "label": ["human"] * 10 + ["AI"] * 10,
    }).to_parquet(data)

    fb = tmp_path / "feedback.csv"
    add_manual_label(str(fb), HUMAN_TEXTS[10], "Human-Written")
    save_feedback(str(fb), AI_TEXTS[10], "AI-Generated", "High", 0.9, thumbs="up")
    save_feedback(str(fb), AI_TEXTS[11], "Human-Written", "Low", 0.4, thumbs="down")

    texts, labels = train.load_feedback_data(fb)
    assert labels == ["human", "ai"]  # thumbs-down without a correction is skipped

    out = tmp_path / "model.joblib"
    assert train.main(["--data", str(data), "--feedback", str(fb), "--out", str(out)]) == 0
    assert out.exists()


def test_requires_some_input(train) -> None:
    with pytest.raises(SystemExit):
        train.main([])


def test_rejects_single_class(train, tmp_path: Path) -> None:
    data = tmp_path / "one_class.csv"
    pd.DataFrame({"text": ["only human text here"], "label": ["human"]}).to_csv(data, index=False)
    assert train.main(["--data", str(data), "--out", str(tmp_path / "m.joblib")]) == 2


def test_drop_duplicate_texts_last_wins(train) -> None:
    texts, labels, dropped = train.drop_duplicate_texts(
        ["a", "b", "a", "c"], ["human", "ai", "ai", "human"])
    assert (texts, labels, dropped) == (["b", "a", "c"], ["ai", "ai", "human"], 1)


def test_rejects_bad_paths_before_training(train, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train.main(["--data", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit):
        train.main(["--feedback", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit):
        train.main(["--seed-demo", "--out", str(tmp_path)])


def test_test_size_too_small_is_clean_error(train, tmp_path: Path) -> None:
    data = tmp_path / "tiny.csv"
    pd.DataFrame({"text": ["h one", "h two", "a one", "a two"],
                  "label": ["human", "human", "ai", "ai"]}).to_csv(data, index=False)
    out = tmp_path / "m.joblib"
    assert train.main(["--data", str(data), "--test-size", "0.2", "--out", str(out)]) == 2
    assert not out.exists()
