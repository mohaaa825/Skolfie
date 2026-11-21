"""
tests/test_feedback.py
-----------------------
Round-trip of the feedback CSV (save / load / manual label / export), odd and
NA-like texts, concurrent appends and formula escaping on export.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from skolfie.data.feedback import (
    FEEDBACK_COLUMNS,
    add_manual_label,
    export_feedback,
    load_feedback,
    save_feedback,
)
from tests.conftest import load_script


def test_load_creates_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "feedback.csv"
    df = load_feedback(str(csv_path))
    assert csv_path.exists()
    assert df.empty
    assert list(df.columns) == FEEDBACK_COLUMNS


def test_save_load_round_trip(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "feedback.csv")
    text = 'A text with "quotes", commas,\nand a newline.'
    save_feedback(
        csv_path, text=text, predicted_label="AI-Generated", predicted_confidence="High",
        ai_probability=0.912345, user_label="Human-Written", thumbs="down",
        comment="wrong", features={"avg_word_length": 4.5},
    )
    save_feedback(
        csv_path, text="Second.", predicted_label="Human-Written", predicted_confidence="Low",
        ai_probability=0.4, thumbs="up",
    )

    df = load_feedback(csv_path)
    assert len(df) == 2
    first = df.iloc[0]
    assert first["text"] == text
    assert first["predicted_label"] == "AI-Generated"
    assert first["user_label"] == "Human-Written"
    assert first["thumbs"] == "down"
    assert first["comment"] == "wrong"
    assert abs(float(first["ai_probability"]) - 0.9123) < 1e-9
    assert json.loads(first["features"]) == {"avg_word_length": 4.5}
    assert df.iloc[1]["thumbs"] == "up"


def test_manual_label_and_export(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "feedback.csv")
    add_manual_label(csv_path, "Hand-labeled text.", "AI-Generated", comment="curated")
    out = export_feedback(csv_path, str(tmp_path / "export.csv"))

    exported = load_feedback(out)
    assert len(exported) == 1
    assert exported.iloc[0]["user_label"] == "AI-Generated"
    assert float(exported.iloc[0]["ai_probability"]) == 1.0


def test_mixed_label(tmp_path: Path) -> None:
    import pytest

    from skolfie.data.feedback import (
        LABEL_AI,
        LABEL_HUMAN,
        LABEL_MIXED,
        USER_LABELS,
        implied_ai_probability,
    )

    assert USER_LABELS == (LABEL_HUMAN, LABEL_AI, LABEL_MIXED) == (
        "Human-Written", "AI-Generated", "Mixed")
    assert implied_ai_probability(LABEL_AI) == 1.0
    assert implied_ai_probability(LABEL_HUMAN) == 0.0
    assert implied_ai_probability(LABEL_MIXED) is None

    csv_path = str(tmp_path / "feedback.csv")
    add_manual_label(csv_path, "Half mine, half not.", LABEL_MIXED)
    save_feedback(csv_path, "Checked text.", "mixed", "", None, user_label=LABEL_MIXED,
                  thumbs="up")
    df = load_feedback(csv_path)
    assert df["user_label"].tolist() == [LABEL_MIXED, LABEL_MIXED]
    assert df["ai_probability"].isna().all()
    with pytest.raises(ValueError):
        add_manual_label(csv_path, "x", "robot")


def test_binary_training_skips_mixed_rows(tmp_path: Path) -> None:
    train = load_script("train")

    csv_path = str(tmp_path / "feedback.csv")
    add_manual_label(csv_path, "Human text.", "Human-Written")
    add_manual_label(csv_path, "Mixed text.", "Mixed")
    save_feedback(csv_path, "Predicted mixed, thumbs up.", "mixed", "", 0.5, thumbs="up")
    save_feedback(csv_path, "Predicted ai, thumbs up.", "ai", "", 0.9, thumbs="up")
    texts, labels = train.load_feedback_data(csv_path)
    assert list(zip(texts, labels)) == [("Human text.", "human"),
                                        ("Predicted ai, thumbs up.", "ai")]


@pytest.mark.parametrize("text", ["NA", "null", "N/A", "nan", "None", "12345"])
def test_feedback_keeps_na_like_and_numeric_texts(tmp_path: Path, text: str) -> None:
    csv_path = str(tmp_path / "fb.csv")
    save_feedback(csv_path, text, "AI-Generated", "High", 0.9, comment="NA")
    row = load_feedback(csv_path).iloc[0]
    assert row["text"] == text and row["comment"] == "NA"
    assert isinstance(row["text"], str)


def test_feedback_empty_cells_still_missing(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    save_feedback(csv_path, "text", "AI-Generated", "High", 0.9)
    row = load_feedback(csv_path).iloc[0]
    assert pd.isna(row["user_label"]) and pd.isna(row["comment"])
    assert row["ai_probability"] == pytest.approx(0.9)


def test_feedback_odd_characters(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    save_feedback(csv_path, "a\x00b", "AI-Generated", "High", 0.5)
    save_feedback(csv_path, "lone \ud800 surrogate", "AI-Generated", "High", 0.5)
    save_feedback(csv_path, "line1\r\nline2\rline3​‮RTL", "x", "y", 0.5)
    texts = load_feedback(csv_path)["text"].tolist()
    assert texts == ["ab", "lone ? surrogate", "line1\r\nline2\rline3​‮RTL"]


def test_feedback_concurrent_appends(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    n = 16
    barrier = threading.Barrier(n)

    def write(i: int) -> None:
        barrier.wait()
        save_feedback(csv_path, f"text {i} " + "x" * 20_000, "AI-Generated", "High", 0.5)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    df = load_feedback(csv_path)
    assert sorted(int(t.split()[1]) for t in df["text"]) == list(range(n))
    assert list(df.columns) == FEEDBACK_COLUMNS


def test_export_escapes_formulas_but_master_stays_raw(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    payloads = ["=HYPERLINK(\"http://x\")", "+1", "-2", "@SUM(A1)", "\tx", "plain - text"]
    for p in payloads:
        save_feedback(csv_path, p, "AI-Generated", "High", 0.5, comment=p)
    exported = pd.read_csv(export_feedback(csv_path, str(tmp_path / "out.csv")),
                           keep_default_na=False)
    expected = ["'" + p for p in payloads[:-1]] + ["plain - text"]
    assert exported["text"].tolist() == expected
    assert exported["comment"].tolist() == expected
    assert load_feedback(csv_path)["text"].tolist() == payloads
    raw = export_feedback(csv_path, str(tmp_path / "raw.csv"), escape_formulas=False)
    assert load_feedback(raw)["text"].tolist() == payloads
