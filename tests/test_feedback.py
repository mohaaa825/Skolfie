"""
tests/test_feedback.py
-----------------------
Round-trip of the feedback CSV (save / load / manual label / export).
"""

from __future__ import annotations

import json
from pathlib import Path

from skolfie.data.feedback import (
    FEEDBACK_COLUMNS,
    add_manual_label,
    export_feedback,
    load_feedback,
    save_feedback,
)


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
