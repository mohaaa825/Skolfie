"""
tests/test_app.py
------------------
The Gradio app builds without a trained model and its helpers behave.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skolfie.app import gradio_app


def test_missing_model_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKOLFIE_MODEL", str(tmp_path / "nope.joblib"))
    holder = gradio_app.ModelHolder()
    status = holder.load()
    assert not holder.ready
    assert "scripts/train.py" in status
    md, probs, rows, state = gradio_app.analyze(holder, "Some text " * 20, None)
    assert "scripts/train.py" in md and state is None


def test_feedback_helpers(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    assert "Analyze a text first" in gradio_app.submit_feedback(None, "up", None, "", csv_path)
    state = {"text": "hello there", "label": "AI-Generated", "confidence": "High",
             "ai_probability": 0.9, "features": {"x": 1.0}}
    msg = gradio_app.submit_feedback(state, "down", "Human-Written", "nope", csv_path)
    assert "saved" in msg
    table = gradio_app.load_feedback_table(csv_path)
    assert len(table) == 1 and "features" not in table.columns
    assert table.iloc[0]["user_label"] == "Human-Written"
    assert "Added" in gradio_app.add_manual_example("x y z", "AI-Generated", "", csv_path)
    assert Path(gradio_app.export_feedback_file(csv_path)).exists()


def test_build_app_returns_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gr = pytest.importorskip("gradio")
    monkeypatch.setenv("SKOLFIE_MODEL", str(tmp_path / "missing.joblib"))
    demo = gradio_app.build_app()
    assert isinstance(demo, gr.Blocks)
