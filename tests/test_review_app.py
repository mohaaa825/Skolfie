"""
tests/test_review_app.py
-------------------------
Regression tests from the review of the user-facing surface: upload parsing
limits, the Gradio app's input handling, the feedback CSV, the old ensemble's
Burrows' Delta leak, and scripts/train.py argument handling.
"""

from __future__ import annotations

import importlib.util
import math
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.app import gradio_app
from skolfie.data import feedback as fb
from skolfie.data.file_parsers import (
    FileParseError,
    extract_text_from_file,
    read_docx,
    read_pdf,
)
from tests.test_file_parsers import _minimal_pdf

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Upload parsing
# --------------------------------------------------------------------------- #
def _docx_with_padding(tmp_path: Path, pad_bytes: int) -> Path:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Hello docx.")
    src = tmp_path / "src.docx"
    document.save(str(src))
    out = tmp_path / "padded.docx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = data.replace(b"<w:body>", b"<w:body>" + b" " * pad_bytes)
            zout.writestr(item, data)
    return out


def test_docx_zip_bomb_rejected_before_unpacking(tmp_path: Path) -> None:
    bomb = _docx_with_padding(tmp_path, 60 * 1024 * 1024)
    assert bomb.stat().st_size < 1024 * 1024  # tiny upload, huge once unpacked
    with pytest.raises(FileParseError, match="too large once unpacked"):
        extract_text_from_file(bomb)


def test_docx_small_padding_still_reads(tmp_path: Path) -> None:
    assert read_docx(str(_docx_with_padding(tmp_path, 1000))) == "Hello docx."


def test_oversized_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("word " * 1000)
    with pytest.raises(FileParseError, match="too large"):
        extract_text_from_file(path, max_bytes=100)


@pytest.mark.parametrize("name, content", [
    ("fake.docx", b"plain text, not a zip"),
    ("fake.pdf", b"plain text, not a pdf"),
    ("binary.txt", bytes(range(256)) * 4),
])
def test_content_must_match_extension(tmp_path: Path, name: str, content: bytes) -> None:
    pytest.importorskip("docx")
    pytest.importorskip("pypdf")
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(FileParseError) as info:
        extract_text_from_file(path)
    assert str(tmp_path) not in str(info.value)


def test_corrupt_pdf_body_gives_safe_error(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\n garbage without xref")
    with pytest.raises(FileParseError, match="Could not read this .pdf") as info:
        extract_text_from_file(path)
    assert str(tmp_path) not in str(info.value)


def test_password_protected_pdf(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    src = tmp_path / "plain.pdf"
    src.write_bytes(_minimal_pdf("Secret words"))
    writer = pypdf.PdfWriter(clone_from=str(src))
    writer.encrypt(user_password="hunter2", owner_password="owner")
    locked = tmp_path / "locked.pdf"
    writer.write(str(locked))
    with pytest.raises(FileParseError, match="password-protected"):
        extract_text_from_file(locked)


def test_pdf_page_cap(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    src = tmp_path / "one.pdf"
    src.write_bytes(_minimal_pdf("Page text"))
    writer = pypdf.PdfWriter(clone_from=str(src))
    for _ in range(4):
        writer.add_page(writer.pages[0])
    many = tmp_path / "many.pdf"
    writer.write(str(many))
    with pytest.raises(FileParseError, match="5 pages; the limit is 3"):
        read_pdf(str(many), max_pages=3)
    assert read_pdf(str(many)).count("Page text") == 5


def test_unsupported_extension_is_still_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.rtf"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_file(path)


# --------------------------------------------------------------------------- #
# Gradio app helpers
# --------------------------------------------------------------------------- #
class _FakeModel:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def predict(self, text: str) -> dict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("secret internal /srv/path detail")
        return {"label": "AI-Generated", "ai_probability": 0.8, "human_probability": 0.2,
                "confidence": "Medium", "top_features": [], "raw_features": {}}


def _holder(model: _FakeModel) -> gradio_app.ModelHolder:
    holder = gradio_app.ModelHolder(path=Path("unused.joblib"))
    holder.model = model
    return holder


def test_analyze_rejects_overlong_text_before_model() -> None:
    model = _FakeModel()
    md, probs, rows, state = gradio_app.analyze(
        _holder(model), "word " * (gradio_app.MAX_TEXT_CHARS // 5 + 1), None)
    assert "too long" in md and state is None and model.calls == 0


def test_analyze_file_errors_hide_paths(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    bad = tmp_path / "upload.docx"
    bad.write_bytes(b"not a zip")
    md, _, _, state = gradio_app.analyze(_holder(_FakeModel()), "", str(bad))
    assert "Could not read the file" in md and state is None
    assert str(tmp_path) not in md


def test_analyze_unexpected_read_error_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args):
        raise OSError("/very/secret/path is gone")

    monkeypatch.setattr(gradio_app, "extract_text_from_file", boom)
    md, *_ = gradio_app.analyze(_holder(_FakeModel()), "", "/x/upload.txt")
    assert "secret" not in md and "Could not read the file" in md


def test_analyze_escapes_markdown_in_error(tmp_path: Path) -> None:
    path = tmp_path / "x.[a](http:evil)"
    path.write_text("hello")
    md, *_ = gradio_app.analyze(_holder(_FakeModel()), "", str(path))
    assert "[a](http:evil)" not in md


def test_analyze_prediction_error_is_generic() -> None:
    md, probs, rows, state = gradio_app.analyze(
        _holder(_FakeModel(fail=True)), "one two three four five six", None)
    assert "could not analyze" in md and "secret" not in md and state is None


def test_submit_feedback_validates_choices(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    state = {"text": "t", "label": "AI-Generated", "confidence": "High", "ai_probability": 0.9}
    assert "up or down" in gradio_app.submit_feedback(state, "sideways", None, "", csv_path)
    assert "valid" in gradio_app.submit_feedback(state, None, "ai", "", csv_path)
    long_comment = "x" * (gradio_app.MAX_COMMENT_CHARS + 1)
    assert "under" in gradio_app.submit_feedback(state, "up", None, long_comment, csv_path)
    assert not csv_path.exists()
    msg = gradio_app.submit_feedback(state, "up", None, "", csv_path)
    assert "saved" in msg and str(tmp_path) not in msg


def test_manual_example_caps_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    text = "x" * (gradio_app.MAX_TEXT_CHARS + 1)
    assert "too long" in gradio_app.add_manual_example(text, "AI-Generated", "", csv_path)
    assert not csv_path.exists()


def test_export_uses_unique_temp_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    fb.save_feedback(str(csv_path), "=1+1", "AI-Generated", "High", 0.9)
    first = gradio_app.export_feedback_file(csv_path)
    second = gradio_app.export_feedback_file(csv_path)
    try:
        assert first != second  # was one fixed name per second in the shared temp dir
        assert pd.read_csv(first)["text"].iloc[0] == "'=1+1"
    finally:
        Path(first).unlink(missing_ok=True)
        Path(second).unlink(missing_ok=True)


def test_main_defaults_are_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: dict = {}

    class _App:
        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    monkeypatch.setattr(gradio_app, "build_app", lambda: _App())
    gradio_app.main([])
    assert launched["server_name"] == "127.0.0.1"
    assert launched["share"] is False
    assert launched["max_file_size"] == gradio_app.MAX_FILE_BYTES


# --------------------------------------------------------------------------- #
# Feedback CSV
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["NA", "null", "N/A", "nan", "None", "12345"])
def test_feedback_keeps_na_like_and_numeric_texts(tmp_path: Path, text: str) -> None:
    csv_path = str(tmp_path / "fb.csv")
    fb.save_feedback(csv_path, text, "AI-Generated", "High", 0.9, comment="NA")
    row = fb.load_feedback(csv_path).iloc[0]
    assert row["text"] == text and row["comment"] == "NA"
    assert isinstance(row["text"], str)


def test_feedback_empty_cells_still_missing(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    fb.save_feedback(csv_path, "text", "AI-Generated", "High", 0.9)
    row = fb.load_feedback(csv_path).iloc[0]
    assert pd.isna(row["user_label"]) and pd.isna(row["comment"])
    assert row["ai_probability"] == pytest.approx(0.9)


def test_feedback_odd_characters(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    fb.save_feedback(csv_path, "a\x00b", "AI-Generated", "High", 0.5)
    fb.save_feedback(csv_path, "lone \ud800 surrogate", "AI-Generated", "High", 0.5)
    fb.save_feedback(csv_path, "line1\r\nline2\rline3​‮RTL", "x", "y", 0.5)
    texts = fb.load_feedback(csv_path)["text"].tolist()
    assert texts == ["ab", "lone ? surrogate", "line1\r\nline2\rline3​‮RTL"]


def test_feedback_concurrent_appends(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    n = 16
    barrier = threading.Barrier(n)

    def write(i: int) -> None:
        barrier.wait()
        fb.save_feedback(csv_path, f"text {i} " + "x" * 20_000, "AI-Generated", "High", 0.5)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    df = fb.load_feedback(csv_path)
    assert sorted(int(t.split()[1]) for t in df["text"]) == list(range(n))
    assert list(df.columns) == fb.FEEDBACK_COLUMNS


def test_export_escapes_formulas_but_master_stays_raw(tmp_path: Path) -> None:
    csv_path = str(tmp_path / "fb.csv")
    payloads = ["=HYPERLINK(\"http://x\")", "+1", "-2", "@SUM(A1)", "\tx", "plain - text"]
    for p in payloads:
        fb.save_feedback(csv_path, p, "AI-Generated", "High", 0.5, comment=p)
    exported = pd.read_csv(fb.export_feedback(csv_path, str(tmp_path / "out.csv")),
                           keep_default_na=False)
    expected = ["'" + p for p in payloads[:-1]] + ["plain - text"]
    assert exported["text"].tolist() == expected
    assert exported["comment"].tolist() == expected
    assert fb.load_feedback(csv_path)["text"].tolist() == payloads
    raw = fb.export_feedback(csv_path, str(tmp_path / "raw.csv"), escape_formulas=False)
    assert fb.load_feedback(raw)["text"].tolist() == payloads


# --------------------------------------------------------------------------- #
# Old ensemble
# --------------------------------------------------------------------------- #
def test_ensemble_burrows_training_features_do_not_leak_labels() -> None:
    from sklearn.metrics import roc_auc_score

    from skolfie.features.stylometric import tokenize_words
    from skolfie.models.ensemble import AIDetectorModel
    from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

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


def test_ensemble_oof_rows_neutral_with_tiny_class() -> None:
    from skolfie.models.ensemble import AIDetectorModel

    rows = AIDetectorModel._oof_burrows_rows([["the", "a"]] * 3, ["human", "human", "ai"])
    assert rows.tolist() == [[0.0, 0.0, 0.5]] * 3


def test_ensemble_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skolfie.models import ensemble

    path = tmp_path / "model.joblib"
    path.write_bytes(b"previous good model")

    def broken_dump(obj, target):
        Path(target).write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(ensemble.joblib, "dump", broken_dump)
    with pytest.raises(KeyboardInterrupt):
        ensemble.AIDetectorModel().save(str(path))
    assert path.read_bytes() == b"previous good model"
    assert [p.name for p in tmp_path.iterdir()] == ["model.joblib"]


# --------------------------------------------------------------------------- #
# Stylometric features
# --------------------------------------------------------------------------- #
def test_fallback_tokenizer_keeps_unicode_words() -> None:
    from skolfie.features.stylometric import _simple_word_tokenize

    assert _simple_word_tokenize("naïve café Привет don't 42") == [
        "naïve", "café", "Привет", "don't"]


@pytest.mark.parametrize("text", [
    "", "   ", "​​", "😀😀😀", "!!!", "a", "مرحبا بالعالم", "Привет, мир. Да!",
    "x" * 5000,
])
def test_features_finite_on_odd_inputs(extractor, text: str) -> None:
    features = extractor.extract_features(text)
    assert all(math.isfinite(v) for v in features.values())


# --------------------------------------------------------------------------- #
# scripts/train.py
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("skolfie_train_review",
                                                  REPO_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_drops_duplicates_last_wins(train) -> None:
    texts, labels, dropped = train.drop_duplicate_texts(
        ["a", "b", "a", "c"], ["human", "ai", "ai", "human"])
    assert (texts, labels, dropped) == (["b", "a", "c"], ["ai", "ai", "human"], 1)


def test_train_rejects_bad_paths_before_training(train, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train.main(["--data", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit):
        train.main(["--feedback", str(tmp_path / "missing.csv")])
    with pytest.raises(SystemExit):
        train.main(["--seed-demo", "--out", str(tmp_path)])


def test_train_test_size_too_small_is_clean_error(train, tmp_path: Path) -> None:
    data = tmp_path / "tiny.csv"
    pd.DataFrame({"text": ["h one", "h two", "a one", "a two"],
                  "label": ["human", "human", "ai", "ai"]}).to_csv(data, index=False)
    out = tmp_path / "m.joblib"
    assert train.main(["--data", str(data), "--test-size", "0.2", "--out", str(out)]) == 2
    assert not out.exists()
