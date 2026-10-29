"""
tests/test_app.py
------------------
The web app on top of the Detector (roadmap step 11), without Gradio, torch
or real models: a fake detector stands in for `skolfie.service.Detector`.
Covers the result -> HTML formatting, model location, the auth / share
guard, the Feedback & Dataset tab gating and feedback saving with Mixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skolfie.app import gradio_app as app
from skolfie.data import feedback as fb

TEXT = "Intro <b>bold</b> & more. " + "word " * 40 + "The end."


def _result(label: str = "mixed", **over) -> dict:
    ai_start = TEXT.index("word")
    result = {
        "label": label,
        "headline": {"mixed": "Mixed: parts look AI-generated",
                     "ai": "Likely AI-generated",
                     "human": "Likely written by a human"}.get(label, "Uncertain"),
        "p_ai": 0.42,
        "n_words": len(TEXT.split()),
        "verdict": "uncertain",
        "chunks": [{"start": 0, "end": ai_start, "p": 0.1, "label": "human"},
                   {"start": ai_start, "end": len(TEXT), "p": 0.95, "label": "ai"}],
        "highlights": [{"start": 0, "end": ai_start, "p": 0.1, "label": "human"},
                       {"start": ai_start, "end": len(TEXT), "p": 0.95, "label": "ai"}],
        "top_features": [{"feature": "ttr", "value": 0.31, "direction": "ai",
                          "reason": "low vocabulary variety"}],
        "reasons": ["Low vocabulary <variety>, typical of AI writing"],
        "notes": ["The style features alone point to human."],
        "warnings": [
            {"kind": "short_text", "message": "Short text: less reliable."},
            {"kind": "fairness", "message": "Essays by non-native English writers are flagged "
                                            "25% of the time."},
        ],
        "disclaimer": "Never the sole basis for a misconduct decision.",
    }
    result.update(over)
    return result


class FakeDetector:
    def __init__(self, result: dict | None = None, fail: bool = False) -> None:
        self.result = result or _result()
        self.fail = fail
        self.texts: list[str] = []

    def check(self, text: str) -> dict:
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("boom at /secret/path")
        return dict(self.result)

    def info(self) -> dict:
        return {"source": "dirs", "files": {"verdict": "/m/stacked.joblib"},
                "missing": ["reference.json"], "mixed_enabled": True, "known_limitations": 2,
                "cutoffs": {"ai_threshold": 0.7}}


def _holder(detector=None) -> app.DetectorHolder:
    holder = app.DetectorHolder(source=app.ModelSource(models_dir=Path("unused")))
    holder.detector = detector if detector is not None else FakeDetector()
    return holder


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def test_format_percent() -> None:
    assert app.format_percent(0.834) == "83%"
    assert app.format_percent(None) == "n/a"
    assert app.format_percent(float("nan")) == "n/a"


@pytest.mark.parametrize("label,name", [("human", "Human"), ("ai", "AI"), ("mixed", "Mixed"),
                                        ("uncertain", "Uncertain"), ("too_short", "Too short")])
def test_badge_names(label: str, name: str) -> None:
    html = app.badge_html(label)
    assert f">{name}</span>" in html and f"sk-badge-{label}" in html


def test_badge_unknown_label_is_escaped() -> None:
    html = app.badge_html("<script>")
    assert "<script>" not in html and "sk-badge-uncertain" in html


def test_highlighted_text_escapes_and_marks_only_given_spans() -> None:
    text = "a <i>b</i> c d"
    spans = [{"start": 2, "end": 10, "p": 0.9, "label": "ai"},
             {"start": 5, "end": 12, "p": 0.8, "label": "ai"}]  # overlaps: clipped
    html = app.highlighted_text_html(text, spans)
    assert "<i>" not in html and "&lt;i&gt;b&lt;/i&gt;" in html
    assert html.count("<mark") == 2
    assert 'title="AI score 90%"' in html


def test_ai_spans_falls_back_to_chunks() -> None:
    result = {"chunks": [{"start": 0, "end": 3, "p": 0.9, "label": "ai"},
                         {"start": 3, "end": 6, "p": 0.1, "label": "human"}]}
    assert [s["start"] for s in app.ai_spans(result)] == [0]
    assert app.ai_spans({}) == []


def test_render_result_html_has_every_part_and_escapes() -> None:
    page = app.render_result_html(_result(), TEXT)
    assert "<b>bold</b>" not in page and "&lt;b&gt;bold&lt;/b&gt;" in page
    assert ">Mixed</span>" in page and "Mixed: parts look AI-generated" in page
    assert "42%" in page                                    # AI likelihood
    assert page.count("<mark") == 2                         # legend + the AI span
    assert "&lt;variety&gt;" in page                        # reasons, escaped
    assert "The style features alone point to human." in page
    assert "Never the sole basis" in page
    # The fairness warning comes first, in its own box, before the text and reasons.
    fairness = page.index("non-native English writers")
    assert fairness < page.index("Short text: less reliable.") < page.index("The text</p>")
    assert "known limitations" in page


def test_render_result_without_ai_parts_says_so() -> None:
    result = _result("human", highlights=[{"start": 0, "end": len(TEXT), "p": 0.1,
                                           "label": "human"}], warnings=[])
    page = app.render_result_html(result, TEXT)
    assert "No part of the text stands out" in page and "<mark" not in page
    assert "known limitations" not in page


def test_render_too_short_has_no_meter_or_text() -> None:
    result = {"label": "too_short", "headline": "Too short to judge (needs at least 30 words)",
              "p_ai": None, "n_words": 5, "warnings": [], "disclaimer": "D"}
    page = app.render_result_html(result, "five words of text here")
    assert ">Too short</span>" in page and "AI likelihood" not in page
    assert "sk-text" not in page and "5 words" in page


def test_render_uses_default_disclaimer_and_plain_string_warnings() -> None:
    page = app.render_result_html({"label": "ai", "p_ai": 0.9, "warnings": ["odd <x>"]}, "t")
    assert app.DISCLAIMER.split(".")[0] in page
    assert "odd &lt;x&gt;" in page


def test_css_and_head_come_from_the_tokens() -> None:
    tokens = dict(app.THEME_TOKENS, sky_400="#123456", font_serif="Newsreader")
    css = app.app_css(tokens)
    assert "--sk-sky-400: #123456" in css and "--sk-bg: #ffffff" in css
    assert '"Newsreader", Georgia' in css
    assert "family=Newsreader" in app.app_head(tokens)


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


@pytest.mark.parametrize("fg,bg", app.CONTRAST_PAIRS)
def test_text_colours_meet_wcag_aa(fg: str, bg: str) -> None:
    a, b = (_luminance(app.THEME_TOKENS[k]) for k in (fg, bg))
    ratio = (max(a, b) + 0.05) / (min(a, b) + 0.05)
    assert ratio >= 4.5, f"{fg} on {bg}: {ratio:.2f}"


def test_attach_note_escapes_the_file_name() -> None:
    assert "Upload" in app.attach_note_html(None)
    note = app.attach_note_html("/tmp/gradio/abc/<i>essay&.docx")
    assert "&lt;i&gt;essay&amp;.docx" in note and "/tmp/gradio" not in note


def test_format_info_markdown() -> None:
    md = app.format_info_markdown(FakeDetector().info())
    assert "stacked.joblib" in md and "reference.json" in md and "ai_threshold" in md
    assert "Mixed detection:** on" in md and "limitations shown with AI / Mixed results:** 2" in md
    assert app.format_info_markdown({}) == ""
    assert "`a'b`" in app.format_info_markdown({"missing": ["a`b"]})


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #
def test_check_text_with_fake_detector() -> None:
    detector = FakeDetector()
    html, state = app.check_text(_holder(detector), TEXT, None)
    assert detector.texts == [TEXT]
    assert "sk-result" in html
    assert state == {"text": TEXT, "label": "mixed", "p_ai": 0.42, "features": {"ttr": 0.31}}


def test_check_text_uploaded_file_wins(tmp_path: Path) -> None:
    path = tmp_path / "essay.txt"
    path.write_text("From the file. " * 10, encoding="utf-8")
    detector = FakeDetector()
    app.check_text(_holder(detector), "pasted", str(path))
    assert detector.texts == ["From the file. " * 10]


def test_check_text_empty_and_not_ready() -> None:
    detector = FakeDetector()
    html, state = app.check_text(_holder(detector), "   ", None)
    assert "Paste a text" in html and state is None and not detector.texts
    holder = _holder()
    holder.detector = None
    html, state = app.check_text(holder, TEXT, None)
    assert "No models are loaded" in html and state is None


def test_holder_without_models_explains_how_to_get_them() -> None:
    def missing(_source):
        raise FileNotFoundError("stacked.joblib not found; run notebooks/stack.ipynb")

    holder = app.DetectorHolder(source=app.ModelSource(models_dir=Path("nowhere")),
                                loader=missing)
    status = holder.load()
    assert not holder.ready and holder.info() == {}
    assert "No models found" in status and "stack.ipynb" in status
    assert "skolfie bundle" in status and "notebooks/" in status


def test_holder_load_errors_are_generic() -> None:
    def corrupt(_source):
        raise ValueError("pickle data at /secret/x was truncated")

    holder = app.DetectorHolder(source=app.ModelSource(models_dir=Path("m")), loader=corrupt)
    status = holder.load()
    assert "Could not load" in status and "ValueError" in status and "secret" not in status


def test_holder_loads_with_loader() -> None:
    holder = app.DetectorHolder(source=app.ModelSource(bundle=Path("b")),
                                loader=lambda _s: FakeDetector())
    assert "loaded" in holder.load() and holder.ready
    assert holder.info()["cutoffs"] == {"ai_threshold": 0.7}


# --------------------------------------------------------------------------- #
# Model location
# --------------------------------------------------------------------------- #
def test_resolve_model_source_precedence(tmp_path: Path) -> None:
    env = {"SKOLFIE_BUNDLE": "/env/bundle", "SKOLFIE_MODELS_DIR": "/env/models",
           "SKOLFIE_REPORTS_DIR": "/env/reports"}
    assert app.resolve_model_source("/cli/bundle", environ=env).bundle == Path("/cli/bundle")
    src = app.resolve_model_source(models_dir="/cli/models", environ=env)
    assert src.bundle is None and src.models_dir == Path("/cli/models")
    assert src.reports_dir == Path("/env/reports")
    assert app.resolve_model_source(environ=env).bundle == Path("/env/bundle")
    src = app.resolve_model_source(environ={"SKOLFIE_MODELS_DIR": "/env/models"})
    assert src.models_dir == Path("/env/models") and src.reports_dir is None
    assert app.resolve_model_source(environ={}).models_dir == app.DEFAULT_MODELS_DIR
    with pytest.raises(ValueError):
        app.resolve_model_source("/b", "/m", environ={})


def test_reports_dir_defaults_to_sibling(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    src = app.resolve_model_source(models_dir=tmp_path / "models", environ={})
    assert src.reports_dir == tmp_path / "reports"


def test_model_source_load_calls_the_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    calls = []

    class Detector:
        @classmethod
        def from_bundle(cls, path):
            calls.append(("bundle", path))
            return "B"

        @classmethod
        def from_dirs(cls, models_dir, reports_dir=None):
            calls.append(("dirs", models_dir, reports_dir))
            return "D"

    monkeypatch.setitem(sys.modules, "skolfie.service",
                        types.SimpleNamespace(Detector=Detector))
    assert app.ModelSource(bundle=Path("b")).load() == "B"
    assert app.ModelSource(models_dir=Path("m"), reports_dir=Path("r")).load() == "D"
    assert calls == [("bundle", Path("b")), ("dirs", Path("m"), Path("r"))]


# --------------------------------------------------------------------------- #
# Auth, share guard, feedback tab gating
# --------------------------------------------------------------------------- #
def test_parse_auth() -> None:
    assert app.parse_auth(None) is None and app.parse_auth("") is None
    assert app.parse_auth("teacher:pa:ss") == ("teacher", "pa:ss")
    for bad in ("nopassword", ":pw", "user:"):
        with pytest.raises(ValueError):
            app.parse_auth(bad)


def test_exposure_error() -> None:
    auth = ("u", "p")
    assert app.exposure_error("127.0.0.1", False, None, False) is None
    assert app.exposure_error("0.0.0.0", False, None, False)
    assert app.exposure_error("127.0.0.1", True, None, False)
    assert app.exposure_error("0.0.0.0", True, auth, False) is None
    assert app.exposure_error("0.0.0.0", True, None, True) is None


class _App:
    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.skolfie_launch_kwargs = {"css": "x"}

    def launch(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch):
    for name in ("GRADIO_SERVER_NAME", "SKOLFIE_APP_AUTH", "SKOLFIE_SHOW_FEEDBACK",
                 "SKOLFIE_BUNDLE", "SKOLFIE_MODELS_DIR", "SKOLFIE_REPORTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    built: dict = {}

    def build(holder, show_feedback):
        built.update(holder=holder, show_feedback=show_feedback, app=_App())
        return built["app"]

    monkeypatch.setattr(app.DetectorHolder, "load", lambda self: "no models")
    monkeypatch.setattr(app, "build_app", build)
    return built


def test_main_refuses_share_or_public_host_without_auth(launcher) -> None:
    for argv in (["--share"], ["--host", "0.0.0.0"]):
        with pytest.raises(SystemExit):
            app.main(argv)
    assert not launcher


def test_main_passes_auth_to_launch(launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    app.main(["--share", "--auth", "teacher:secret", "--models-dir", "/m"])
    kwargs = launcher["app"].kwargs
    assert kwargs["auth"] == ("teacher", "secret") and kwargs["share"] is True
    assert kwargs["css"] == "x"
    assert launcher["holder"].source.models_dir == Path("/m")
    assert launcher["show_feedback"] is False

    monkeypatch.setenv("SKOLFIE_APP_AUTH", "env:pw")
    app.main(["--host", "0.0.0.0"])
    assert launcher["app"].kwargs["auth"] == ("env", "pw")


def test_main_explicit_no_auth_flag(launcher) -> None:
    app.main(["--share", app.NO_AUTH_FLAG])
    assert launcher["app"].kwargs["auth"] is None


def test_main_rejects_malformed_auth_and_two_sources(launcher) -> None:
    with pytest.raises(SystemExit):
        app.main(["--auth", "nocolon"])
    with pytest.raises(SystemExit):
        app.main(["--bundle", "b", "--models-dir", "m"])


def test_feedback_tab_gating(launcher, monkeypatch: pytest.MonkeyPatch) -> None:
    app.main([])
    assert launcher["show_feedback"] is False
    app.main(["--show-feedback"])
    assert launcher["show_feedback"] is True
    monkeypatch.setenv("SKOLFIE_SHOW_FEEDBACK", "1")
    app.main([])
    assert launcher["show_feedback"] is True
    monkeypatch.setenv("SKOLFIE_SHOW_FEEDBACK", "0")
    app.main([])
    assert launcher["show_feedback"] is False


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def test_feedback_with_mixed_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    assert "Check a text first" in app.submit_feedback(None, "up", None, "", csv_path)
    _, state = app.check_text(_holder(), TEXT, None)
    msg = app.submit_feedback(state, "down", "Mixed", "  first half is hers ", csv_path)
    assert "saved" in msg
    table = app.load_feedback_table(csv_path)
    assert len(table) == 1 and "features" not in table.columns
    row = table.iloc[0]
    assert row["user_label"] == fb.LABEL_MIXED and row["predicted_label"] == "mixed"
    assert row["thumbs"] == "down" and row["comment"] == "first half is hers"
    assert float(row["ai_probability"]) == pytest.approx(0.42)
    assert row["text"] == TEXT

    app.submit_feedback(state, None, "Human", "", csv_path)
    app.submit_feedback(state, None, "AI", "", csv_path)
    labels = app.load_feedback_table(csv_path)["user_label"].tolist()
    assert labels == [fb.LABEL_MIXED, fb.LABEL_HUMAN, fb.LABEL_AI]


def test_feedback_on_too_short_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    state = {"text": "tiny", "label": "too_short", "p_ai": None, "features": None}
    assert "saved" in app.submit_feedback(state, None, "Human", "", csv_path)
    row = app.load_feedback_table(csv_path).iloc[0]
    assert row["predicted_label"] == "too_short"


def test_manual_examples_and_export(tmp_path: Path) -> None:
    csv_path = tmp_path / "fb.csv"
    assert "Choose a label" in app.add_manual_example("x y z", "Robot", "", csv_path)
    assert "Added" in app.add_manual_example("x y z", "Mixed", "", csv_path)
    assert "Added" in app.add_manual_example("a b c", "AI", "", csv_path)
    table = app.load_feedback_table(csv_path)
    assert table["user_label"].tolist() == [fb.LABEL_MIXED, fb.LABEL_AI]
    assert table["ai_probability"].isna().tolist() == [True, False]
    out = Path(app.export_feedback_file(csv_path))
    try:
        assert out.exists()
    finally:
        out.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# With Gradio installed: the app builds, the feedback panel only when asked
# --------------------------------------------------------------------------- #
def _panel_labels(demo) -> list[str]:
    import gradio as gr

    return [b.label for b in demo.blocks.values() if isinstance(b, gr.Accordion)]


def test_build_app_gates_the_feedback_panel() -> None:
    gr = pytest.importorskip("gradio")
    demo = app.build_app(_holder())
    assert isinstance(demo, gr.Blocks)
    assert "Feedback & Dataset" not in _panel_labels(demo)
    assert "Model info" in _panel_labels(demo)
    assert "css" in demo.skolfie_launch_kwargs or demo.css
    demo = app.build_app(_holder(), show_feedback=True)
    assert "Feedback & Dataset" in _panel_labels(demo)


def test_build_app_without_models() -> None:
    pytest.importorskip("gradio")
    holder = app.DetectorHolder(source=app.ModelSource(models_dir=Path("nowhere")),
                                loader=lambda _s: (_ for _ in ()).throw(FileNotFoundError("x")))
    demo = app.build_app(holder)
    assert not holder.ready and "No models found" in holder.status
    assert demo is not None
