"""
skolfie/app/gradio_app.py
--------------------------
Gradio web UI for the skolfie AI-vs-human text detector.

Tabs:
    1. Detector             paste text or upload .txt/.docx/.pdf, see the
                            label, AI probability, confidence and the top
                            contributing features (plus a warning when the
                            text hides invisible characters or look-alike
                            letters), then leave feedback (thumbs,
                            corrected label, comment)
    2. Feedback & Dataset   browse and export collected feedback, or add
                            labeled examples by hand
    3. Model                which model file is loaded, reload it

The model is loaded from `models/detector_model.joblib` (override with the
SKOLFIE_MODEL environment variable). The app never trains a model itself:
if the file is missing it says so and points at `python scripts/train.py`.
Feedback goes to `feedback_data.csv` (override with SKOLFIE_FEEDBACK).

Trust boundary: the model path comes only from the operator (environment),
never from the UI, because loading a joblib pickle runs code. Uploads and
pasted text are untrusted: uploads are size- and content-checked by
`skolfie.data.file_parsers`, texts are capped at MAX_TEXT_CHARS before any
model runs, and errors shown in the UI carry no paths or tracebacks. The
app binds to 127.0.0.1 without a share link by default; anyone who can open
it can read every collected feedback row on the Feedback & Dataset tab, so
only expose it (--host 0.0.0.0 / --share) to people allowed to see that.

Gradio is imported lazily inside `build_app()` so `skolfie.app` can be
imported (and its helpers tested) without Gradio installed.

Run with:  python -m skolfie.app
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skolfie.data import feedback as feedback_store
from skolfie.data.file_parsers import MAX_FILE_BYTES, FileParseError, extract_text_from_file
from skolfie.robust.normalize import obfuscation_note

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models") / "detector_model.joblib"
DEFAULT_FEEDBACK_PATH = Path("feedback_data.csv")

LABEL_AI = "AI-Generated"
LABEL_HUMAN = "Human-Written"
THUMBS = ("up", "down")

# Upper bounds on untrusted input, checked before any model runs.
MAX_TEXT_CHARS = 100_000
MAX_COMMENT_CHARS = 2_000

TRAIN_HINT = (
    "Train one first, e.g.\n\n"
    "```bash\n"
    "python scripts/train.py --data your_labeled_texts.csv\n"
    "```\n\n"
    "(`python scripts/train.py --seed-demo` builds a toy model from the "
    "synthetic test fixture for a quick try-out only.) "
    "Then use **Reload model** on the Model tab, or restart the app."
)

DISCLAIMER = (
    "AI-text detection is probabilistic. Do not use this result as the sole "
    "basis for high-stakes decisions (e.g. academic misconduct) without human review."
)


def model_path() -> Path:
    """Model location: $SKOLFIE_MODEL if set, else models/detector_model.joblib."""
    return Path(os.environ.get("SKOLFIE_MODEL") or DEFAULT_MODEL_PATH)


def feedback_path() -> Path:
    """Feedback CSV location: $SKOLFIE_FEEDBACK if set, else feedback_data.csv."""
    return Path(os.environ.get("SKOLFIE_FEEDBACK") or DEFAULT_FEEDBACK_PATH)


# ---------------------------------------------------------------------------
# Model handling (no Gradio needed)
# ---------------------------------------------------------------------------
@dataclass
class ModelHolder:
    """Holds the currently loaded model (or why there isn't one)."""

    path: Path = field(default_factory=model_path)
    model: Any = None
    status: str = ""

    def load(self) -> str:
        self.model = None
        if not self.path.exists():
            self.status = f"**No trained model found at `{self.path}`.**\n\n{TRAIN_HINT}"
            return self.status
        try:
            from skolfie.models.ensemble import AIDetectorModel

            self.model = AIDetectorModel.load(os.fspath(self.path))
        except Exception as exc:  # corrupt file, version mismatch, missing module...
            self.status = (
                f"**Could not load the model at `{self.path}`:** `{type(exc).__name__}: {exc}`\n\n"
                f"Retrain it. {TRAIN_HINT}"
            )
            return self.status
        mtime = datetime.fromtimestamp(self.path.stat().st_mtime).isoformat(timespec="seconds")
        self.status = f"Loaded model from `{self.path}` (modified {mtime})."
        return self.status

    @property
    def ready(self) -> bool:
        return self.model is not None


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~])")


def _md_escape(text: str) -> str:
    """Backslash-escape Markdown/HTML syntax in text shown in a gr.Markdown."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def resolve_input_text(text: Optional[str], file_path: Optional[str]) -> str:
    """An uploaded file wins over pasted text."""
    if file_path:
        return extract_text_from_file(file_path)
    return text or ""


def format_result(result: Dict[str, Any]) -> str:
    """Markdown summary of a prediction dict from AIDetectorModel.predict()."""
    lines = [
        f"## {result['label']}",
        f"**AI probability:** {result['ai_probability']:.1%}  ",
        f"**Confidence:** {result['confidence']}",
        "",
        "| Signal | AI probability |",
        "|---|---|",
        f"| ML ensemble | {result.get('ml_ai_probability', float('nan')):.1%} |",
        f"| Burrows' Delta | {result.get('burrows_ai_probability', float('nan')):.1%} |",
    ]
    obfuscation = result.get("obfuscation") or {}
    if obfuscation.get("invisible") or obfuscation.get("lookalikes"):
        lines += ["", f"**Warning:** {obfuscation_note(obfuscation)}"]
    top_words = result.get("burrows_top_words") or []
    if top_words:
        words = ", ".join(f"`{w}`" for w, _ in top_words)
        lines += ["", f"**Most distinctive function words:** {words}"]
    lines += ["", f"_{DISCLAIMER}_"]
    return "\n".join(lines)


def top_features_rows(result: Dict[str, Any]) -> List[List[Any]]:
    return [
        [f["feature"], f["direction"], round(float(f["importance"]), 4)]
        for f in result.get("top_features", [])
    ]


def analyze(holder: ModelHolder, text: Optional[str], file_path: Optional[str]
            ) -> Tuple[str, Dict[str, float], List[List[Any]], Optional[Dict[str, Any]]]:
    """
    Run the detector. Returns (markdown, label->probability, feature rows,
    state). `state` carries what the feedback buttons need and is None when
    nothing was analyzed.
    """
    if not holder.ready:
        return holder.status or holder.load(), {}, [], None
    try:
        content = resolve_input_text(text, file_path)
    except FileParseError as exc:
        return f"**Could not read the file:** {_md_escape(str(exc))}", {}, [], None
    except Exception:
        logger.exception("reading the uploaded file failed")
        return "**Could not read the file.** Is it a valid .txt, .docx or .pdf?", {}, [], None
    if len(content) > MAX_TEXT_CHARS:
        return (f"The text is too long ({len(content):,} characters); the limit is "
                f"{MAX_TEXT_CHARS:,}. Please use a shorter excerpt."), {}, [], None
    if len(content.split()) < 5:
        return "Please paste or upload some text (at least a few sentences).", {}, [], None

    try:
        result = holder.model.predict(content)
    except Exception:
        logger.exception("prediction failed")
        return "**The model could not analyze this text.**", {}, [], None
    probs = {LABEL_AI: result["ai_probability"], LABEL_HUMAN: result["human_probability"]}
    state = {
        "text": content,
        "label": result["label"],
        "confidence": result["confidence"],
        "ai_probability": result["ai_probability"],
        "features": result.get("raw_features"),
    }
    return format_result(result), probs, top_features_rows(result), state


def submit_feedback(state: Optional[Dict[str, Any]], thumbs: Optional[str],
                    correct_label: Optional[str], comment: Optional[str],
                    csv_path: Optional[Path] = None) -> str:
    """Append one feedback row for the last analysis."""
    if not state:
        return "Analyze a text first, then leave feedback."
    if not thumbs and not correct_label and not (comment or "").strip():
        return "Pick a thumbs up/down, a correct label, or write a comment."
    if thumbs and thumbs not in THUMBS:
        return "Thumbs must be up or down."
    if correct_label and correct_label not in (LABEL_AI, LABEL_HUMAN):
        return "Choose a valid correct label."
    if len(comment or "") > MAX_COMMENT_CHARS:
        return f"Please keep the comment under {MAX_COMMENT_CHARS:,} characters."
    path = Path(csv_path or feedback_path())
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    feedback_store.save_feedback(
        csv_path=os.fspath(path),
        text=state["text"],
        predicted_label=state["label"],
        predicted_confidence=state["confidence"],
        ai_probability=state["ai_probability"],
        user_label=correct_label or None,
        thumbs=thumbs or None,
        comment=(comment or "").strip() or None,
        features=state.get("features"),
    )
    return "Thanks! Feedback saved."


def load_feedback_table(csv_path: Optional[Path] = None):
    """Feedback rows without the bulky JSON features column."""
    df = feedback_store.load_feedback(os.fspath(csv_path or feedback_path()))
    return df.drop(columns=["features"], errors="ignore")


def export_feedback_file(csv_path: Optional[Path] = None) -> str:
    """Spreadsheet-safe copy of the feedback CSV in a fresh, unguessable temp file."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fd, out = tempfile.mkstemp(prefix=f"skolfie_feedback_{stamp}_", suffix=".csv")
    os.close(fd)
    return feedback_store.export_feedback(os.fspath(csv_path or feedback_path()), out)


def add_manual_example(text: Optional[str], label: Optional[str], comment: Optional[str],
                       csv_path: Optional[Path] = None) -> str:
    if not (text or "").strip():
        return "Enter some text first."
    if len(text) > MAX_TEXT_CHARS:
        return f"The text is too long; the limit is {MAX_TEXT_CHARS:,} characters."
    if len(comment or "") > MAX_COMMENT_CHARS:
        return f"Please keep the comment under {MAX_COMMENT_CHARS:,} characters."
    if label not in (LABEL_AI, LABEL_HUMAN):
        return "Choose a label."
    feedback_store.add_manual_label(
        os.fspath(csv_path or feedback_path()), text.strip(), label, comment or ""
    )
    return f"Added a {label} example."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app(holder: Optional[ModelHolder] = None):
    """Build and return the gr.Blocks app (does not launch it)."""
    import gradio as gr

    holder = holder or ModelHolder()
    if not holder.ready and not holder.status:
        holder.load()

    feature_headers = ["Feature", "This text vs. training average", "Contribution"]

    with gr.Blocks(title="skolfie: AI vs Human Text Detector") as demo:
        gr.Markdown(
            "# skolfie: AI vs Human Text Detector\n"
            "Stylometry (Burrows' Delta) + an ML ensemble estimate whether a text "
            "was written by a person or generated by an AI."
        )
        model_banner = gr.Markdown(
            "" if holder.ready else holder.status, visible=not holder.ready
        )
        last_result = gr.State(None)

        with gr.Tab("Detector"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_in = gr.Textbox(
                        label="Text", lines=12, placeholder="Paste the text to check here...",
                        max_length=MAX_TEXT_CHARS,
                    )
                    file_in = gr.File(
                        label="...or upload a .txt, .docx or .pdf file "
                              "(used instead of the text box)",
                        file_types=[".txt", ".docx", ".pdf"], type="filepath",
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    result_md = gr.Markdown()
                    probs_out = gr.Label(label="Prediction", num_top_classes=2)
            features_out = gr.Dataframe(
                headers=feature_headers, label="Top contributing features",
                interactive=False, wrap=True,
            )

            with gr.Accordion("Was this right? Leave feedback", open=False):
                with gr.Row():
                    thumbs_in = gr.Radio(["up", "down"], label="Thumbs")
                    correct_in = gr.Radio([LABEL_AI, LABEL_HUMAN], label="Correct label")
                comment_in = gr.Textbox(
                    label="Comment (optional)", lines=2, max_length=MAX_COMMENT_CHARS
                )
                feedback_btn = gr.Button("Submit feedback")
                feedback_status = gr.Markdown()

        with gr.Tab("Feedback & Dataset"):
            gr.Markdown(
                "Every feedback submission is stored here. Retrain with "
                "`python scripts/train.py --data ... --feedback "
                f"{feedback_path()}` to learn from it."
            )
            refresh_btn = gr.Button("Refresh")
            feedback_table = gr.Dataframe(label="Collected feedback", interactive=False, wrap=True)
            with gr.Row():
                export_btn = gr.Button("Export as CSV")
                export_file = gr.File(label="Exported dataset", interactive=False)
            with gr.Accordion("Add a labeled example manually", open=False):
                manual_text = gr.Textbox(label="Text", lines=6, max_length=MAX_TEXT_CHARS)
                manual_label = gr.Radio([LABEL_AI, LABEL_HUMAN], label="Label")
                manual_comment = gr.Textbox(
                    label="Comment (optional)", max_length=MAX_COMMENT_CHARS
                )
                manual_btn = gr.Button("Add example")
                manual_status = gr.Markdown()

        with gr.Tab("Model"):
            model_status = gr.Markdown(holder.status)
            reload_btn = gr.Button("Reload model")

        # --- wiring -------------------------------------------------------
        def _analyze(text, file):
            return analyze(holder, text, file)

        analyze_btn.click(
            _analyze, inputs=[text_in, file_in],
            outputs=[result_md, probs_out, features_out, last_result],
        )

        def _feedback(state, thumbs, correct, comment):
            return submit_feedback(state, thumbs, correct, comment)

        feedback_btn.click(
            _feedback, inputs=[last_result, thumbs_in, correct_in, comment_in],
            outputs=[feedback_status],
        )

        refresh_btn.click(lambda: load_feedback_table(), outputs=[feedback_table])
        export_btn.click(lambda: export_feedback_file(), outputs=[export_file])

        def _manual(text, label, comment):
            return add_manual_example(text, label, comment), load_feedback_table()

        manual_btn.click(
            _manual, inputs=[manual_text, manual_label, manual_comment],
            outputs=[manual_status, feedback_table],
        )

        def _reload():
            status = holder.load()
            banner = gr.update(value="" if holder.ready else status, visible=not holder.ready)
            return status, banner

        reload_btn.click(_reload, outputs=[model_status, model_banner])
        demo.load(lambda: load_feedback_table(), outputs=[feedback_table])

    return demo


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m skolfie.app", description="Launch the skolfie Gradio app."
    )
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link (anyone with it can read collected feedback).",
    )
    args = parser.parse_args(argv)

    if args.share or args.host not in ("127.0.0.1", "localhost", "::1"):
        print("[warning] the app is reachable beyond this machine: anyone who can open it "
              "can read and export every collected feedback text.")
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share,
                       max_file_size=MAX_FILE_BYTES)


if __name__ == "__main__":
    main()
