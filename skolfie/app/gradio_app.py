"""
skolfie/app/gradio_app.py
--------------------------
Gradio web UI for Skolfie, for teachers checking student essays (roadmap
step 11). It runs on `skolfie.service.Detector`.

One page, top to bottom:
    Checker                 a composer card: paste an essay or attach a
                            .txt/.docx/.pdf with "+", press Check; then result
                            cards: a headline badge (Human / AI / Mixed /
                            Uncertain / Too short), the AI likelihood,
                            warnings (known fairness limitations first and
                            most prominently), the text with its AI-looking
                            parts highlighted, the plain-language reasons and
                            notes, and the disclaimer
    Feedback                looks right / wrong, corrected label Human / AI /
                            Mixed, comment
    How it works            verdicts, limitations, privacy
    Model info              Detector.info(): which files were loaded or are
                            missing, cut-offs, versions; a Reload button
    Feedback & Dataset      browse and export collected feedback, or add
                            labeled examples by hand. Only built when the
                            operator asks for it (--show-feedback or
                            SKOLFIE_SHOW_FEEDBACK=1), because it shows every
                            stored student text to whoever can open the app

Where the models come from (operator only, never the UI, since loading a
joblib file runs code), first match wins:
    --bundle DIR                 a folder made by `skolfie bundle`
    --models-dir DIR             the models folder (as saved by the notebooks),
      [--reports-dir DIR]        with the reports folder (default: a `reports`
                                 folder next to the models folder, if any)
    $SKOLFIE_BUNDLE, $SKOLFIE_MODELS_DIR [$SKOLFIE_REPORTS_DIR]
    ./models (and ./reports)
Without models the app still starts and says how to get them.

Access: the app binds to 127.0.0.1 without a share link by default. A
username and password (--auth user:password or $SKOLFIE_APP_AUTH) go to
Gradio's login page. The app refuses to start with --share or a non-local
--host unless a password is set (or --no-auth-i-know-the-risks is given).
Feedback goes to feedback_data.csv (override with $SKOLFIE_FEEDBACK).

Untrusted input: uploads are size- and content-checked by
`skolfie.data.file_parsers`, texts are capped at MAX_TEXT_CHARS before any
model runs, every piece of text in the result page is HTML-escaped, and
errors shown in the UI carry no paths or tracebacks.

Look and feel: every colour, font and size is in THEME_TOKENS below;
`app_css()` and `app_theme()` turn them into the page CSS and the Gradio
theme, so the app can be restyled without touching the layout code.

The pure pieces (result dict -> HTML, option parsing, the access guard,
feedback saving) need neither Gradio nor the models; Gradio is imported
lazily inside `build_app()` and the Detector inside `ModelSource.load()`.

Run with:  python -m skolfie.app [--models-dir DIR | --bundle DIR] [--auth user:pw]
"""

from __future__ import annotations

import html
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from skolfie.data import feedback as feedback_store
from skolfie.data.file_parsers import MAX_FILE_BYTES, FileParseError, extract_text_from_file

try:  # one source of truth once the service exists
    from skolfie.service import DISCLAIMER, MAX_TEXT_CHARS
except ImportError:  # pragma: no cover - only before skolfie/service.py is present
    MAX_TEXT_CHARS = 100_000
    DISCLAIMER = (
        "AI-text detection is probabilistic. Never use this result as the sole basis for an "
        "academic misconduct decision; talk to the student and look at their other work."
    )

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_PATH = Path("feedback_data.csv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORTS_DIR = Path("reports")

MAX_COMMENT_CHARS = 2_000
THUMBS = ("up", "down")
THUMBS_CHOICES = [("Looks right", "up"), ("Looks wrong", "down")]
# What the teacher picks -> what the feedback CSV stores.
CORRECT_LABELS = {
    "Human": feedback_store.LABEL_HUMAN,
    "AI": feedback_store.LABEL_AI,
    "Mixed": feedback_store.LABEL_MIXED,
}

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
TRUE_VALUES = ("1", "true", "yes", "on")
NO_AUTH_FLAG = "--no-auth-i-know-the-risks"

LABEL_NAMES = {"human": "Human", "ai": "AI", "mixed": "Mixed", "uncertain": "Uncertain",
               "too_short": "Too short"}
WARNING_TITLES = {
    "fairness": "Known limitation",
    "short_text": "Short text",
    "obfuscation": "Hidden characters",
    "language": "Language",
}

NO_MODELS_HINT = (
    "Skolfie needs its trained models to check texts. To get them:\n\n"
    "1. Train them on Colab with the notebooks in `notebooks/` (build_corpus, evaluate, "
    "feature_baseline, transformer, stack, chunks, mixed, explain, fairness). They save to "
    "`My Drive/skolfie/models` and `My Drive/skolfie/reports`.\n"
    "2. Start the app pointing at them:\n\n"
    "```bash\n"
    "python -m skolfie.app --models-dir /path/to/skolfie/models "
    "--reports-dir /path/to/skolfie/reports\n"
    "```\n\n"
    "   or copy just the files the app needs into one portable folder with `skolfie bundle` "
    "(see `skolfie bundle --help`) and start with `python -m skolfie.app --bundle FOLDER`.\n\n"
    "On Colab, `notebooks/app.ipynb` does all of this for you. "
    "After adding the files, press **Reload models** under Model info."
)

# --------------------------------------------------------------------------- #
# Look and feel: all colours, fonts and sizes live here
# --------------------------------------------------------------------------- #
# White and baby blue. The verdict colours (rose for AI, amber for Mixed, green for human)
# stay apart from the brand blue. Every text colour pair meets WCAG AA (4.5:1) on its
# background; tests/test_app.py checks the pairs listed in CONTRAST_PAIRS.
THEME_TOKENS: Dict[str, str] = {
    "font_sans": "Inter",            # Google Fonts; system sans-serif is the fallback
    "font_serif": "Source Serif 4",  # the headline; Georgia is the fallback
    "bg": "#ffffff",
    "text": "#111111",
    "text_muted": "#6b7280",
    "border": "#e5e7eb",
    "sky_50": "#f0f8ff",             # soft panel / section backgrounds
    "sky_100": "#dff0fd",            # hover, chips, subtle fills
    "sky_300": "#a7d8f7",            # borders, focus ring
    "sky_400": "#7cc4f2",            # main brand baby blue (fills, logo mark; never text)
    "sky_400_hover": "#68b8ee",
    "sky_700": "#1f6fa8",            # text and links on white, dark pill buttons
    "sky_700_hover": "#195d8e",
    "on_sky_400": "#0b2a42",         # text on the baby-blue Check button
    "on_sky_700": "#ffffff",
    "ai_rgb": "244, 63, 94",         # AI-looking highlight tint (rose; alpha added per span)
    "ai_bg": "#ffe4e6", "ai_fg": "#9f1239",
    "mixed_bg": "#fef3c7", "mixed_fg": "#92400e",
    "human_bg": "#dcfce7", "human_fg": "#166534",
    "uncertain_bg": "#f3f4f6", "uncertain_fg": "#374151",
    "warn_bg": "#fffbeb", "warn_border": "#fcd34d", "warn_fg": "#92400e",
    "shadow": "0 1px 2px rgba(17, 17, 17, 0.04), 0 8px 24px rgba(17, 17, 17, 0.05)",
    "radius": "20px",
    "radius_small": "12px",
    "max_width": "760px",
    "text_size": "15px",
    "headline_size": "64px",
}

# (foreground token, background token) pairs used for text; each must reach 4.5:1.
CONTRAST_PAIRS = [
    ("text", "bg"), ("text_muted", "bg"), ("text_muted", "sky_50"), ("sky_700", "bg"),
    ("sky_700", "sky_50"), ("sky_700", "sky_100"), ("on_sky_400", "sky_400"),
    ("on_sky_400", "sky_400_hover"), ("on_sky_700", "sky_700"), ("on_sky_700", "sky_700_hover"),
    ("ai_fg", "ai_bg"), ("mixed_fg", "mixed_bg"), ("human_fg", "human_bg"),
    ("uncertain_fg", "uncertain_bg"), ("warn_fg", "warn_bg"), ("text", "warn_bg"),
]

# Highlight strength: an AI span's background alpha runs from MIN to MAX with its score,
# capped so the near-black text on it stays easy to read.
HIGHLIGHT_MIN_ALPHA = 0.10
HIGHLIGHT_MAX_ALPHA = 0.30

GITHUB_URL = "https://github.com/mohaaa825/Skolfie"


def app_head(tokens: Mapping[str, str] = THEME_TOKENS) -> str:
    """<head> additions: the Google Fonts stylesheet for the sans and serif fonts."""
    families = "&".join(
        "family=" + tokens[k].replace(" ", "+") + spec
        for k, spec in (("font_sans", ":wght@400;500;600"),
                        ("font_serif", ":opsz,wght@8..60,400")))
    return ('<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{families}'
            '&display=swap">')


def app_css(tokens: Mapping[str, str] = THEME_TOKENS) -> str:
    """The page CSS: the tokens as CSS variables, then rules that use only those variables."""
    t = tokens
    sans = f'"{t["font_sans"]}", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
    serif = f'"{t["font_serif"]}", Georgia, "Times New Roman", serif'
    return f"""
:root, .dark {{
  --sk-bg: {t['bg']}; --sk-text: {t['text']}; --sk-muted: {t['text_muted']};
  --sk-border: {t['border']}; --sk-sky-50: {t['sky_50']}; --sk-sky-100: {t['sky_100']};
  --sk-sky-300: {t['sky_300']}; --sk-sky-400: {t['sky_400']};
  --sk-sky-400-hover: {t['sky_400_hover']}; --sk-sky-700: {t['sky_700']};
  --sk-sky-700-hover: {t['sky_700_hover']}; --sk-on-sky-400: {t['on_sky_400']};
  --sk-on-sky-700: {t['on_sky_700']}; --sk-ai-rgb: {t['ai_rgb']};
  --sk-ai-bg: {t['ai_bg']}; --sk-ai-fg: {t['ai_fg']};
  --sk-mixed-bg: {t['mixed_bg']}; --sk-mixed-fg: {t['mixed_fg']};
  --sk-human-bg: {t['human_bg']}; --sk-human-fg: {t['human_fg']};
  --sk-uncertain-bg: {t['uncertain_bg']}; --sk-uncertain-fg: {t['uncertain_fg']};
  --sk-warn-bg: {t['warn_bg']}; --sk-warn-border: {t['warn_border']};
  --sk-warn-fg: {t['warn_fg']}; --sk-shadow: {t['shadow']};
  --sk-radius: {t['radius']}; --sk-radius-small: {t['radius_small']};
  --sk-max-width: {t['max_width']}; --sk-text-size: {t['text_size']};
  --sk-headline-size: {t['headline_size']};
  --sk-sans: {sans}; --sk-serif: {serif};
}}
html {{ scroll-behavior: smooth; }}
body, gradio-app, .gradio-container, .main, .wrap, .contain {{
  background: var(--sk-bg) !important; }}
.gradio-container {{ max-width: var(--sk-max-width) !important; margin: 0 auto !important;
  padding: 0 16px 48px !important; font-family: var(--sk-sans) !important;
  font-size: var(--sk-text-size); color: var(--sk-text); }}
/* Quiet Gradio chrome: no footer, no panel backgrounds or boxes around plain blocks. */
footer, .gradio-container > footer {{ display: none !important; }}
.gradio-container .block, .gradio-container .form {{ background: transparent;
  border: none; box-shadow: none; }}
.gradio-container .prose, .gradio-container .prose p {{ font-size: var(--sk-text-size); }}
a {{ color: var(--sk-sky-700); }}

/* Top bar */
.sk-topbar {{ display: flex; align-items: center; justify-content: space-between;
  padding: 20px 0; }}
.sk-logo {{ display: inline-flex; align-items: center; gap: 8px; font-weight: 600;
  font-size: 1.05rem; color: var(--sk-sky-700); text-decoration: none; letter-spacing: -0.01em; }}
.sk-logo-mark {{ width: 14px; height: 14px; border-radius: 50%; background: var(--sk-sky-400);
  display: inline-block; }}
.sk-nav {{ display: flex; align-items: center; gap: 18px; font-size: 0.9rem; }}
.sk-nav a {{ color: var(--sk-muted); text-decoration: none; }}
.sk-nav a:hover {{ color: var(--sk-sky-700); }}
.sk-pill-link {{ display: inline-block; border: 1px solid var(--sk-sky-300); border-radius: 999px;
  padding: 6px 14px; color: var(--sk-sky-700) !important; background: var(--sk-bg); }}
.sk-pill-link:hover {{ background: var(--sk-sky-100); }}

/* Hero */
.sk-hero {{ text-align: center; padding: 56px 0 32px; }}
.sk-hero h1 {{ font-family: var(--sk-serif); font-weight: 400; color: var(--sk-text);
  font-size: clamp(40px, 9vw, var(--sk-headline-size)); line-height: 1.05;
  letter-spacing: -0.02em; margin: 0 0 18px; }}
.sk-hero p {{ color: var(--sk-muted); max-width: 34em; margin: 0 auto; line-height: 1.6; }}

/* Composer: the big rounded input card */
#sk-composer {{ background: var(--sk-bg); border: 1px solid var(--sk-border);
  border-radius: var(--sk-radius); box-shadow: var(--sk-shadow); padding: 14px 14px 12px;
  gap: 8px; }}
#sk-composer:focus-within {{ border-color: var(--sk-sky-300);
  box-shadow: 0 0 0 3px var(--sk-sky-100), var(--sk-shadow); }}
#sk-composer .block, #sk-composer label, #sk-composer .input-container {{
  border: none !important; box-shadow: none !important; outline: none !important; }}
#sk-composer textarea {{ border: none !important; box-shadow: none !important;
  background: transparent !important; font-size: 1rem; line-height: 1.6; padding: 6px 8px;
  color: var(--sk-text); resize: vertical; }}
#sk-composer textarea::placeholder {{ color: var(--sk-muted); }}
.sk-composer-bar {{ align-items: center !important; gap: 10px !important; flex-wrap: nowrap; }}
.sk-composer-bar > * {{ flex: 0 0 auto !important; min-width: 0 !important; }}
.sk-attach-note {{ flex: 1 1 auto !important; color: var(--sk-muted); font-size: 0.85rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.sk-plus {{ width: 36px !important; height: 36px; border-radius: 50% !important;
  border: 1px solid var(--sk-border) !important; background: var(--sk-bg) !important;
  color: var(--sk-text) !important; font-size: 1.25rem !important; padding: 0 !important;
  box-shadow: none !important; }}
.sk-plus:hover {{ background: var(--sk-sky-100) !important; }}
.sk-check {{ margin-left: auto; }}

/* Buttons */
.gradio-container button {{ border-radius: 999px !important; font-weight: 500;
  font-family: var(--sk-sans); }}
.gradio-container button.primary {{ background: var(--sk-sky-400) !important;
  color: var(--sk-on-sky-400) !important; border: none !important; padding: 9px 22px;
  box-shadow: none !important; }}
.gradio-container button.primary:hover {{ background: var(--sk-sky-400-hover) !important; }}
.gradio-container button.secondary {{ background: var(--sk-bg) !important;
  color: var(--sk-sky-700) !important; border: 1px solid var(--sk-sky-300) !important;
  box-shadow: none !important; }}
.gradio-container button.secondary:hover {{ background: var(--sk-sky-100) !important; }}
.gradio-container button.sk-solid {{ background: var(--sk-sky-700) !important;
  color: var(--sk-on-sky-700) !important; border: none !important; }}
.gradio-container button.sk-solid:hover {{ background: var(--sk-sky-700-hover) !important; }}
.gradio-container button:focus-visible {{ outline: 3px solid var(--sk-sky-300);
  outline-offset: 2px; }}

/* Small centred pill that scrolls to "How it works" */
.sk-center {{ text-align: center; margin: 18px 0 8px; }}
.sk-chip {{ display: inline-block; background: var(--sk-sky-50); color: var(--sk-sky-700);
  border: 1px solid var(--sk-sky-100); border-radius: 999px; padding: 5px 14px;
  font-size: 0.85rem; text-decoration: none; }}
.sk-chip:hover {{ background: var(--sk-sky-100); }}

/* Result cards */
.sk-result {{ display: flex; flex-direction: column; gap: 14px; margin-top: 8px; }}
.sk-card {{ background: var(--sk-bg); border: 1px solid var(--sk-border);
  border-radius: var(--sk-radius); padding: 22px 24px; color: var(--sk-text); }}
.sk-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; }}
.sk-badge {{ display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600;
  font-size: 0.95rem; }}
.sk-badge-ai {{ background: var(--sk-ai-bg); color: var(--sk-ai-fg); }}
.sk-badge-mixed {{ background: var(--sk-mixed-bg); color: var(--sk-mixed-fg); }}
.sk-badge-human {{ background: var(--sk-human-bg); color: var(--sk-human-fg); }}
.sk-badge-uncertain, .sk-badge-too_short {{ background: var(--sk-uncertain-bg);
  color: var(--sk-uncertain-fg); }}
.sk-headline {{ font-family: var(--sk-serif); font-size: 1.6rem; font-weight: 400;
  line-height: 1.2; margin: 0; color: var(--sk-text); }}
.sk-meta {{ color: var(--sk-muted); font-size: 0.85rem; margin: 10px 0 0; }}
.sk-meter {{ margin: 18px 0 0; }}
.sk-meter-label {{ display: flex; justify-content: space-between; align-items: baseline;
  color: var(--sk-muted); }}
.sk-meter-label b {{ font-size: 1.5rem; font-weight: 600; color: var(--sk-text);
  font-variant-numeric: tabular-nums; }}
.sk-meter-track {{ height: 6px; background: var(--sk-sky-50); border-radius: 999px;
  overflow: hidden; margin-top: 8px; }}
.sk-meter-fill {{ height: 100%; background: var(--sk-sky-400); border-radius: 999px; }}
.sk-section-title {{ color: var(--sk-muted); font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 10px; }}
.sk-warn-box {{ background: var(--sk-warn-bg); border: 1px solid var(--sk-warn-border);
  border-radius: var(--sk-radius); padding: 18px 22px; color: var(--sk-text); }}
.sk-warn-box .sk-section-title {{ color: var(--sk-warn-fg); }}
.sk-warn-box p {{ margin: 0 0 8px; }}
.sk-section + .sk-section {{ margin-top: 18px; }}
.sk-list {{ margin: 0; padding-left: 1.2em; line-height: 1.6; }}
.sk-list li {{ margin: 4px 0; }}
.sk-text {{ white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.8; font-size: 1rem;
  max-height: 560px; overflow-y: auto; color: var(--sk-text); }}
.sk-ai-span {{ border-radius: 4px; padding: 2px 0; color: var(--sk-text);
  box-decoration-break: clone; -webkit-box-decoration-break: clone; }}
.sk-legend {{ color: var(--sk-muted); font-size: 0.85rem; margin: 0 0 12px; }}
.sk-legend .sk-ai-span {{ padding: 1px 6px; }}
.sk-disclaimer {{ color: var(--sk-muted); font-size: 0.85rem; margin: 2px 4px 0;
  line-height: 1.5; }}
.sk-message {{ color: var(--sk-text); }}
.sk-message-error {{ border-color: var(--sk-ai-bg); background: #fff; }}

/* Accordions (feedback, model info, dataset) as quiet cards */
.gradio-container .sk-panel {{ border: 1px solid var(--sk-border) !important;
  border-radius: var(--sk-radius) !important; background: var(--sk-bg) !important;
  margin-top: 14px; padding: 6px 10px; }}

/* How it works */
.sk-how {{ background: var(--sk-sky-50); border-radius: var(--sk-radius); padding: 32px;
  margin-top: 56px; line-height: 1.65; }}
.sk-how h2 {{ font-family: var(--sk-serif); font-weight: 400; font-size: 2rem;
  margin: 0 0 14px; color: var(--sk-text); letter-spacing: -0.01em; }}
.sk-how h3 {{ font-size: 0.95rem; font-weight: 600; margin: 22px 0 6px; color: var(--sk-text); }}
.sk-how p, .sk-how li {{ color: var(--sk-text); }}

/* Footer */
.sk-footer {{ display: grid; grid-template-columns: 1.2fr repeat(3, 1fr); gap: 24px;
  border-top: 1px solid var(--sk-border); margin-top: 72px; padding: 40px 0 24px;
  font-size: 0.85rem; }}
.sk-footer h4 {{ color: #374151; font-size: 0.85rem; font-weight: 600; margin: 0 0 10px; }}
.sk-footer a, .sk-footer p {{ display: block; color: var(--sk-muted); text-decoration: none;
  margin: 0 0 8px; line-height: 1.5; }}
.sk-footer a:hover {{ color: var(--sk-sky-700); }}
@media (max-width: 640px) {{
  .gradio-container {{ width: 100% !important; min-width: 0 !important;
    box-sizing: border-box; }}
  .sk-composer-bar .sk-attach-note {{ display: none !important; }}
  .sk-topbar .sk-pill-link {{ display: none; }}
  .sk-footer {{ grid-template-columns: 1fr 1fr; }}
  .sk-hero {{ padding-top: 32px; }}
  .sk-card, .sk-how {{ padding: 18px; }}
}}
"""


def app_theme(tokens: Mapping[str, str] = THEME_TOKENS):
    """The Gradio theme built from the tokens (the same in light and dark mode)."""
    import gradio as gr

    t = tokens
    theme = gr.themes.Base(
        font=(gr.themes.GoogleFont(t["font_sans"]), "ui-sans-serif", "system-ui",
              "-apple-system", "Segoe UI", "sans-serif"),
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_lg,
    )
    values = {
        "body_background_fill": t["bg"],
        "background_fill_primary": t["bg"],
        "background_fill_secondary": t["sky_50"],
        "body_text_color": t["text"],
        "body_text_color_subdued": t["text_muted"],
        "block_background_fill": t["bg"],
        "block_border_color": t["border"],
        "block_border_width": "0px",
        "block_label_text_color": t["text_muted"],
        "block_label_background_fill": t["bg"],
        "block_title_text_color": t["text_muted"],
        "block_radius": t["radius"],
        "block_shadow": "none",
        "panel_background_fill": t["bg"],
        "border_color_primary": t["border"],
        "border_color_accent": t["sky_300"],
        "input_background_fill": t["bg"],
        "input_border_color": t["border"],
        "input_border_color_focus": t["sky_300"],
        "input_placeholder_color": t["text_muted"],
        "button_primary_background_fill": t["sky_400"],
        "button_primary_background_fill_hover": t["sky_400_hover"],
        "button_primary_text_color": t["on_sky_400"],
        "button_secondary_background_fill": t["bg"],
        "button_secondary_background_fill_hover": t["sky_100"],
        "button_secondary_text_color": t["sky_700"],
        "button_secondary_border_color": t["sky_300"],
        "checkbox_label_background_fill": t["bg"],
        "checkbox_label_background_fill_hover": t["sky_50"],
        "checkbox_label_background_fill_selected": t["sky_100"],
        "checkbox_label_text_color": t["text"],
        "checkbox_label_text_color_selected": t["text"],
        "checkbox_label_border_color": t["border"],
        "checkbox_background_color_selected": t["sky_700"],
        "checkbox_border_color": t["sky_300"],
        "color_accent": t["sky_400"],
        "color_accent_soft": t["sky_100"],
        "link_text_color": t["sky_700"],
        "table_even_background_fill": t["bg"],
        "table_odd_background_fill": t["sky_50"],
        "shadow_drop": "none",
    }
    # Same values for Gradio's dark mode, so the page stays white whatever the browser prefers.
    theme.set(**values, **{f"{k}_dark": v for k, v in values.items()
                           if hasattr(theme, f"{k}_dark")})
    return theme


TOPBAR_HTML = (
    '<nav class="sk-topbar"><a class="sk-logo" href="#sk-top"><span class="sk-logo-mark">'
    '</span>Skolfie</a><span class="sk-nav"><a href="#how-it-works">How it works</a>'
    '<a class="sk-pill-link" href="#sk-composer">Check an essay</a></span></nav>'
)
HERO_HTML = (
    '<div class="sk-hero" id="sk-top"><h1>Human or AI?<br>Check any essay.</h1>'
    "<p>Paste a student essay or upload a file. Skolfie estimates how likely it is to be "
    "AI-generated, highlights the parts that read as machine-written and explains why.</p></div>"
)
HOW_LINK_HTML = ('<p class="sk-center"><a class="sk-chip" href="#how-it-works">'
                 "How it works ↓</a></p>")
HOW_IT_WORKS_HTML = (
    '<section class="sk-how" id="how-it-works"><h2>How it works</h2>'
    "<p>Skolfie combines two models trained on human and AI-written English texts: one looks "
    "at style (vocabulary variety, sentence rhythm, punctuation, how predictable the wording "
    "is), the other is a language model that reads the text itself. Their scores are combined "
    "into one AI likelihood.</p>"
    "<h3>The verdicts</h3><ul class=\"sk-list\">"
    "<li><b>Human</b> or <b>AI</b>: the combined score is clearly on one side.</li>"
    "<li><b>Uncertain</b>: no clear signal either way.</li>"
    "<li><b>Mixed</b>: the text is also scored in parts of about 60 words; some parts look "
    "AI-generated and others do not. Those parts are highlighted.</li>"
    "<li><b>Too short</b>: under 30 words there is no verdict, and under 150 words results "
    "are less reliable.</li></ul>"
    '<h3 id="limitations">Limitations</h3><ul class="sk-list">'
    "<li>Skolfie is trained on English only.</li>"
    "<li>It flags some groups of human writers more often than others, notably students "
    "writing in a second language. Known cases are shown with each result.</li>"
    "<li>Editing, paraphrasing tools and disguised characters can hide AI text.</li>"
    "<li>A result is a reason to look closer and talk with the student, never proof of "
    "misconduct on its own.</li></ul>"
    "<h3>Privacy</h3><p>The text is checked on the server running Skolfie. It is not saved "
    "unless you send feedback; uploaded files are removed from the server's temporary cache "
    "within an hour.</p></section>"
)
FOOTER_HTML = (
    '<footer class="sk-footer" role="contentinfo"><div><a class="sk-logo" href="#sk-top">'
    '<span class="sk-logo-mark"></span>Skolfie</a></div>'
    '<div><h4>Skolfie</h4><a href="#how-it-works">How it works</a>'
    '<a href="#limitations">Limitations</a><a href="#model-info">Model info</a></div>'
    "<div><h4>Privacy</h4><p>Your text is not stored unless you send feedback.</p></div>"
    f'<div><h4>Project</h4><a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>'
    "</div></footer>"
)


# --------------------------------------------------------------------------- #
# Result dict -> HTML (pure functions; everything that is text is escaped)
# --------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _finite(p: Any) -> Optional[float]:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if math.isfinite(p) else None


def format_percent(p: Any) -> str:
    """0.834 -> "83%"; None / NaN -> "n/a"."""
    p = _finite(p)
    return "n/a" if p is None else f"{p:.0%}"


def badge_html(label: str) -> str:
    """The coloured headline badge: Human / AI / Mixed / Uncertain / Too short."""
    key = label if label in LABEL_NAMES else "uncertain"
    name = LABEL_NAMES.get(label, str(label).replace("_", " ").title() or "Unknown")
    return f'<span class="sk-badge sk-badge-{key}">{_esc(name)}</span>'


def likelihood_html(p_ai: Any) -> str:
    """The AI likelihood as a large number with a thin meter; a note when not scored."""
    p = _finite(p_ai)
    if p is None:
        return ('<div class="sk-meter"><div class="sk-meter-label"><span>AI likelihood</span>'
                "<span>not scored</span></div></div>")
    width = f"{min(1.0, max(0.0, p)) * 100:.1f}%"
    return ('<div class="sk-meter"><div class="sk-meter-label"><span>AI likelihood</span>'
            f"<b>{format_percent(p)}</b></div>"
            f'<div class="sk-meter-track"><div class="sk-meter-fill" style="width: {width}">'
            "</div></div></div>")


def _ai_alpha(p: Optional[float]) -> float:
    """Background alpha of an AI span: stronger the more confident the score."""
    if p is None:
        return HIGHLIGHT_MIN_ALPHA
    strength = min(1.0, max(0.0, (p - 0.5) / 0.5))
    return HIGHLIGHT_MIN_ALPHA + (HIGHLIGHT_MAX_ALPHA - HIGHLIGHT_MIN_ALPHA) * strength


def _span_html(text: str, p: Optional[float]) -> str:
    tip = f"AI score {format_percent(p)}"
    return (f'<mark class="sk-ai-span" style="background: rgba(var(--sk-ai-rgb), '
            f'{_ai_alpha(p):.2f})" title="{_esc(tip)}">{_esc(text)}</mark>')


def ai_spans(result: Mapping[str, Any]) -> List[dict]:
    """The result's AI-looking spans: `highlights` labelled "ai" (else `chunks` labelled "ai")."""
    spans = result.get("highlights") or result.get("chunks") or []
    return [s for s in spans if isinstance(s, Mapping) and s.get("label") == "ai"]


def highlighted_text_html(text: str, spans: Sequence[Mapping[str, Any]]) -> str:
    """The text with the given spans (character offsets) marked; overlaps are clipped."""
    parts, pos = [], 0
    for s in sorted(spans, key=lambda s: (int(s.get("start", 0)), int(s.get("end", 0)))):
        start = max(pos, int(s.get("start", 0)))
        end = min(len(text), int(s.get("end", 0)))
        if end <= start:
            continue
        parts.append(_esc(text[pos:start]))
        parts.append(_span_html(text[start:end], _finite(s.get("p"))))
        pos = end
    parts.append(_esc(text[pos:]))
    return f'<div class="sk-text">{"".join(parts)}</div>'


def _warning_parts(warnings: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(fairness messages, [(kind, message)] for the rest), tolerating plain strings."""
    fairness, other = [], []
    for w in warnings or []:
        if isinstance(w, Mapping):
            kind, message = str(w.get("kind") or ""), w.get("message")
        else:
            kind, message = "", w
        if not message:
            continue
        if kind == "fairness":
            fairness.append(str(message))
        else:
            other.append((kind, str(message)))
    return fairness, other


def warnings_html(warnings: Any) -> str:
    """Fairness warnings first in a prominent box, then the other warnings."""
    fairness, other = _warning_parts(warnings)
    out = []
    if fairness:
        out.append(
            '<div class="sk-warn-box sk-fairness"><p class="sk-section-title">'
            "Before you act on this: known limitations</p>"
            "<p>Skolfie wrongly flags some groups of human writers more often than others. "
            "Keep this in mind, especially for students writing in a second language.</p>"
            '<ul class="sk-list">'
            + "".join(f"<li>{_esc(m)}</li>" for m in fairness)
            + "</ul></div>")
    if other:
        out.append('<div class="sk-warn-box"><p class="sk-section-title">Warnings</p>'
                   '<ul class="sk-list">'
                   + "".join(f"<li><b>{_esc(WARNING_TITLES.get(k, 'Note'))}:</b> {_esc(m)}</li>"
                             for k, m in other)
                   + "</ul></div>")
    return "".join(out)


def list_section_html(title: str, items: Any) -> str:
    """A titled bullet list of plain-text items; "" when there are none."""
    items = [str(i) for i in (items or []) if i]
    if not items:
        return ""
    return (f'<div class="sk-section"><p class="sk-section-title">{_esc(title)}</p>'
            '<ul class="sk-list">' + "".join(f"<li>{_esc(i)}</li>" for i in items)
            + "</ul></div>")


def text_section_html(result: Mapping[str, Any], text: str) -> str:
    """The text with its AI-looking parts highlighted, with a legend line."""
    if result.get("label") == "too_short" or not text:
        return ""
    spans = ai_spans(result)
    has_chunks = bool(result.get("highlights") or result.get("chunks"))
    if spans:
        legend = ('<p class="sk-legend"><mark class="sk-ai-span" style="background: '
                  f'rgba(var(--sk-ai-rgb), {HIGHLIGHT_MAX_ALPHA:.2f})">Highlighted</mark> '
                  "parts read as AI-generated; the stronger the colour, the stronger the "
                  "signal. Hover a part for its score.</p>")
    elif has_chunks:
        legend = '<p class="sk-legend">No part of the text stands out as AI-generated.</p>'
    else:
        legend = '<p class="sk-legend">This text was not scored part by part.</p>'
    return ('<div class="sk-section"><p class="sk-section-title">The text</p>'
            f"{legend}{highlighted_text_html(text, spans)}</div>")


def render_result_html(result: Mapping[str, Any], text: str) -> str:
    """The result cards for one `Detector.check()` result on `text`: verdict and likelihood,
    warnings (fairness first), the highlighted text, reasons and notes, the disclaimer."""
    label = str(result.get("label") or "uncertain")
    n_words = result.get("n_words")
    meta = f"{int(n_words):,} words" if isinstance(n_words, (int, float)) else ""
    verdict = ('<div class="sk-card sk-verdict"><div class="sk-head">'
               f"{badge_html(label)}"
               f'<p class="sk-headline">{_esc(result.get("headline") or "")}</p></div>'
               + (f'<p class="sk-meta">{_esc(meta)}</p>' if meta else "")
               + (likelihood_html(result.get("p_ai")) if label != "too_short" else "")
               + "</div>")
    text_part = text_section_html(result, text)
    why = (list_section_html("Why", result.get("reasons"))
           + list_section_html("Notes", result.get("notes")))
    cards = [
        verdict,
        warnings_html(result.get("warnings")),
        f'<div class="sk-card">{text_part}</div>' if text_part else "",
        f'<div class="sk-card">{why}</div>' if why else "",
        f'<p class="sk-disclaimer">{_esc(result.get("disclaimer") or DISCLAIMER)}</p>',
    ]
    return '<div class="sk-result">' + "".join(c for c in cards if c) + "</div>"


def message_html(message: str, error: bool = False) -> str:
    """A plain message card (input problems, missing models...). The message is escaped."""
    cls = "sk-card sk-message" + (" sk-message-error" if error else "")
    return f'<div class="{cls}">{_esc(message)}</div>'


# --------------------------------------------------------------------------- #
# Model location and loading (no Gradio needed)
# --------------------------------------------------------------------------- #
@dataclass
class ModelSource:
    """Where the Detector's files are: a bundle folder, or models (+ reports) folders."""

    bundle: Optional[Path] = None
    models_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None

    def describe(self) -> str:
        if self.bundle is not None:
            return f"bundle `{self.bundle}`"
        reports = f", reports `{self.reports_dir}`" if self.reports_dir else ", no reports folder"
        return f"models `{self.models_dir}`{reports}"

    def load(self) -> Any:
        from skolfie.service import Detector

        if self.bundle is not None:
            return Detector.from_bundle(self.bundle)
        return Detector.from_dirs(self.models_dir, self.reports_dir)


def resolve_model_source(bundle: Optional[os.PathLike | str] = None,
                         models_dir: Optional[os.PathLike | str] = None,
                         reports_dir: Optional[os.PathLike | str] = None,
                         environ: Optional[Mapping[str, str]] = None) -> ModelSource:
    """
    Command-line values first (a bundle, else a models folder), then
    $SKOLFIE_BUNDLE, then $SKOLFIE_MODELS_DIR, then ./models. The reports
    folder is --reports-dir, else $SKOLFIE_REPORTS_DIR, else a `reports`
    folder next to the models folder when there is one.
    """
    env = os.environ if environ is None else environ
    if bundle and models_dir:
        raise ValueError("pass either a bundle or a models folder, not both")
    if bundle:
        return ModelSource(bundle=Path(bundle))
    if not models_dir and env.get("SKOLFIE_BUNDLE"):
        return ModelSource(bundle=Path(env["SKOLFIE_BUNDLE"]))
    models = Path(models_dir or env.get("SKOLFIE_MODELS_DIR") or DEFAULT_MODELS_DIR)
    reports: Optional[Path] = None
    if reports_dir or env.get("SKOLFIE_REPORTS_DIR"):
        reports = Path(reports_dir or env["SKOLFIE_REPORTS_DIR"])
    elif (models.parent / DEFAULT_REPORTS_DIR.name).is_dir():
        reports = models.parent / DEFAULT_REPORTS_DIR.name
    return ModelSource(models_dir=models, reports_dir=reports)


def _md_code(value: Any) -> str:
    """Inline code for Markdown that cannot be broken out of by backticks in the value."""
    s = " ".join(str(value).split()).replace("`", "'")
    return f"`{s}`"


def format_info_markdown(info: Optional[Mapping[str, Any]]) -> str:
    """A short summary of `Detector.info()` (the full dict is shown as JSON next to it)."""
    if not info:
        return ""
    lines: List[str] = []
    if info.get("source"):
        lines += [f"**Source:** {_md_code(info['source'])}", ""]
    if "mixed_enabled" in info:
        lines.append(f"**Mixed detection:** {'on' if info['mixed_enabled'] else 'off'}  ")
    if isinstance(info.get("known_limitations"), (int, float)):
        lines.append(f"**Known fairness limitations shown with AI / Mixed results:** "
                     f"{int(info['known_limitations'])}")
    if lines and lines[-1]:
        lines.append("")
    loaded = info.get("loaded") or info.get("files")
    if isinstance(loaded, Mapping):
        lines.append("**Loaded:**")
        lines += [f"- {_md_code(k)}: {_md_code(v)}" for k, v in loaded.items()]
    elif isinstance(loaded, (list, tuple)):
        lines.append("**Loaded:**")
        lines += [f"- {_md_code(v)}" for v in loaded]
    missing = info.get("missing")
    if missing:
        lines.append("")
        lines.append("**Missing (optional, skipped):**")
        if isinstance(missing, Mapping):
            lines += [f"- {_md_code(k)}: {_md_code(v)}" for k, v in missing.items()]
        elif isinstance(missing, (list, tuple)):
            lines += [f"- {_md_code(v)}" for v in missing]
        else:
            lines.append(f"- {_md_code(missing)}")
    cutoffs = info.get("cutoffs")
    if isinstance(cutoffs, Mapping) and cutoffs:
        lines.append("")
        lines.append("**Cut-offs:** " + ", ".join(f"{_md_code(k)} {_md_code(v)}"
                                                    for k, v in cutoffs.items()))
    return "\n".join(lines)


@dataclass
class DetectorHolder:
    """Holds the loaded Detector (or why there isn't one); checks run one at a time."""

    source: ModelSource = field(default_factory=resolve_model_source)
    detector: Any = None
    status: str = ""
    loader: Optional[Callable[[ModelSource], Any]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> str:
        """(Re)load the Detector; returns the Markdown status for the page."""
        try:
            detector = (self.loader or ModelSource.load)(self.source)
        except FileNotFoundError as exc:
            with self._lock:
                self.detector = None
            self.status = (f"**No models found** ({self.source.describe()}).\n\n"
                           f"{exc}\n\n{NO_MODELS_HINT}")
            return self.status
        except ImportError as exc:
            logger.exception("loading the detector failed")
            with self._lock:
                self.detector = None
            self.status = (f"**Could not load the models:** a package is missing "
                           f"(`{getattr(exc, 'name', None) or 'unknown'}`). Install them with "
                           '`pip install -e ".[features,app]"` and restart.')
            return self.status
        except Exception as exc:  # corrupt file, version mismatch...
            logger.exception("loading the detector failed")
            with self._lock:
                self.detector = None
            self.status = (f"**Could not load the models** from {self.source.describe()} "
                           f"(`{type(exc).__name__}`; see the server log). Re-run the notebook "
                           "that saved them, or make a fresh `skolfie bundle`.")
            return self.status
        with self._lock:
            self.detector = detector
        stamp = datetime.now().isoformat(timespec="seconds")
        self.status = f"Models loaded from {self.source.describe()} at {stamp}."
        return self.status

    @property
    def ready(self) -> bool:
        return self.detector is not None

    def info(self) -> Dict[str, Any]:
        if not self.ready:
            return {}
        try:
            return dict(self.detector.info())
        except Exception:
            logger.exception("Detector.info() failed")
            return {}

    def check(self, text: str) -> Dict[str, Any]:
        with self._lock:
            if self.detector is None:
                raise RuntimeError("no detector loaded")
            return self.detector.check(text)


# --------------------------------------------------------------------------- #
# Handlers (no Gradio needed)
# --------------------------------------------------------------------------- #
def feedback_path() -> Path:
    """Feedback CSV location: $SKOLFIE_FEEDBACK if set, else feedback_data.csv."""
    return Path(os.environ.get("SKOLFIE_FEEDBACK") or DEFAULT_FEEDBACK_PATH)


def resolve_input_text(text: Optional[str], file_path: Optional[str]) -> str:
    """An uploaded file wins over pasted text."""
    if file_path:
        return extract_text_from_file(file_path)
    return text or ""


def check_text(holder: DetectorHolder, text: Optional[str], file_path: Optional[str]
               ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Run the Detector. Returns (result HTML, state); `state` carries what the
    feedback form needs and is None when nothing was checked.
    """
    if not holder.ready:
        return message_html("No models are loaded, so texts cannot be checked yet. See the "
                            "Model info section for how to get them.", error=True), None
    try:
        content = resolve_input_text(text, file_path)
    except FileParseError as exc:
        return message_html(f"Could not read the file: {exc}", error=True), None
    except Exception:
        logger.exception("reading the uploaded file failed")
        return message_html("Could not read the file. Is it a valid .txt, .docx or .pdf?",
                            error=True), None
    if len(content) > MAX_TEXT_CHARS:
        return message_html(f"The text is too long ({len(content):,} characters); the limit "
                            f"is {MAX_TEXT_CHARS:,}. Please check a shorter part.",
                            error=True), None
    if not content.strip():
        return message_html("Paste a text or upload a file first."), None
    try:
        result = holder.check(content)
    except Exception:
        logger.exception("checking the text failed")
        return message_html("Skolfie could not check this text. Please try again.",
                            error=True), None
    state = {
        "text": content,
        "label": result.get("label"),
        "p_ai": _finite(result.get("p_ai")),
        "features": {f["feature"]: f.get("value") for f in result.get("top_features") or []
                     if isinstance(f, Mapping) and "feature" in f} or None,
    }
    return render_result_html(result, content), state


def submit_feedback(state: Optional[Dict[str, Any]], thumbs: Optional[str],
                    correct_label: Optional[str], comment: Optional[str],
                    csv_path: Optional[Path] = None) -> str:
    """Append one feedback row for the last check. `correct_label` is Human / AI / Mixed."""
    if not state:
        return "Check a text first, then leave feedback."
    if not thumbs and not correct_label and not (comment or "").strip():
        return "Pick looks right / wrong, a correct label, or write a comment."
    if thumbs and thumbs not in THUMBS:
        return "Choose looks right or looks wrong."
    user_label = None
    if correct_label:
        user_label = CORRECT_LABELS.get(correct_label)
        if user_label is None and correct_label in feedback_store.USER_LABELS:
            user_label = correct_label
        if user_label is None:
            return "Choose a valid correct label."
    if len(comment or "") > MAX_COMMENT_CHARS:
        return f"Please keep the comment under {MAX_COMMENT_CHARS:,} characters."
    path = Path(csv_path or feedback_path())
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    feedback_store.save_feedback(
        csv_path=os.fspath(path),
        text=state["text"],
        predicted_label=state.get("label") or "",
        predicted_confidence="",
        ai_probability=state.get("p_ai"),
        user_label=user_label,
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
    stored = CORRECT_LABELS.get(label or "") or (
        label if label in feedback_store.USER_LABELS else None)
    if stored is None:
        return "Choose a label."
    feedback_store.add_manual_label(
        os.fspath(csv_path or feedback_path()), text.strip(), stored, comment or ""
    )
    return f"Added a {label} example."


# --------------------------------------------------------------------------- #
# Launch options: auth, exposure guard, feedback tab gating
# --------------------------------------------------------------------------- #
def parse_auth(value: Optional[str]) -> Optional[Tuple[str, str]]:
    """ "user:password" -> ("user", "password"); None / "" -> None. The password may hold ':'."""
    if not value:
        return None
    user, sep, password = value.partition(":")
    if not sep or not user.strip() or not password:
        raise ValueError("auth must look like user:password (both non-empty)")
    return user.strip(), password


def env_flag(name: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(name, "")).strip().lower() in TRUE_VALUES


def is_exposed(host: str, share: bool) -> bool:
    """True when people other than this machine's user can reach the app."""
    return bool(share) or host not in LOCAL_HOSTS


def exposure_error(host: str, share: bool, auth: Optional[Tuple[str, str]],
                   allow_no_auth: bool) -> Optional[str]:
    """Why the app must not start like this, or None when it may."""
    if not is_exposed(host, share) or auth is not None or allow_no_auth:
        return None
    where = "a public share link" if share else f"host {host}"
    return (f"refusing to serve on {where} without a password: anyone who finds the address "
            "could use the app. Pass "
            "--auth user:password (or set SKOLFIE_APP_AUTH), or "
            f"{NO_AUTH_FLAG} if that is really what you want.")


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m skolfie.app",
        description="Launch the Skolfie web app (for teachers checking essays).")
    where = parser.add_argument_group("models (operator only; never chosen in the UI)")
    where.add_argument("--bundle", help="folder made by `skolfie bundle` [$SKOLFIE_BUNDLE]")
    where.add_argument("--models-dir", help="models folder [$SKOLFIE_MODELS_DIR, ./models]")
    where.add_argument("--reports-dir",
                       help="reports folder [$SKOLFIE_REPORTS_DIR, a reports/ folder next to "
                            "the models folder]")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio share link (needs --auth)")
    parser.add_argument("--auth", metavar="USER:PASSWORD",
                        help="require this login [$SKOLFIE_APP_AUTH]; prefer the env variable, "
                             "a command line is visible to other users of the machine")
    parser.add_argument(NO_AUTH_FLAG, dest="no_auth_ok", action="store_true",
                        help="allow --share / a non-local --host without a password")
    parser.add_argument("--show-feedback", action="store_true",
                        help="add the Feedback & Dataset tab that lists every stored text "
                             "[$SKOLFIE_SHOW_FEEDBACK=1]")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        auth = parse_auth(args.auth or os.environ.get("SKOLFIE_APP_AUTH"))
        source = resolve_model_source(args.bundle, args.models_dir, args.reports_dir)
    except ValueError as exc:
        parser.error(str(exc))
    error = exposure_error(args.host, args.share, auth, args.no_auth_ok)
    if error:
        parser.error(error)
    show_feedback = args.show_feedback or env_flag("SKOLFIE_SHOW_FEEDBACK")
    if is_exposed(args.host, args.share) and auth is None:
        print("[warning] the app is reachable beyond this machine WITHOUT a password.")
    if show_feedback and is_exposed(args.host, args.share):
        print("[warning] the Feedback & Dataset tab is on: everyone who can log in can read "
              "and export every stored text.")

    holder = DetectorHolder(source=source)
    print(f"[skolfie] {holder.load().splitlines()[0]}")
    demo = build_app(holder, show_feedback=show_feedback)
    launch_kwargs = dict(getattr(demo, "skolfie_launch_kwargs", None) or {})
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                auth=auth, auth_message="Skolfie: please log in." if auth else None,
                max_file_size=MAX_FILE_BYTES, **launch_kwargs)


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #
# Uploaded files Gradio copied to its cache are deleted this long after upload (seconds).
UPLOAD_CACHE_SECONDS = 3600


def attach_note_html(file_path: Optional[str]) -> str:
    """The note next to the "+" button: the attached file's name (escaped), or a hint."""
    if not file_path:
        return '<span class="sk-attach-note">Upload a .txt, .docx or .pdf with +</span>'
    return (f'<span class="sk-attach-note">Attached: <b>{_esc(Path(file_path).name)}</b> '
            "(checked instead of the pasted text)</span>")


def _style_kwargs(gr) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(Blocks kwargs, launch kwargs) for theme, CSS and fonts: Gradio 6 takes them in launch()."""
    import inspect

    style = {"theme": app_theme(), "css": app_css(), "head": app_head()}
    if "css" in inspect.signature(gr.Blocks.launch).parameters:
        return {}, style
    return style, {}


def build_app(holder: Optional[DetectorHolder] = None, *, show_feedback: bool = False):
    """
    Build and return the gr.Blocks app (does not launch it). One page: top bar, hero,
    composer, results, feedback, How it works, Model info, the Feedback & Dataset panel
    (only with `show_feedback`) and the footer. Theme, CSS and fonts are in
    `demo.skolfie_launch_kwargs` for Gradio versions that take them in launch().
    """
    import gradio as gr

    holder = holder or DetectorHolder()
    if not holder.ready and not holder.status:
        holder.load()
    blocks_style, launch_style = _style_kwargs(gr)
    blocks_kwargs: Dict[str, Any] = dict(blocks_style)
    try:
        import inspect

        if "delete_cache" in inspect.signature(gr.Blocks.__init__).parameters:
            blocks_kwargs["delete_cache"] = (UPLOAD_CACHE_SECONDS, UPLOAD_CACHE_SECONDS)
    except (TypeError, ValueError):  # pragma: no cover - unusual Gradio builds
        pass

    with gr.Blocks(title="Skolfie: human or AI?", **blocks_kwargs) as demo:
        gr.HTML(TOPBAR_HTML)
        gr.HTML(HERO_HTML)
        model_banner = gr.Markdown("" if holder.ready else holder.status,
                                   visible=not holder.ready, elem_classes="sk-panel")
        last_result = gr.State(None)
        file_state = gr.State(None)

        with gr.Column(elem_id="sk-composer"):
            text_in = gr.Textbox(
                show_label=False, container=False, lines=8, max_lines=24,
                max_length=MAX_TEXT_CHARS, placeholder="Paste an essay…",
            )
            with gr.Row(elem_classes="sk-composer-bar"):
                upload_btn = gr.UploadButton(
                    "+", file_types=[".txt", ".docx", ".pdf"], type="filepath",
                    elem_classes="sk-plus", size="sm", scale=0, min_width=36,
                )
                attach_note = gr.HTML(attach_note_html(None), elem_classes="sk-attach-note")
                remove_btn = gr.Button("Remove file", size="sm", visible=False, scale=0,
                                       min_width=0, variant="secondary")
                check_btn = gr.Button("Check", variant="primary", elem_classes="sk-check",
                                      scale=0, min_width=96)
        gr.HTML(HOW_LINK_HTML)
        result_out = gr.HTML()

        with gr.Accordion("Was this right? Leave feedback", open=False,
                          elem_classes="sk-panel"):
            gr.Markdown(
                "Feedback stores this text, the result and your comment on the server "
                "running Skolfie, so the models can be improved. Only send it for texts "
                "you may share that way.")
            with gr.Row():
                thumbs_in = gr.Radio(THUMBS_CHOICES, label="The result")
                correct_in = gr.Radio(list(CORRECT_LABELS), label="It was actually")
            comment_in = gr.Textbox(label="Comment (optional)", lines=2,
                                    max_length=MAX_COMMENT_CHARS)
            feedback_btn = gr.Button("Send feedback", elem_classes="sk-solid", scale=0)
            feedback_status = gr.Markdown()

        gr.HTML(HOW_IT_WORKS_HTML)

        with gr.Accordion("Model info", open=False, elem_id="model-info",
                          elem_classes="sk-panel"):
            model_status = gr.Markdown(holder.status)
            info_md = gr.Markdown(format_info_markdown(holder.info()))
            info_json = gr.JSON(holder.info() or None, label="Detector.info()")
            reload_btn = gr.Button("Reload models", elem_classes="sk-solid", scale=0)

        feedback_table = None
        if show_feedback:
            with gr.Accordion("Feedback & Dataset", open=False, elem_id="feedback-dataset",
                              elem_classes="sk-panel"):
                gr.Markdown("Every feedback submission, newest last. It holds student texts: "
                            "export and share it only where that is allowed.")
                refresh_btn = gr.Button("Refresh", scale=0)
                feedback_table = gr.Dataframe(label="Collected feedback", interactive=False,
                                              wrap=True)
                with gr.Row():
                    export_btn = gr.Button("Export as CSV", elem_classes="sk-solid", scale=0)
                    export_file = gr.File(label="Exported dataset", interactive=False)
                gr.Markdown("**Add a labeled example by hand**")
                manual_text = gr.Textbox(label="Text", lines=6, max_length=MAX_TEXT_CHARS)
                manual_label = gr.Radio(list(CORRECT_LABELS), label="Label")
                manual_comment = gr.Textbox(label="Comment (optional)",
                                            max_length=MAX_COMMENT_CHARS)
                manual_btn = gr.Button("Add example", scale=0)
                manual_status = gr.Markdown()

        gr.HTML(FOOTER_HTML)

        # --- wiring -------------------------------------------------------
        def _attach(path):
            return path, attach_note_html(path), gr.update(visible=bool(path))

        upload_btn.upload(_attach, inputs=[upload_btn],
                          outputs=[file_state, attach_note, remove_btn])
        remove_btn.click(lambda: _attach(None), outputs=[file_state, attach_note, remove_btn])

        def _check(text, file):
            return check_text(holder, text, file)

        check_btn.click(_check, inputs=[text_in, file_state], outputs=[result_out, last_result])
        feedback_btn.click(
            lambda state, thumbs, correct, comment: submit_feedback(state, thumbs, correct,
                                                                    comment),
            inputs=[last_result, thumbs_in, correct_in, comment_in], outputs=[feedback_status],
        )

        if feedback_table is not None:
            refresh_btn.click(lambda: load_feedback_table(), outputs=[feedback_table])
            export_btn.click(lambda: export_feedback_file(), outputs=[export_file])

            def _manual(text, label, comment):
                return add_manual_example(text, label, comment), load_feedback_table()

            manual_btn.click(_manual, inputs=[manual_text, manual_label, manual_comment],
                             outputs=[manual_status, feedback_table])
            demo.load(lambda: load_feedback_table(), outputs=[feedback_table])

        def _reload():
            status = holder.load()
            info = holder.info()
            banner = gr.update(value="" if holder.ready else status, visible=not holder.ready)
            return status, format_info_markdown(info), info or None, banner

        reload_btn.click(_reload, outputs=[model_status, info_md, info_json, model_banner])

    demo.skolfie_launch_kwargs = launch_style
    return demo


if __name__ == "__main__":
    main()
