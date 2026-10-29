"""
skolfie/data/feedback.py
------------------------
Handles reading/writing user feedback to feedback_data.csv.

Every piece of feedback given in the app (or manually added via the
"Feedback & Dataset" tab) is appended as a row here. Over time this file
becomes a growing, real-world labeled dataset that can be used to retrain
the model (see scripts/train.py).

Corrected labels (`user_label`) use the vocabulary in `USER_LABELS`:
"Human-Written", "AI-Generated" and "Mixed" (part human, part AI). The
binary training script (scripts/train.py) only learns from the first two
and skips Mixed rows; they are kept for evaluating the Mixed detector.

The CSV holds user-supplied text verbatim, so it is the training source of
truth. `export_feedback` writes the copy meant for spreadsheets: there,
cells that Excel/Sheets would run as a formula (starting with = + - @) are
prefixed with an apostrophe.
"""

import csv
import io
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

FEEDBACK_COLUMNS = [
    "text", "predicted_label", "predicted_confidence", "ai_probability",
    "user_label", "thumbs", "comment", "timestamp", "features",
]
_STRING_COLUMNS = [c for c in FEEDBACK_COLUMNS if c != "ai_probability"]

LABEL_HUMAN = "Human-Written"
LABEL_AI = "AI-Generated"
LABEL_MIXED = "Mixed"
USER_LABELS = (LABEL_HUMAN, LABEL_AI, LABEL_MIXED)
_IMPLIED_AI_PROBABILITY = {LABEL_AI: 1.0, LABEL_HUMAN: 0.0}
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Gradio runs handlers in worker threads: serialise header creation + appends.
_WRITE_LOCK = threading.Lock()


def _csv_line(values) -> bytes:
    """One CSV record as UTF-8 bytes (unencodable lone surrogates become '?')."""
    buf = io.StringIO()
    csv.writer(buf).writerow(values)
    return buf.getvalue().encode("utf-8", errors="replace")


def _append_bytes(csv_path: str, data: bytes, exclusive: bool = False) -> None:
    """Write `data` with a single write() on an O_APPEND descriptor."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    if exclusive:
        flags |= os.O_EXCL
    fd = os.open(csv_path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def _ensure_csv_exists(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return
    try:
        _append_bytes(csv_path, _csv_line(FEEDBACK_COLUMNS), exclusive=True)
    except FileExistsError:
        pass  # another writer created it first


def _clean_cell(value: Any) -> str:
    # NUL characters make pandas' CSV reader silently truncate the cell.
    return str(value).replace("\x00", "")


def save_feedback(
    csv_path: str,
    text: str,
    predicted_label: str,
    predicted_confidence: str,
    ai_probability: Optional[float],
    user_label: Optional[str] = None,
    thumbs: Optional[str] = None,
    comment: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None,
):
    """Append a single feedback row to the CSV file, creating it if needed."""
    row = {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_confidence": predicted_confidence,
        "ai_probability": round(float(ai_probability), 4) if ai_probability is not None else "",
        "user_label": user_label or "",
        "thumbs": thumbs or "",
        "comment": comment or "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": json.dumps(features) if features else "",
    }
    line = _csv_line([_clean_cell(row[c]) for c in FEEDBACK_COLUMNS])

    with _WRITE_LOCK:
        _ensure_csv_exists(csv_path)
        _append_bytes(csv_path, line)


def load_feedback(csv_path: str) -> pd.DataFrame:
    """
    Load all feedback rows as a DataFrame. Returns an empty frame if none yet.

    Text columns stay strings: a text or comment such as "NA", "null" or
    "12345" is kept as written; only empty cells become NaN.
    """
    with _WRITE_LOCK:
        _ensure_csv_exists(csv_path)
    try:
        df = pd.read_csv(
            csv_path, keep_default_na=False, na_values=[""],
            dtype={c: str for c in _STRING_COLUMNS},
        )
        if df.empty and list(df.columns) != FEEDBACK_COLUMNS:
            return pd.DataFrame(columns=FEEDBACK_COLUMNS)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def implied_ai_probability(label: Optional[str]) -> Optional[float]:
    """1.0 for an AI label, 0.0 for a human one, None for Mixed (no single probability)."""
    return _IMPLIED_AI_PROBABILITY.get(label or "")


def add_manual_label(csv_path: str, text: str, label: str, comment: str = ""):
    """
    Manually add a labeled text straight into the dataset, bypassing the
    "analyze first" flow. Useful for bulk-curating training data from the
    Feedback & Dataset tab. `label` is one of `USER_LABELS`.
    """
    if label not in USER_LABELS:
        raise ValueError(f"label must be one of {', '.join(USER_LABELS)}")
    save_feedback(
        csv_path=csv_path,
        text=text,
        predicted_label="",
        predicted_confidence="",
        ai_probability=implied_ai_probability(label),
        user_label=label,
        thumbs="",
        comment=comment,
        features=None,
    )


def escape_formula(value: Any) -> Any:
    """Prefix a string that a spreadsheet would evaluate as a formula with '."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def export_feedback(csv_path: str, export_path: str, escape_formulas: bool = True) -> str:
    """
    Copy the current feedback dataset to a new path (for download/export).

    With `escape_formulas` (the default) string cells starting with
    = + - @ (or a tab / carriage return) get a leading apostrophe so that
    opening the export in Excel or Google Sheets cannot run user-supplied
    formulas. Pass False for a byte-faithful copy for retraining.
    """
    df = load_feedback(csv_path)
    if escape_formulas:
        for col in _STRING_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map(escape_formula)
    df.to_csv(export_path, index=False)
    return export_path
