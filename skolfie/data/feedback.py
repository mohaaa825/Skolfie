"""
skolfie/data/feedback.py
------------------------
Handles reading/writing user feedback to feedback_data.csv.

Every piece of feedback given in the app (or manually added via the
"Feedback & Dataset" tab) is appended as a row here. Over time this file
becomes a growing, real-world labeled dataset that can be used to retrain
the model (see train.py).
"""

import os
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

FEEDBACK_COLUMNS = [
    "text", "predicted_label", "predicted_confidence", "ai_probability",
    "user_label", "thumbs", "comment", "timestamp", "features",
]


def _ensure_csv_exists(csv_path: str):
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(csv_path, index=False)


def save_feedback(
    csv_path: str,
    text: str,
    predicted_label: str,
    predicted_confidence: str,
    ai_probability: float,
    user_label: Optional[str] = None,
    thumbs: Optional[str] = None,
    comment: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None,
):
    """Append a single feedback row to the CSV file, creating it if needed."""
    _ensure_csv_exists(csv_path)

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

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        writer.writerow(row)


def load_feedback(csv_path: str) -> pd.DataFrame:
    """Load all feedback rows as a DataFrame. Returns an empty frame if none yet."""
    _ensure_csv_exists(csv_path)
    try:
        df = pd.read_csv(csv_path)
        if df.empty and list(df.columns) != FEEDBACK_COLUMNS:
            return pd.DataFrame(columns=FEEDBACK_COLUMNS)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)


def add_manual_label(csv_path: str, text: str, label: str, comment: str = ""):
    """
    Manually add a labeled text straight into the dataset, bypassing the
    "analyze first" flow. Useful for bulk-curating training data from the
    Feedback & Dataset tab.
    """
    implied_prob = 1.0 if label == "AI-Generated" else 0.0
    save_feedback(
        csv_path=csv_path,
        text=text,
        predicted_label="",
        predicted_confidence="",
        ai_probability=implied_prob,
        user_label=label,
        thumbs="",
        comment=comment,
        features=None,
    )


def export_feedback(csv_path: str, export_path: str) -> str:
    """Copy the current feedback dataset to a new path (for download/export)."""
    df = load_feedback(csv_path)
    df.to_csv(export_path, index=False)
    return export_path
