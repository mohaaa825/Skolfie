"""
skolfie/eval/length_match.py
----------------------------
Length matching for training (added 2026-11-20).

The modern AI essays are 800-1,500 words; most human student essays are
shorter. A model trained on that can learn "long = AI". `length_match_rows`
shortens some AI training texts to lengths drawn from the human training
texts, so length stops carrying the label:

- rows chosen: label "ai" and `source` in `sources` (default: the fresh
  essays); each is cropped with probability `prob` (every oversampled copy
  decides on its own, so a text can appear both whole and cropped);
- target length: the word count of a random human train row (from
  `reference_sources` when given, else every human row of the same domain,
  else every human row);
- only shortened, never lengthened; at least `min_words` words are kept;
- cut at the last paragraph or sentence end that fits, so the text still ends
  cleanly; the start of the essay is kept (as a student would hand in a
  shorter version).

Only the texts passed to `fit()` change; evaluation splits are untouched, so
the reports still show how the detector does on whole essays.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

_SENT_END = re.compile(r"[.!?][\"')\]]?(?=\s|$)")


def crop_to_words(text: str, n_words: int) -> str:
    """The longest prefix of `text` with at most `n_words` words that ends at a paragraph or
    sentence end (falls back to a plain word cut when no sentence end fits)."""
    words = list(re.finditer(r"\S+", text))
    if len(words) <= n_words:
        return text
    limit = words[n_words - 1].end()
    head = text[:limit]
    para = head.rfind("\n\n")
    ends = [m.end() for m in _SENT_END.finditer(head)]
    cut = max([e for e in ends] + ([para] if para > 0 else []), default=0)
    if cut < 0.6 * limit:          # nothing sensible: plain word cut
        cut = limit
    return head[:cut].rstrip()


def length_match_rows(df: pd.DataFrame, *, sources: Sequence[str] = ("fresh",),
                      reference_sources: Sequence[str] | None = None, prob: float = 0.5,
                      min_words: int = 150, seed: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (frame with some AI texts cropped, info). See the module docstring."""
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if df.empty or prob == 0:
        return df, {}
    rng = np.random.default_rng(seed)
    label = df["label"].astype(str).str.lower()
    src = df["source"].astype(str) if "source" in df.columns else pd.Series("", index=df.index)
    human = df[label == "human"]
    if reference_sources:
        human = human[src[label == "human"].isin(list(reference_sources))]
    if human.empty:
        raise ValueError("length matching needs human train rows to take lengths from")
    lengths = human["text"].str.split().str.len().to_numpy()
    by_domain: dict[str, np.ndarray] = {}
    if "domain" in df.columns:
        for dom, rows in human.groupby(human["domain"].astype(str)):
            by_domain[dom] = rows["text"].str.split().str.len().to_numpy()

    pick = ((label == "ai") & src.isin(list(sources))).to_numpy()
    chosen = pick & (rng.random(len(df)) < prob)
    texts = df["text"].tolist()
    domains = df["domain"].astype(str).tolist() if "domain" in df.columns else [""] * len(df)
    n_cropped, before, after = 0, [], []
    for i in np.flatnonzero(chosen):
        pool = by_domain.get(domains[i], lengths)
        target = max(int(rng.choice(pool)), min_words)
        n = len(texts[i].split())
        if target >= n:
            continue
        new = crop_to_words(texts[i], target)
        before.append(n)
        after.append(len(new.split()))
        texts[i] = new
        n_cropped += 1
    out = df.copy()
    out["text"] = texts
    info = {"sources": list(sources), "prob": prob, "n_candidates": int(pick.sum()),
            "n_cropped": n_cropped,
            "median_words_before": int(np.median(before)) if before else None,
            "median_words_after": int(np.median(after)) if after else None,
            "reference": list(reference_sources) if reference_sources else "same-domain human"}
    return out, info
