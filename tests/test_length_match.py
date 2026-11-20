"""Tests for skolfie.eval.length_match (cropping AI train texts to human lengths)."""

from __future__ import annotations

import pandas as pd
import pytest

from skolfie.eval.length_match import crop_to_words, length_match_rows


def _essay(n_sent: int) -> str:
    paras = []
    for p in range(0, n_sent, 5):
        paras.append(" ".join(f"Sentence {i} has exactly six words."
                              for i in range(p, min(p + 5, n_sent))))
    return "\n\n".join(paras)


def test_crop_ends_at_a_sentence_and_never_grows() -> None:
    text = _essay(200)                        # 1,200 words
    out = crop_to_words(text, 400)
    assert len(out.split()) <= 400 and len(out.split()) >= 390
    assert out.endswith(".") and text.startswith(out)
    assert crop_to_words(text, 5000) == text
    assert crop_to_words("one two three four five six", 3) == "one two three"


def _frame() -> pd.DataFrame:
    rows = [{"text": _essay(200), "label": "ai", "source": "fresh", "domain": "academic"}
            for _ in range(200)]
    rows += [{"text": _essay(n), "label": "human", "source": "persuade", "domain": "academic"}
             for n in (50, 60, 70, 80)]
    rows += [{"text": _essay(150), "label": "ai", "source": "m4gt", "domain": "academic"}]
    return pd.DataFrame(rows)


def test_length_match_crops_only_chosen_ai_rows_to_human_lengths() -> None:
    df = _frame()
    out, info = length_match_rows(df, sources=["fresh"], reference_sources=["persuade"],
                                  prob=0.5, seed=0)
    words = out["text"].str.split().str.len()
    fresh = words[df["source"] == "fresh"]
    assert info["n_candidates"] == 200 and 70 < info["n_cropped"] < 130
    assert set(fresh[fresh < 1200]) <= set(range(290, 481))   # 300..480-word human lengths
    assert (words[df["label"] == "human"] == df[df["label"] == "human"]["text"]
            .str.split().str.len()).all()
    assert out.loc[df["source"] == "m4gt", "text"].iloc[0] == df.loc[df["source"] == "m4gt",
                                                                     "text"].iloc[0]
    again, _ = length_match_rows(df, sources=["fresh"], reference_sources=["persuade"], seed=0)
    assert again["text"].tolist() == out["text"].tolist()
    same, info0 = length_match_rows(df, prob=0.0)
    assert same is df and info0 == {}
    with pytest.raises(ValueError):
        length_match_rows(df, prob=2)
    with pytest.raises(ValueError):
        length_match_rows(df, reference_sources=["nope"])


def test_run_evaluation_passes_length_match() -> None:
    from skolfie.eval import report

    df = _frame()
    df.index = [f"r{i:03d}" for i in range(len(df))]
    df["id"] = df.index
    df["generator"] = df["label"].map({"ai": "gemini", "human": "human"})
    splits = pd.DataFrame({"id": list(df.index) + list(df.index[-6:]),
                           "split": ["train"] * len(df) + ["test"] * 6})
    seen: dict = {}

    class Scorer:
        name = "fake"

        def fit(self, texts, labels):
            seen["words"] = [len(t.split()) for t in texts]

        def predict_proba(self, texts):
            return [0.5] * len(texts)

    rep, _ = report.run_evaluation(Scorer(), df, splits, eval_splits=("test",),
                                   length_match={"sources": ["fresh"], "prob": 1.0,
                                                 "reference_sources": ["persuade"]},
                                   log=lambda m: None)
    assert 1200 not in seen["words"] and 900 in seen["words"]   # fresh cropped, m4gt kept
    assert rep["train"]["length_match"]["n_cropped"] == 200
