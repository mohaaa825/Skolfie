"""Tests for skolfie.mixed.detector (roadmap step 7), with a fake scorer and chunker."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from skolfie.mixed.detector import (
    LABELS3,
    ChunkRule,
    WholeTextRule,
    ai_char_mask,
    fit_rule,
    highlight,
    score_texts,
    span_metrics,
    three_way_metrics,
)

AI_VOCAB = {"delve", "tapestry", "furthermore", "crucial", "leverage", "robust", "seamless",
            "holistic"}


class FakeScorer:
    """P(ai) = share of words from AI_VOCAB; records every call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict_proba(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            words = re.findall(r"[a-z]+", t.lower())
            out.append(sum(w in AI_VOCAB for w in words) / len(words) if words else 0.0)
        return np.array(out)

    @property
    def n_scored(self) -> int:
        return sum(len(c) for c in self.calls)


def sentence_chunker(text: str) -> list[tuple[int, int]]:
    """One span per sentence, split on '. ' (the period stays with its sentence)."""
    spans, start = [], 0
    for m in re.finditer(r"\. ", text):
        spans.append((start, m.start() + 1))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


HUMAN = "we went to the shop and bought some bread"
AI = "delve into the robust tapestry and leverage seamless holistic synergy"


def scored_row(p, chunk_ps, length=10):
    chunks = [(i * length, (i + 1) * length, cp) for i, cp in enumerate(chunk_ps)]
    return {"p": p, "chunks": chunks}


# --------------------------------------------------------------------------- #
# score_texts
# --------------------------------------------------------------------------- #
def test_score_texts_structure_and_spans():
    texts = [f"{HUMAN}. {AI}. {HUMAN}.", f"{AI}."]
    scorer = FakeScorer()
    scored = score_texts(scorer, texts, chunker=sentence_chunker, log=None)
    assert len(scored) == 2
    first = scored[0]
    assert set(first) == {"p", "chunks"}
    assert [(s, e) for s, e, _ in first["chunks"]] == sentence_chunker(texts[0])
    starts = [s for s, _, _ in first["chunks"]]
    assert starts == sorted(starts)
    for (s, e, p) in first["chunks"]:
        assert p == pytest.approx(scorer.predict_proba([texts[0][s:e]])[0])
    ps = [p for _, _, p in first["chunks"]]
    assert ps[0] == 0.0 and ps[1] > 0.5 and ps[2] == 0.0
    assert first["p"] == pytest.approx(scorer.predict_proba([texts[0]])[0])


def test_score_texts_single_chunk_reuses_whole_score():
    texts = [f"{AI}.", f"{HUMAN}.", f"{HUMAN}. {AI}."]
    scorer = FakeScorer()
    scored = score_texts(scorer, texts, chunker=sentence_chunker, log=None)
    # 3 whole texts + 2 chunks of the last one; the single-chunk texts are not rescored.
    assert scorer.n_scored == 5
    assert scored[0]["chunks"] == [(0, len(texts[0]), scored[0]["p"])]
    assert scored[1]["chunks"][0][2] == scored[1]["p"]


def test_score_texts_empty_chunker_output_gives_one_chunk():
    scored = score_texts(FakeScorer(), [""], chunker=lambda t: [], log=None)
    assert scored == [{"p": 0.0, "chunks": [(0, 0, 0.0)]}]


def test_score_texts_batches_across_boundaries():
    texts = [f"{HUMAN}. {AI}. {HUMAN}." for _ in range(3)] + [f"{AI}."] * 2
    scorer = FakeScorer()
    lines: list[str] = []
    scored = score_texts(scorer, texts, chunker=sentence_chunker, batch_size=4, log=lines.append)
    # 5 whole texts -> batches of 4 + 1; 9 chunk texts -> 4 + 4 + 1.
    assert [len(c) for c in scorer.calls] == [4, 1, 4, 4, 1]
    assert len(lines) == 5
    reference = score_texts(FakeScorer(), texts, chunker=sentence_chunker, batch_size=1000,
                            log=None)
    assert scored == reference


def test_score_texts_rejects_bad_batch_size():
    with pytest.raises(ValueError):
        score_texts(FakeScorer(), ["x"], chunker=sentence_chunker, batch_size=0, log=None)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def test_chunk_rule_labels():
    rule = ChunkRule(lo=0.2, hi=0.8)
    scored = [
        scored_row(0.6, [0.1, 0.9]),        # clear disagreement -> mixed
        scored_row(0.9, [0.85, 0.95]),      # all AI
        scored_row(0.1, [0.05, 0.1]),       # all human
        scored_row(0.5, [0.3, 0.7]),        # disagreement not clear enough -> ai (p >= 0.5)
        scored_row(0.9, [0.95]),            # one chunk can't be mixed
    ]
    assert list(rule.predict(scored)) == ["mixed", "ai", "human", "ai", "ai"]


def test_chunk_rule_min_chunks_and_min_share():
    row = scored_row(0.2, [0.1, 0.05, 0.1, 0.1, 0.9])   # AI share 1/5 = 0.2
    assert ChunkRule().predict([row])[0] == "mixed"
    assert ChunkRule(min_chunks=6).predict([row])[0] == "human"
    assert ChunkRule(min_share=0.2).predict([row])[0] == "mixed"
    assert ChunkRule(min_share=0.25).predict([row])[0] == "human"
    # Share is length-weighted: a long AI chunk makes the AI share large.
    weighted = {"p": 0.9, "chunks": [(0, 10, 0.1), (10, 100, 0.9)]}   # AI share 0.9
    assert ChunkRule(min_share=0.1).predict([weighted])[0] == "mixed"
    assert ChunkRule(min_share=0.2).predict([weighted])[0] == "ai"


def test_whole_text_rule():
    scored = [{"p": p, "chunks": []} for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert list(WholeTextRule(0.3, 0.7).predict(scored)) == ["human", "mixed", "mixed",
                                                             "mixed", "ai"]
    assert list(WholeTextRule(0.5, 0.5).predict(scored)) == ["human", "human", "ai", "ai", "ai"]


def _toy_set():
    rows, labels = [], []
    for k in range(6):
        rows.append(scored_row(0.1 + 0.01 * k, [0.05, 0.1, 0.15]))
        labels.append("human")
        rows.append(scored_row(0.9 - 0.01 * k, [0.9, 0.95, 0.85]))
        labels.append("ai")
        rows.append(scored_row(0.6 - 0.05 * k, [0.1 + 0.02 * k, 0.9 - 0.02 * k, 0.5]))
        labels.append("mixed")
    return rows, labels


def test_fit_rule_beats_trivial_and_is_deterministic():
    rows, labels = _toy_set()
    trivial = three_way_metrics(labels, WholeTextRule(0.5, 0.5).predict(rows))["macro_f1"]
    rule, score = fit_rule(ChunkRule, rows, labels)
    assert isinstance(rule, ChunkRule)
    assert score == pytest.approx(1.0)
    assert score > trivial
    assert three_way_metrics(labels, rule.predict(rows))["macro_f1"] == pytest.approx(score)
    again, score2 = fit_rule(ChunkRule, rows, labels)
    assert again == rule and score2 == score

    whole, whole_score = fit_rule(WholeTextRule, rows, labels)
    assert whole.lo <= whole.hi
    assert whole_score >= trivial
    assert fit_rule(WholeTextRule, rows, labels) == (whole, whole_score)


def test_fit_rule_custom_grid_and_tie_break():
    rows, labels = _toy_set()
    # Every lo in the grid separates the toy set perfectly; the first one wins.
    rule, score = fit_rule(ChunkRule, rows, labels, grid={"lo": [0.3, 0.25], "hi": [0.8]})
    assert (rule.lo, rule.hi, rule.min_share) == (0.3, 0.8, 0.0)
    assert score == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# three_way_metrics
# --------------------------------------------------------------------------- #
def test_three_way_metrics_hand_computed():
    y_true = ["human", "human", "human", "ai", "ai", "mixed", "mixed"]
    y_pred = ["human", "ai", "human", "ai", "ai", "ai", "human"]   # "mixed" never predicted
    m = three_way_metrics(y_true, y_pred)
    assert m["n"] == 7
    assert m["accuracy"] == pytest.approx(4 / 7)
    # human: tp 2, pred 3, true 3 -> P 2/3, R 2/3, F1 2/3
    # ai:    tp 2, pred 4, true 2 -> P 1/2, R 1,   F1 2/3
    # mixed: tp 0, pred 0, true 2 -> P 0,   R 0,   F1 0
    assert m["per_class"]["human"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["ai"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["ai"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["ai"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["mixed"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                       "support": 2, "n_pred": 0}
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["confusion"] == {
        "human": {"human": 2, "ai": 1, "mixed": 0},
        "ai": {"human": 0, "ai": 2, "mixed": 0},
        "mixed": {"human": 1, "ai": 1, "mixed": 0},
    }
    assert m["human_to_ai_rate"] == pytest.approx(1 / 3)
    assert m["mixed_recall"] == 0.0


def test_three_way_metrics_absent_class_and_validation():
    m = three_way_metrics(["human", "ai"], ["human", "ai"])
    assert m["macro_f1"] == 1.0          # "mixed" occurs nowhere and is left out
    assert math.isnan(m["mixed_recall"])
    assert set(m["confusion"]) == set(LABELS3)
    with pytest.raises(ValueError):
        three_way_metrics(["human"], ["uncertain"])


# --------------------------------------------------------------------------- #
# Span metrics
# --------------------------------------------------------------------------- #
def test_ai_char_mask():
    mask = ai_char_mask(10, [[2, 4], [8, 20], [-3, 1]])
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, True, False, False, False, False, True, True]
    assert not ai_char_mask(5, []).any()


def _mixed_example():
    h1, h2, a1 = "one two three four", "five six seven eight", "nine ten eleven twelve"
    text = f"{h1}. {h2}. {a1}."
    spans = sentence_chunker(text)          # 3 sentences, 4 words each
    ai_start = spans[2][0]
    return text, spans, [[ai_start, len(text)]]


def test_span_metrics_perfect_prediction():
    text, spans, true = _mixed_example()
    row = {"p": 0.5, "chunks": [(spans[0][0], spans[0][1], 0.1), (spans[1][0], spans[1][1], 0.2),
                                (spans[2][0], spans[2][1], 0.9)]}
    m = span_metrics([text], [true], [row])
    assert m["char_precision"] == m["char_recall"] == m["char_f1"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["n_boundary_texts"] == 1
    assert m["boundary_error_mean"] == 0.0 and m["boundary_error_median"] == 0.0


def test_span_metrics_shifted_prediction():
    text, spans, true = _mixed_example()
    # Prediction flips one sentence (4 words) too early.
    row = {"p": 0.5, "chunks": [(spans[0][0], spans[0][1], 0.1), (spans[1][0], spans[1][1], 0.9),
                                (spans[2][0], spans[2][1], 0.9)]}
    m = span_metrics([text], [true], [row])
    true_chars = sum(not c.isspace() for c in text[spans[2][0]:])
    pred_chars = sum(not c.isspace() for c in text[spans[1][0]:])
    assert m["char_recall"] == 1.0
    assert m["char_precision"] == pytest.approx(true_chars / pred_chars)
    assert m["boundary_error_mean"] == 4.0

    # No flip at all (everything human): distance to the nearer end = the 4 AI words.
    flat = {"p": 0.1, "chunks": [(s, e, 0.1) for s, e in spans]}
    m2 = span_metrics([text], [true], [flat])
    assert m2["boundary_error_mean"] == 4.0
    assert m2["char_recall"] == 0.0 and m2["char_f1"] == 0.0

    both = span_metrics([text, text], [true, true], [row, flat])
    assert both["n_boundary_texts"] == 2 and both["boundary_error_median"] == 4.0


def test_span_metrics_ignore_whitespace_and_ah_pattern():
    text = "alpha beta.   gamma delta."
    true = [[0, 14]]       # AI span includes the run of spaces before the human sentence
    row = {"p": 0.5, "chunks": [(0, 11, 0.9), (14, len(text), 0.1)]}
    m = span_metrics([text], [true], [row])
    assert m["char_f1"] == 1.0
    assert m["n_boundary_texts"] == 1 and m["boundary_error_mean"] == 0.0


def test_span_metrics_skips_boundary_for_hah():
    text, spans, _ = _mixed_example()
    true = [[spans[1][0], spans[1][1]]]     # AI in the middle: two boundaries
    row = {"p": 0.5, "chunks": [(s, e, 0.1) for s, e in spans]}
    m = span_metrics([text], [true], [row])
    assert m["n_boundary_texts"] == 0 and math.isnan(m["boundary_error_mean"])


# --------------------------------------------------------------------------- #
# highlight
# --------------------------------------------------------------------------- #
def test_highlight_merges_adjacent_same_label():
    text = "x" * 40
    chunks = [(0, 10, 0.1), (10, 30, 0.4), (30, 35, 0.9), (35, 40, 0.7)]
    spans = highlight(text, chunks)
    assert [(s["start"], s["end"], s["label"]) for s in spans] == [(0, 30, "human"),
                                                                   (30, 40, "ai")]
    assert spans[0]["p"] == pytest.approx((0.1 * 10 + 0.4 * 20) / 30)
    assert spans[1]["p"] == pytest.approx(0.8)
    assert [s["label"] for s in highlight(text, chunks, threshold=0.05)] == ["ai"]
    assert highlight(text, []) == []
