"""
tests/test_mixed_data.py
-------------------------
Data side of Mixed detection: sentence spans and chunks
(`skolfie/mixed/chunking.py`) and the synthetic Mixed evaluation set
(`skolfie/mixed/synth.py`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from skolfie.mixed.chunking import chunk_spans, split_sentences
from skolfie.mixed.synth import MIXED_COLUMNS, make_mixed_set
from tests.test_evaluate import make_corpus


def _sents(text: str) -> list[str]:
    return [text[s:e] for s, e in split_sentences(text)]


# ----------------------------------------------------------------- sentences

def test_split_sentences_simple_cases():
    assert _sents("One two. Three four! Five six?") == ["One two.", "Three four!", "Five six?"]
    assert _sents("What?! Really... Yes") == ["What?!", "Really...", "Yes"]
    assert _sents("No final punctuation") == ["No final punctuation"]
    assert _sents("Pi is 3.14 today. OK.") == ["Pi is 3.14 today.", "OK."]


def test_split_sentences_newlines_and_quotes():
    text = 'He said "stop." Then left.\nA heading\n\n  (An aside.) End'
    assert _sents(text) == ['He said "stop."', "Then left.", "A heading", "(An aside.)", "End"]
    # closing curly quotes stay with their sentence; no lowercase lookahead
    assert _sents("She asked “why?” and left.") == ["She asked “why?”", "and left."]


def test_split_sentences_spans_are_stripped_and_cover_text():
    text = "  First one.   Second one!  \n  third  "
    spans = split_sentences(text)
    for s, e in spans:
        assert text[s:e] == text[s:e].strip() and text[s:e]
    assert [s for s, _ in spans] == sorted(s for s, _ in spans)
    covered = "".join(text[s:e] for s, e in spans)
    assert "".join(covered.split()) == "".join(text.split())
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []


# -------------------------------------------------------------------- chunks

def _check_chunks(text: str, spans: list[tuple[int, int]]) -> None:
    prev_end = 0
    for s, e in spans:
        assert prev_end <= s < e
        assert text[s:e] == text[s:e].strip()
        prev_end = e
    joined = "".join(text[s:e] for s, e in spans)
    assert "".join(joined.split()) == "".join(text.split())


def _sentence_text(n_sentences: int, words_per_sentence: int = 10) -> str:
    return " ".join(" ".join(f"w{i}x{j}" for j in range(words_per_sentence)) + "."
                    for i in range(n_sentences))


def test_chunks_cover_text_in_order_near_target():
    text = _sentence_text(30)  # 300 words in 10-word sentences
    spans = chunk_spans(text, target_words=60, min_words=25)
    _check_chunks(text, spans)
    sizes = [len(text[s:e].split()) for s, e in spans]
    assert sizes == [60] * 5
    # chunk boundaries fall on sentence boundaries
    sentence_ends = {e for _, e in split_sentences(text)}
    assert all(e in sentence_ends for _, e in spans)


def test_chunks_merge_short_tail():
    text = _sentence_text(14)  # 140 words: 60 + 60 + 20 -> the 20-word tail is merged
    spans = chunk_spans(text, target_words=60, min_words=25)
    _check_chunks(text, spans)
    assert [len(text[s:e].split()) for s, e in spans] == [60, 80]
    text = _sentence_text(15)  # 60 + 60 + 30: tail long enough to stay
    assert [len(text[s:e].split()) for s, e in chunk_spans(text, 60, 25)] == [60, 60, 30]


def test_chunks_split_long_single_sentence():
    text = " ".join(f"word{i}" for i in range(300))  # one unpunctuated sentence
    spans = chunk_spans(text, target_words=60, min_words=25)
    _check_chunks(text, spans)
    sizes = [len(text[s:e].split()) for s, e in spans]
    assert sum(sizes) == 300 and len(spans) == 5
    assert all(50 <= n <= 70 for n in sizes)
    # a long sentence in the middle of normal ones
    text = _sentence_text(3) + " " + text + ". " + _sentence_text(3)
    spans = chunk_spans(text, 60, 25)
    _check_chunks(text, spans)
    assert max(len(text[s:e].split()) for s, e in spans) <= 2 * 60


def test_chunks_short_text_is_one_chunk():
    text = "  A short text. Only a few words here!  "
    assert chunk_spans(text) == [(2, len(text.rstrip()))]
    one = " ".join(f"w{i}" for i in range(100)) + "."  # one sentence, <= 2 * target
    assert chunk_spans(one) == [(0, len(one))]
    assert chunk_spans("") == []
    assert chunk_spans(" \n ") == []


# ----------------------------------------------------------------- mixed set

@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return make_corpus(n_per_cell=30)


@pytest.fixture(scope="module")
def mixed(corpus) -> pd.DataFrame:
    ids = corpus.loc[corpus["domain"] != "academic", "id"].tolist()
    return make_mixed_set(corpus, ids, n_mixed=40, n_pure=10, seed=3, min_part_words=15,
                          max_words=200)


def test_mixed_set_counts_and_patterns(corpus, mixed):
    assert list(mixed.columns) == MIXED_COLUMNS
    assert mixed["label3"].value_counts().to_dict() == {"mixed": 40, "human": 10, "ai": 10}
    assert mixed["id"].is_unique
    mix = mixed[mixed["label3"] == "mixed"]
    assert mix["pattern"].value_counts().to_dict() == {p: 10 for p in ("ha", "ah", "hah", "aha")}
    assert set(mixed.loc[mixed["label3"] == "human", "pattern"]) == {"h"}
    assert set(mixed.loc[mixed["label3"] == "ai", "pattern"]) == {"a"}
    assert "academic" not in set(mixed["domain"])  # only the allowed ids are used
    used = {i for p in mixed["parts"] for i in json.loads(p)}
    assert used <= set(corpus.loc[corpus["domain"] != "academic", "id"])


def _ai_mask(text: str, spans: list[list[int]]) -> np.ndarray:
    mask = np.zeros(len(text), dtype=bool)
    for s, e in spans:
        mask[s:e] = True
    return mask


def test_mixed_spans_come_from_the_right_sources(corpus, mixed):
    by_id = corpus.set_index("id")
    for row in mixed[mixed["label3"] == "mixed"].itertuples():
        parts = json.loads(row.parts)
        spans = json.loads(row.ai_spans)
        labels = [by_id.at[i, "label"] for i in parts]
        assert "".join("a" if lab == "ai" else "h" for lab in labels) == row.pattern
        # all parts share source and domain, which are reported
        assert {by_id.at[i, "source"] for i in parts} == {row.source}
        assert {by_id.at[i, "domain"] for i in parts} == {row.domain}
        # each AI span is a substring of an AI part text; the rest of the text, split at
        # the AI spans, consists of substrings of the human part texts
        ai_texts = [by_id.at[i, "text"] for i in parts if by_id.at[i, "label"] == "ai"]
        human_texts = [by_id.at[i, "text"] for i in parts if by_id.at[i, "label"] == "human"]
        assert len(spans) == row.pattern.count("a")
        for s, e in spans:
            assert any(row.text[s:e] in t for t in ai_texts)
        mask = _ai_mask(row.text, spans)
        human_runs = "".join(ch if not m else "\0" for ch, m in zip(row.text, mask))
        for piece in human_runs.split("\0"):
            if piece.strip():
                assert any(piece.strip() in t for t in human_texts)
        assert 0.2 <= row.ai_fraction <= 0.8
        n_ai = sum(len("".join(row.text[s:e].split())) for s, e in spans)
        assert row.ai_fraction == pytest.approx(n_ai / len("".join(row.text.split())))
        assert len(row.text.split()) <= 200


def test_pure_rows(corpus, mixed):
    by_id = corpus.set_index("id")
    for row in mixed[mixed["label3"] != "mixed"].itertuples():
        (src_id,) = json.loads(row.parts)
        assert by_id.at[src_id, "label"] == row.label3
        assert row.text and by_id.at[src_id, "text"].startswith(row.text)
        assert len(row.text.split()) <= 200
        if row.label3 == "ai":
            assert json.loads(row.ai_spans) == [[0, len(row.text)]] and row.ai_fraction == 1.0
        else:
            assert json.loads(row.ai_spans) == [] and row.ai_fraction == 0.0


def test_pure_texts_cut_at_sentence_boundary():
    text = _sentence_text(20)  # 200 words
    corpus = pd.DataFrame([
        {"id": "x:h", "text": text, "label": "human", "source": "x", "domain": "news"},
        {"id": "x:a", "text": text.upper(), "label": "ai", "source": "x", "domain": "news"},
    ])
    out = make_mixed_set(corpus, ["x:h", "x:a"], n_mixed=0, n_pure=1, max_words=55)
    for t in out["text"]:
        assert len(t.split()) == 50 and t.endswith(".")


def test_mixed_set_is_deterministic(corpus):
    ids = corpus["id"].tolist()
    a = make_mixed_set(corpus, ids, 20, 5, seed=7, min_part_words=15, max_words=200)
    b = make_mixed_set(corpus, ids, 20, 5, seed=7, min_part_words=15, max_words=200)
    c = make_mixed_set(corpus, ids, 20, 5, seed=8, min_part_words=15, max_words=200)
    pd.testing.assert_frame_equal(a, b)
    assert not a["text"].equals(c["text"])


def test_mixed_set_needs_both_classes(corpus):
    humans = corpus.loc[corpus["label"] == "human", "id"].tolist()
    with pytest.raises(ValueError, match="AI rows"):
        make_mixed_set(corpus, humans, n_mixed=5, n_pure=0, min_part_words=15)
    with pytest.raises(ValueError, match="AI rows"):
        make_mixed_set(corpus, humans, n_mixed=0, n_pure=3)
    few = humans + corpus.loc[corpus["label"] == "ai", "id"].tolist()[:2]
    with pytest.raises(ValueError, match="need 3, have 2"):
        make_mixed_set(corpus, few, n_mixed=0, n_pure=3)


_TOPICS = ["apples", "rivers", "engines", "stars", "bread", "trains", "clouds", "songs",
           "bikes", "lakes", "coins", "maps"]


def _hc3_corpus(paired: bool = True) -> pd.DataFrame:
    """HC3-style ids: two human answers and one ChatGPT answer per question line
    (with `paired=False` the ChatGPT answers sit on other lines than the human ones)."""
    rows = []
    for line, topic in enumerate(_TOPICS):
        for kind, n_answers in (("human", 2), ("chatgpt", 1)):
            for i in range(n_answers):
                style = "honestly i reckon" if kind == "human" else "it is important to note"
                text = " ".join(f"{style} that {topic} matter a lot in case {k} for sure."
                                for k in range(8))
                q = line if paired or kind == "human" else line + 100
                rows.append({"id": f"hc3:finance:{q}:{kind}:{i}", "text": text,
                             "label": "human" if kind == "human" else "ai",
                             "source": "hc3", "domain": "qa"})
    return pd.DataFrame(rows)


def _question(corpus_id: str) -> str:
    return ":".join(corpus_id.split(":")[:3])


def test_hc3_pairs_answers_to_the_same_question():
    corpus = _hc3_corpus()
    out = make_mixed_set(corpus, corpus["id"], n_mixed=6, n_pure=0, seed=1, min_part_words=20,
                         max_words=200)
    assert len(out) == 6
    for parts in out["parts"]:
        assert len({_question(i) for i in json.loads(parts)}) == 1


def test_hc3_falls_back_to_source_and_domain():
    corpus = _hc3_corpus(paired=False)
    out = make_mixed_set(corpus, corpus["id"], n_mixed=6, n_pure=0, seed=1, min_part_words=20,
                         max_words=200)
    assert len(out) == 6
    for parts in out["parts"]:
        assert len({_question(i) for i in json.loads(parts)}) > 1


def test_mixed_set_raises_when_constraints_cannot_be_met():
    corpus = _hc3_corpus()
    with pytest.raises(ValueError, match="could build only"):
        make_mixed_set(corpus, corpus["id"], n_mixed=3, n_pure=0, min_part_words=500)
