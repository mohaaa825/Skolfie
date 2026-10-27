"""
skolfie/mixed/synth.py
-----------------------
A synthetic Mixed evaluation set: texts spliced from runs of whole sentences
of a human text and an AI text, plus pure-human and pure-AI controls.

`make_mixed_set(corpus, ids, n_mixed, n_pure)` uses only the corpus rows whose
id is in `ids` (e.g. one evaluation split) and returns a DataFrame with
`MIXED_COLUMNS`:

    id           "mixed:<k>" / "pure:<k>"
    text         the (spliced or cut) text
    label3       "human", "ai" or "mixed"
    ai_spans     JSON [[start, end], ...]: character spans of `text` that came
                 from AI text ([] for pure human, [[0, len(text)]] for pure AI)
    ai_fraction  non-space characters inside ai_spans / all non-space characters
    pattern      "h", "a" (pure) or "ha", "ah", "hah", "aha" (mixed, roughly
                 equal shares)
    source, domain   of the parts (all parts share them)
    parts        JSON list of the corpus ids used, one per part, in order

Splicing rules. All parts of a mixed text come from the same (source, domain),
so a style jump is not just a dataset jump; ids of the form
"hc3:<subset>:<line>:human|chatgpt:<i>" are paired by question (a human and an
AI answer to the same question) and only fall back to plain (source, domain)
pairing once every question of the group has been used. A part is a contiguous
run of whole sentences with at least `min_part_words` words (and at least two
sentences when the text has them); the first part starts at the beginning of
its text, later parts anywhere (the second run of the repeated class in
"hah"/"aha" comes after the first run in the same text, or from another text
of the group when that one is too short). Parts are joined with a single
space. Targets for the AI share are drawn from [0.25, 0.75]; a candidate is
kept only if it has at most `max_words` words and its AI fraction (by
characters) is within [0.2, 0.8], and identical candidates are not repeated.

Pure texts prefer rows with at least `2 * min_part_words` words that no mixed
text used, and are cut at a sentence boundary to at most `max_words` words.

Deterministic given `seed`. Texts are sentence-split lazily (only sampled
ones), so a few hundred thousand allowed rows are fine.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from typing import Sequence

import numpy as np
import pandas as pd

from skolfie.mixed.chunking import split_sentences

MIXED_COLUMNS = ["id", "text", "label3", "ai_spans", "ai_fraction", "pattern", "source",
                 "domain", "parts"]
PATTERNS = ("ha", "ah", "hah", "aha")
AI_FRACTION_RANGE = (0.2, 0.8)
_HC3_QUESTION = r"^(hc3:[^:]+:[^:]+):(?:human|chatgpt):"


class _Sentences:
    """Lazy per-row sentence spans and cumulative word counts."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self._cache: dict[int, tuple[list[tuple[int, int]], list[int]]] = {}

    def __call__(self, i: int) -> tuple[list[tuple[int, int]], list[int]]:
        info = self._cache.get(i)
        if info is None:
            text = self.texts[i]
            spans = split_sentences(text)
            cum = [0]
            for s, e in spans:
                cum.append(cum[-1] + len(text[s:e].split()))
            info = (spans, cum)
            self._cache[i] = info
        return info


class _Group:
    """Pair sampler for one (source, domain): question pairs first, then any pair."""

    def __init__(self, key: tuple[str, str], human: np.ndarray, ai: np.ndarray,
                 questions: list[tuple[np.ndarray, np.ndarray]],
                 rng: np.random.Generator) -> None:
        self.key = key
        self.pools = {"h": human, "a": ai}
        self.questions = questions
        self.q_order = rng.permutation(len(questions))
        self.q_pos = 0
        self.order = {c: rng.permutation(len(p)) for c, p in self.pools.items()}
        self.pos = {"h": 0, "a": 0}

    @property
    def weight(self) -> int:
        return min(len(self.pools["h"]), len(self.pools["a"]))

    def _next(self, c: str, rng: np.random.Generator) -> int:
        if self.pos[c] >= len(self.order[c]):
            self.order[c] = rng.permutation(len(self.pools[c]))
            self.pos[c] = 0
        i = self.pools[c][self.order[c][self.pos[c]]]
        self.pos[c] += 1
        return int(i)

    def draw(self, rng: np.random.Generator) -> tuple[dict[str, int], dict[str, np.ndarray]]:
        """One human and one AI row, plus the pools to take a second text from."""
        if self.q_pos < len(self.q_order):
            h, a = self.questions[self.q_order[self.q_pos]]
            self.q_pos += 1
            chosen = {"h": int(h[rng.integers(len(h))]), "a": int(a[rng.integers(len(a))])}
            alts = {"h": h if len(h) > 1 else self.pools["h"],
                    "a": a if len(a) > 1 else self.pools["a"]}
            return chosen, alts
        return {c: self._next(c, rng) for c in ("h", "a")}, self.pools


def _run(cum: list[int], start: int, target: float, min_words: int) -> tuple[int, int] | None:
    """Sentences [start, end) holding >= target words (or up to the text's end)."""
    n = len(cum) - 1
    end = min(bisect_left(cum, cum[start] + target, lo=start + 1), n)
    if end - start < 2 and end < n:
        end += 1  # at least two sentences when the text has them
    if cum[end] - cum[start] < min_words:
        return None
    return start, end


def _random_start(cum: list[int], lo: int, min_words: int,
                  rng: np.random.Generator) -> int | None:
    """A sentence index >= lo from which at least min_words words remain."""
    hi = bisect_right(cum, cum[-1] - min_words) - 1  # last start with enough words left
    hi = min(hi, len(cum) - 2)
    if hi < lo:
        return None
    return int(rng.integers(lo, hi + 1))


def _non_space(s: str) -> int:
    return len("".join(s.split()))


def _build_mixed(pattern: str, group: _Group, sents: _Sentences, rng: np.random.Generator,
                 min_part_words: int, max_words: int) -> dict | None:
    chosen, alts = group.draw(rng)
    n_parts = len(pattern)
    t_hi = 0.9 * max_words
    t_lo = min(t_hi, max(1.25 * n_parts * min_part_words, 0.3 * max_words))
    total = rng.uniform(t_lo, t_hi)
    f_ai = rng.uniform(0.25, 0.75)
    class_words = {"a": f_ai * total, "h": (1 - f_ai) * total}

    parts: list[tuple[str, int, int, int]] = []  # (class, row, first sentence, end sentence)
    first_end = 0
    for p, c in enumerate(pattern):
        target = max(min_part_words, class_words[c] / pattern.count(c))
        row = chosen[c]
        cum = sents(row)[1]
        if len(cum) < 2:
            return None
        if p == 0:
            start = 0
        else:
            lo = first_end if c == pattern[0] else 0
            start = _random_start(cum, lo, min_part_words, rng)
            if start is None and c == pattern[0]:
                # Second run of the repeated class: another text of the same class.
                pool = alts[c][alts[c] != row]
                if len(pool) == 0:
                    return None
                row = int(pool[rng.integers(len(pool))])
                cum = sents(row)[1]
                if len(cum) < 2:
                    return None
                start = _random_start(cum, 0, min_part_words, rng)
            if start is None:
                return None
        run = _run(cum, start, target, min_part_words)
        if run is None:
            return None
        if p == 0:
            first_end = run[1]
        parts.append((c, row, run[0], run[1]))

    if sum(sents(r)[1][e] - sents(r)[1][s] for _, r, s, e in parts) > max_words:
        return None

    pieces, ai_spans, offset, ai_chars = [], [], 0, 0
    for c, row, s, e in parts:
        spans = sents(row)[0]
        piece = sents.texts[row][spans[s][0]:spans[e - 1][1]]
        if pieces:
            offset += 1  # the joining space
        if c == "a":
            if ai_spans and ai_spans[-1][1] == offset:
                ai_spans[-1][1] = offset + len(piece)
            else:
                ai_spans.append([offset, offset + len(piece)])
            ai_chars += _non_space(piece)
        pieces.append(piece)
        offset += len(piece)
    text = " ".join(pieces)
    fraction = ai_chars / max(_non_space(text), 1)
    if not AI_FRACTION_RANGE[0] <= fraction <= AI_FRACTION_RANGE[1]:
        return None
    return {"text": text, "ai_spans": ai_spans, "ai_fraction": fraction, "pattern": pattern,
            "rows": [r for _, r, _, _ in parts],
            "signature": (pattern, tuple(parts))}


def _cut(text: str, spans: list[tuple[int, int]], cum: list[int], max_words: int) -> str:
    """Leading whole sentences with at most max_words words (at least the first one,
    cut at a word boundary if it alone is too long)."""
    if not spans:
        return ""
    k = bisect_right(cum, max_words) - 1
    if k >= 1:
        return text[spans[0][0]:spans[k - 1][1]]
    words = text[spans[0][0]:spans[0][1]].split()[:max_words]
    return " ".join(words)


def make_mixed_set(corpus: pd.DataFrame, ids: Sequence[str], n_mixed: int, n_pure: int,
                   seed: int = 0, min_part_words: int = 40,
                   max_words: int = 600) -> pd.DataFrame:
    """Build `n_mixed` spliced texts plus `n_pure` pure-human and `n_pure` pure-AI texts."""
    rng = np.random.default_rng(seed)
    # object dtype: pandas' Arrow-string isin is ~15x slower on large id lists
    keep = corpus["id"].astype(object).isin(pd.unique(np.asarray(list(ids), dtype=object)))
    allowed = corpus[keep.to_numpy()].reset_index(drop=True)
    texts = allowed["text"].astype(str).tolist()
    row_ids = allowed["id"].astype(str).tolist()
    sources = allowed["source"].astype(str).tolist()
    domains = allowed["domain"].astype(str).tolist()
    labels = allowed["label"].to_numpy()
    if "n_words" in allowed:
        n_words = allowed["n_words"].to_numpy()
    else:
        n_words = np.array([len(t.split()) for t in texts])
    is_h = (labels == "human") & (n_words > 0)
    is_a = (labels == "ai") & (n_words > 0)
    for name, mask in (("human", is_h), ("AI", is_a)):
        have = int(mask.sum())
        if have < n_pure or (n_mixed > 0 and have == 0):
            raise ValueError(f"not enough {name} rows among the {len(allowed)} allowed ids: "
                             f"need {max(n_pure, 1 if n_mixed else 0)}, have {have}")
    sents = _Sentences(texts)
    out: list[dict] = []
    used = np.zeros(len(allowed), dtype=bool)

    if n_mixed > 0:
        frame = pd.DataFrame({"source": sources, "domain": domains,
                              "q": pd.Series(row_ids, dtype=object).str.extract(
                                  _HC3_QUESTION, expand=False)})
        questions: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
        q_rows = frame[frame["q"].notna()]
        by_question = q_rows.groupby(["source", "domain", "q"], sort=True).indices
        for (src, dom, _), pos in by_question.items():
            rows = q_rows.index.to_numpy()[pos]
            h, a = rows[is_h[rows]], rows[is_a[rows]]
            if len(h) and len(a):
                questions.setdefault((src, dom), []).append((h, a))
        groups = []
        for key, rows in frame.groupby(["source", "domain"], sort=True).indices.items():
            g = _Group(key, rows[is_h[rows]], rows[is_a[rows]], questions.get(key, []), rng)
            if g.weight > 0:
                groups.append(g)
        if not groups:
            raise ValueError("no (source, domain) group of the allowed ids has both human "
                             "and AI rows, so no mixed text can be built")
        cdf = np.cumsum([g.weight for g in groups], dtype=float)
        cdf /= cdf[-1]

        patterns = rng.permutation([PATTERNS[i % len(PATTERNS)] for i in range(n_mixed)])
        seen: set = set()
        budget = 50 * n_mixed + 500
        for k, pattern in enumerate(patterns):
            item = None
            while item is None:
                if budget <= 0:
                    raise ValueError(
                        f"could build only {k} of {n_mixed} mixed texts satisfying the "
                        f"constraints (min_part_words={min_part_words}, max_words={max_words}, "
                        f"AI fraction in {AI_FRACTION_RANGE}); lower n_mixed or "
                        "min_part_words, or allow more ids")
                budget -= 1
                group = groups[int(np.searchsorted(cdf, rng.random(), side="right"))]
                item = _build_mixed(str(pattern), group, sents, rng, min_part_words, max_words)
                if item is not None and item["signature"] in seen:
                    item = None
            seen.add(item["signature"])
            used[item["rows"]] = True
            out.append({"id": f"mixed:{k}", "text": item["text"], "label3": "mixed",
                        "ai_spans": json.dumps(item["ai_spans"]),
                        "ai_fraction": float(item["ai_fraction"]), "pattern": item["pattern"],
                        "source": group.key[0], "domain": group.key[1],
                        "parts": json.dumps([row_ids[r] for r in item["rows"]])})

    # Pure controls: prefer long enough rows, then rows unused by the mixed texts.
    long_enough = n_words >= 2 * min_part_words
    k = 0
    for label3, mask in (("human", is_h), ("ai", is_a)):
        cand = rng.permutation(np.flatnonzero(mask))
        tier = (~long_enough[cand]).astype(int) * 2 + used[cand].astype(int)
        for r in cand[np.argsort(tier, kind="stable")[:n_pure]]:
            spans, cum = sents(int(r))
            text = _cut(texts[r], spans, cum, max_words)
            out.append({"id": f"pure:{k}", "text": text, "label3": label3,
                        "ai_spans": json.dumps([[0, len(text)]] if label3 == "ai" else []),
                        "ai_fraction": 1.0 if label3 == "ai" else 0.0,
                        "pattern": label3[0], "source": sources[r], "domain": domains[r],
                        "parts": json.dumps([row_ids[r]])})
            k += 1
    return pd.DataFrame(out, columns=MIXED_COLUMNS)
