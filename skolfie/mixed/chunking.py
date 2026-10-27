"""
skolfie/mixed/chunking.py
--------------------------
Sentence spans and sentence-aligned chunks for Mixed (part human, part AI)
detection.

`split_sentences()` returns character spans of sentences. A sentence ends
after a run of `.`, `!` or `?` (plus any closing quotes / brackets) that is
followed by whitespace, and at every newline. Spans never include leading or
trailing whitespace, and together they cover every non-space character of
the text. No abbreviation handling: "Dr. Smith" is two sentences, which is
harmless for chunking.

`chunk_spans()` groups consecutive whole sentences into chunks of about
`target_words` words (sentences are added until the chunk has at least
`target_words`); a last chunk shorter than `min_words` is merged into the
previous one, and a single sentence longer than `2 * target_words` (e.g.
unpunctuated text) is first cut at word boundaries into pieces of about
`target_words`. Chunks are ordered, non-overlapping and stripped, so
`text[start:end]` is the chunk text.

Pure Python + `re` and deterministic: it runs on tens of thousands of texts.
"""

from __future__ import annotations

import re

# End of a sentence: terminal punctuation (+ closing quotes/brackets) before
# whitespace, or a newline (the newline itself is the cut point).
_BOUNDARY = re.compile(r"[.!?]+[\"'”’»)\]]*(?=\s)|\n")
_WORD = re.compile(r"\S+")


def _add_stripped(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    seg = text[start:end]
    stripped = seg.strip()
    if stripped:
        s = start + (len(seg) - len(seg.lstrip()))
        spans.append((s, s + len(stripped)))


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of the sentences of `text`, in order."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if m.group() == "\n":
            _add_stripped(spans, text, start, m.start())
        else:
            _add_stripped(spans, text, start, m.end())
        start = m.end()
    _add_stripped(spans, text, start, len(text))
    return spans


def _split_long(text: str, start: int, end: int, n_words: int,
                target_words: int) -> list[tuple[int, int, int]]:
    """Cut one over-long sentence at word boundaries into ~target_words pieces."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text, start, end)]
    n_pieces = max(2, round(n_words / target_words))
    cuts = [round(i * n_words / n_pieces) for i in range(n_pieces + 1)]
    return [(words[a][0], words[b - 1][1], b - a) for a, b in zip(cuts, cuts[1:]) if b > a]


def chunk_spans(text: str, target_words: int = 60,
                min_words: int = 25) -> list[tuple[int, int]]:
    """Character spans of consecutive sentence groups of about `target_words` words."""
    sents = split_sentences(text)
    if not sents:
        return []
    counts = [len(text[s:e].split()) for s, e in sents]
    total = sum(counts)
    if total < min_words or (len(sents) == 1 and counts[0] <= 2 * target_words):
        return [(sents[0][0], sents[-1][1])]

    # Units: sentences, with over-long ones cut into ~target_words pieces.
    units: list[tuple[int, int, int]] = []
    for (s, e), n in zip(sents, counts):
        if n > 2 * target_words:
            units.extend(_split_long(text, s, e, n, target_words))
        else:
            units.append((s, e, n))

    chunks: list[list[int]] = []  # [start, end, n_words]
    cur: list[int] | None = None
    for s, e, n in units:
        if cur is None:
            cur = [s, e, n]
        else:
            cur[1] = e
            cur[2] += n
        if cur[2] >= target_words:
            chunks.append(cur)
            cur = None
    if cur is not None:
        if chunks and cur[2] < min_words:
            chunks[-1][1] = cur[1]
            chunks[-1][2] += cur[2]
        else:
            chunks.append(cur)
    return [(c[0], c[1]) for c in chunks]
