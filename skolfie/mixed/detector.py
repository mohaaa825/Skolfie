"""
skolfie/mixed/detector.py
--------------------------
Roadmap step 7: call a text Mixed (part human, part AI) when the scores of its
chunks clearly disagree, and highlight the AI spans.

Scoring (`score_texts`)
    Each text is cut into chunks (by default `skolfie.mixed.chunking.chunk_spans`,
    groups of sentences of about 60 words) and every chunk is scored with the
    same binary scorer as the whole text (`scorer.predict_proba(texts) -> P(ai)`).
    All chunk texts go into one flat list that is scored in batches; a text
    with a single chunk reuses its whole-text score for that chunk. The result
    per text is {"p": P(ai) of the whole text, "chunks": [(start, end, p), ...]}
    with character offsets into the text.

Rules (3-way labels, `LABELS3`)
    ChunkRule      Mixed if the text has >= min_chunks chunks, one chunk scores
                   >= hi, another <= lo, and the length-weighted share of chunks
                   with p >= 0.5 lies in [min_share, 1 - min_share]; otherwise AI
                   or Human from the whole-text score at 0.5.
    WholeTextRule  baseline without chunks: Human below lo, AI above hi, Mixed
                   in between (lo == hi never says Mixed).
    `fit_rule()` grid-searches either rule's parameters for 3-way macro-F1,
    optionally only among settings that call at most `max_false_mixed` of the
    pure human texts, and of the pure AI texts, Mixed.

Metrics
    three_way_metrics  macro-F1, accuracy, per-class precision / recall / F1,
                       the confusion matrix, the share of human texts called AI,
                       the shares of pure human / AI texts called Mixed and the
                       recall of Mixed.
    span_metrics       on Mixed texts: character-level precision / recall / F1 of
                       the predicted AI chunks (whitespace ignored) and, for
                       texts with one human/AI boundary, the boundary error in
                       words.

`highlight()` merges neighbouring chunks with the same label into spans for
the UI.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

LABELS3 = ("human", "ai", "mixed")
AI_THRESHOLD = 0.5

Span = tuple[int, int]
Chunker = Callable[[str], Sequence[Span]]

_WORD = re.compile(r"\S+")


def _default_chunker() -> Chunker:
    # Imported lazily so this module works (and is testable) without the chunker module.
    from skolfie.mixed.chunking import chunk_spans

    return chunk_spans


# --------------------------------------------------------------------------- #
# Chunk scoring
# --------------------------------------------------------------------------- #
def _predict_batched(scorer: Any, texts: list[str], batch_size: int, what: str,
                     log: Callable[[str], Any] | None) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.float64)
    n_batches = math.ceil(len(texts) / batch_size)
    for b in range(n_batches):
        lo, hi = b * batch_size, min(len(texts), (b + 1) * batch_size)
        p = np.asarray(scorer.predict_proba(texts[lo:hi]), dtype=np.float64).reshape(-1)
        if p.size != hi - lo:
            raise ValueError(f"scorer returned {p.size} scores for {hi - lo} texts")
        out[lo:hi] = p
        if log is not None:
            log(f"[mixed] scored {what}: batch {b + 1}/{n_batches} ({hi:,}/{len(texts):,})")
    return out


def score_texts(scorer: Any, texts: Sequence[str], chunker: Chunker | None = None,
                batch_size: int = 4096, log: Callable[[str], Any] | None = print) -> list[dict]:
    """
    Score every text as a whole and chunk by chunk.

    Returns one dict per text: {"p": float, "chunks": [(start, end, p), ...]}.
    Chunks come from `chunker(text)` (default `chunk_spans`), in text order. A
    text for which the chunker returns nothing gets one chunk (0, len(text)).
    When a text has exactly one chunk, that chunk gets the whole-text score
    instead of being scored a second time. `log=None` silences progress lines.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chunker = chunker or _default_chunker()
    texts = [str(t) for t in texts]

    all_spans: list[list[Span]] = []
    flat_texts: list[str] = []
    flat_owner: list[int] = []
    for i, text in enumerate(texts):
        spans = [(int(s), int(e)) for s, e in chunker(text)] or [(0, len(text))]
        all_spans.append(spans)
        if len(spans) > 1:
            for s, e in spans:
                flat_texts.append(text[s:e])
                flat_owner.append(i)

    whole = _predict_batched(scorer, texts, batch_size, "whole texts", log) if texts else (
        np.empty(0))
    chunk_p = _predict_batched(scorer, flat_texts, batch_size, "chunks", log) if flat_texts else (
        np.empty(0))

    scored: list[dict] = []
    k = 0
    for i, spans in enumerate(all_spans):
        if len(spans) == 1:
            chunks = [(spans[0][0], spans[0][1], float(whole[i]))]
        else:
            chunks = [(s, e, float(chunk_p[k + j])) for j, (s, e) in enumerate(spans)]
            k += len(spans)
        scored.append({"p": float(whole[i]), "chunks": chunks})
    return scored


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _chunk_stats(scored: Sequence[dict]) -> dict[str, np.ndarray]:
    """Per-text summary used by ChunkRule: whole p, n chunks, max / min chunk p, AI share."""
    n = len(scored)
    stats = {"p": np.empty(n), "n_chunks": np.zeros(n, dtype=np.int64),
             "max_p": np.full(n, np.nan), "min_p": np.full(n, np.nan),
             "ai_share": np.full(n, np.nan)}
    for i, row in enumerate(scored):
        stats["p"][i] = float(row["p"])
        chunks = row.get("chunks") or []
        stats["n_chunks"][i] = len(chunks)
        if not chunks:
            continue
        ps = np.array([c[2] for c in chunks], dtype=np.float64)
        lengths = np.array([max(0, c[1] - c[0]) for c in chunks], dtype=np.float64)
        if lengths.sum() <= 0:
            lengths = np.ones_like(ps)
        stats["max_p"][i] = ps.max()
        stats["min_p"][i] = ps.min()
        stats["ai_share"][i] = float(lengths[ps >= AI_THRESHOLD].sum() / lengths.sum())
    return stats


def _binary_labels(p: np.ndarray) -> np.ndarray:
    return np.where(p >= AI_THRESHOLD, "ai", "human").astype(object)


@dataclass
class ChunkRule:
    """Mixed when the chunk scores clearly disagree (see the module docstring)."""

    lo: float = 0.2
    hi: float = 0.8
    min_chunks: int = 2
    min_share: float = 0.0

    def _predict_stats(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        out = _binary_labels(stats["p"])
        with np.errstate(invalid="ignore"):
            share = stats["ai_share"]
            mixed = ((stats["n_chunks"] >= self.min_chunks)
                     & (stats["max_p"] >= self.hi) & (stats["min_p"] <= self.lo)
                     & (share >= self.min_share - 1e-12)
                     & (share <= 1.0 - self.min_share + 1e-12))
        out[mixed] = "mixed"
        return out

    def predict(self, scored: Sequence[dict]) -> np.ndarray:
        return self._predict_stats(_chunk_stats(scored))


@dataclass
class WholeTextRule:
    """Baseline: Human if p < lo, AI if p > hi, Mixed in between. lo == hi: never Mixed."""

    lo: float = 0.3
    hi: float = 0.7

    def _predict_p(self, p: np.ndarray) -> np.ndarray:
        if self.lo > self.hi:
            raise ValueError(f"lo ({self.lo}) must be <= hi ({self.hi})")
        out = np.full(p.shape, "mixed", dtype=object)
        out[p < self.lo] = "human"
        out[p > self.hi] = "ai"
        if self.lo == self.hi:
            out[p == self.hi] = "ai"  # the one remaining point; p >= threshold counts as AI
        return out

    def predict(self, scored: Sequence[dict]) -> np.ndarray:
        return self._predict_p(np.array([float(r["p"]) for r in scored], dtype=np.float64))


def _steps(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n + 1)]


def default_grid(rule_cls: type) -> dict[str, list]:
    """The parameter grid `fit_rule` searches when none is given."""
    if rule_cls is ChunkRule:
        return {"lo": [0.01, 0.02] + _steps(0.05, 0.4, 0.05),
                "hi": _steps(0.6, 0.95, 0.05) + [0.98, 0.99],
                "min_share": [0.0, 0.1, 0.2, 0.3], "min_chunks": [2, 3]}
    if rule_cls is WholeTextRule:
        vals = _steps(0.05, 0.95, 0.05)
        return {"lo": vals, "hi": vals}
    raise ValueError(f"no default grid for {rule_cls!r}")


def _false_mixed_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The larger of the shares of pure human and of pure AI texts called Mixed."""
    rates = [float(np.mean(y_pred[y_true == c] == "mixed"))
             for c in ("human", "ai") if np.any(y_true == c)]
    return max(rates) if rates else 0.0


def fit_rule(rule_cls: type, scored: Sequence[dict], labels3: Sequence[str],
             grid: dict[str, Sequence] | None = None,
             max_false_mixed: float | None = None,
             log: Callable[[str], Any] | None = print) -> tuple[Any, float]:
    """
    Exhaustive grid search for the rule parameters with the best 3-way macro-F1.

    `grid` maps parameter names to candidate values (default `default_grid`);
    parameters not in the grid keep the class defaults. Combinations are tried
    in a fixed order (parameter names sorted, values in the given order) and
    the first one reaching the best score wins, so the result is deterministic.
    For WholeTextRule, combinations with lo > hi are skipped.

    `max_false_mixed` (a share in [0, 1]) keeps only combinations that call
    at most that share of the pure human texts, and of the pure AI texts,
    Mixed. When none does, the combination with the lowest such share wins
    (best macro-F1 among ties) and a note is logged. Returns (fitted rule, its
    macro-F1).
    """
    if max_false_mixed is not None and not 0 <= max_false_mixed <= 1:
        raise ValueError("max_false_mixed must be between 0 and 1")
    y_true = _check_labels(labels3)
    if len(y_true) != len(scored):
        raise ValueError(f"{len(scored)} scored texts but {len(y_true)} labels")
    grid = dict(grid) if grid is not None else default_grid(rule_cls)
    names = sorted(grid)

    if rule_cls is ChunkRule:
        stats = _chunk_stats(scored)

        def predict(rule: Any) -> np.ndarray:
            return rule._predict_stats(stats)
    elif rule_cls is WholeTextRule:
        p = np.array([float(r["p"]) for r in scored], dtype=np.float64)

        def predict(rule: Any) -> np.ndarray:
            return rule._predict_p(p)
    else:
        def predict(rule: Any) -> np.ndarray:
            return rule.predict(scored)

    base = rule_cls()
    cap = math.inf if max_false_mixed is None else max_false_mixed + 1e-12
    # Ranked by (meets the cap, lowest false-Mixed share if not, macro-F1).
    best_rule, best_key, best_score = None, None, -math.inf
    for values in itertools.product(*(list(grid[k]) for k in names)):
        rule = replace(base, **dict(zip(names, values)))
        if rule_cls is WholeTextRule and rule.lo > rule.hi:
            continue
        y_pred = predict(rule)
        score = _macro_f1(y_true, y_pred)
        false_mixed = _false_mixed_rate(y_true, y_pred) if max_false_mixed is not None else 0.0
        ok = false_mixed <= cap
        key = (ok, 0.0 if ok else -round(false_mixed, 12), score)
        if best_key is None or key[:2] > best_key[:2] or (
                key[:2] == best_key[:2] and score > best_key[2] + 1e-12):
            best_rule, best_key, best_score = rule, key, score
    if best_rule is None:
        raise ValueError("the grid has no valid parameter combination")
    if not best_key[0] and log is not None:
        log(f"[mixed] no {rule_cls.__name__} setting keeps false Mixed at or below "
            f"{max_false_mixed:.1%}; using the lowest ({-best_key[1]:.1%})")
    return best_rule, float(best_score)


# --------------------------------------------------------------------------- #
# 3-way metrics
# --------------------------------------------------------------------------- #
def _check_labels(labels: Sequence[Any]) -> np.ndarray:
    out = np.array([str(lab).strip().lower() for lab in labels], dtype=object)
    bad = sorted(set(out) - set(LABELS3))
    if bad:
        raise ValueError(f"unknown labels {bad}; expected one of {LABELS3}")
    return out


def _per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    per: dict[str, dict[str, float]] = {}
    for c in LABELS3:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        n_pred = int(np.sum(y_pred == c))
        n_true = int(np.sum(y_true == c))
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2 * tp / (n_pred + n_true) if (n_pred + n_true) else 0.0
        per[c] = {"precision": precision, "recall": recall, "f1": f1, "support": n_true,
                  "n_pred": n_pred}
    return per


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
              per: dict[str, dict[str, float]] | None = None) -> float:
    per = per or _per_class(y_true, y_pred)
    present = [c for c in LABELS3 if per[c]["support"] or per[c]["n_pred"]]
    return float(np.mean([per[c]["f1"] for c in present])) if present else math.nan


def three_way_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """
    Metrics for the Human / AI / Mixed labels.

    Returns n, accuracy, macro_f1, per_class {label: {precision, recall, f1,
    support, n_pred}}, confusion {true: {pred: count}} over all of `LABELS3`,
    human_to_ai_rate (share of true-human texts called AI), human_to_mixed_rate
    and ai_to_mixed_rate (shares of pure texts called Mixed) and mixed_recall.
    A class never predicted has precision 0; a class absent from y_true has
    recall 0. macro_f1 averages F1 over the classes that occur in y_true or
    y_pred (a class in neither is left out). Rates are NaN without texts of
    that class.
    """
    yt, yp = _check_labels(y_true), _check_labels(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred differ in length: {yt.size} vs {yp.size}")
    per = _per_class(yt, yp)
    confusion = {t: {p: int(np.sum((yt == t) & (yp == p))) for p in LABELS3} for t in LABELS3}
    n_human = per["human"]["support"]
    n_ai = per["ai"]["support"]
    return {
        "n": int(yt.size),
        "accuracy": float(np.mean(yt == yp)) if yt.size else math.nan,
        "macro_f1": _macro_f1(yt, yp, per),
        "per_class": per,
        "confusion": confusion,
        "human_to_ai_rate": confusion["human"]["ai"] / n_human if n_human else math.nan,
        "human_to_mixed_rate": confusion["human"]["mixed"] / n_human if n_human else math.nan,
        "ai_to_mixed_rate": confusion["ai"]["mixed"] / n_ai if n_ai else math.nan,
        "mixed_recall": per["mixed"]["recall"] if per["mixed"]["support"] else math.nan,
    }


# --------------------------------------------------------------------------- #
# Span-level metrics
# --------------------------------------------------------------------------- #
def ai_char_mask(n_chars: int, spans: Sequence[Sequence[int]]) -> np.ndarray:
    """Boolean mask of length n_chars, True inside any [start, end) span (clipped)."""
    mask = np.zeros(int(n_chars), dtype=bool)
    for span in spans:
        s, e = max(0, int(span[0])), min(int(n_chars), int(span[1]))
        if e > s:
            mask[s:e] = True
    return mask


def _nonspace_mask(text: str) -> np.ndarray:
    return np.fromiter((not ch.isspace() for ch in text), dtype=bool, count=len(text))


def _words_between(word_starts: np.ndarray, a: int, b: int) -> int:
    """Number of words that start in [min(a, b), max(a, b))."""
    lo, hi = min(a, b), max(a, b)
    return int(np.searchsorted(word_starts, hi, "left") - np.searchsorted(word_starts, lo, "left"))


def _single_boundary(true_mask: np.ndarray, nonspace: np.ndarray) -> int | None:
    """Char offset of the boundary for an "ah"/"ha" text, else None."""
    idx = np.flatnonzero(nonspace)
    if idx.size == 0:
        return None
    labels = true_mask[idx]
    flips = np.flatnonzero(labels[1:] != labels[:-1])
    if flips.size != 1:
        return None
    return int(idx[flips[0] + 1])  # first non-space char after the flip


def span_metrics(texts: Sequence[str], true_ai_spans: Sequence[Sequence[Sequence[int]]],
                 scored: Sequence[dict], threshold: float = AI_THRESHOLD) -> dict[str, Any]:
    """
    How well the AI chunks match the true AI spans, on Mixed texts.

    Predicted AI characters are those inside chunks with p >= threshold; only
    non-whitespace characters count. Returns:

        n_texts                          texts evaluated
        char_precision, char_recall,     micro (all characters pooled)
        char_f1
        macro_f1                         mean per-text character F1 (a text with
                                         no true and no predicted AI chars counts 1)
        n_boundary_texts                 texts with exactly one human/AI boundary
                                         (patterns "ha" / "ah")
        boundary_error_mean, _median     |true - predicted boundary| in words

    The true boundary is the first non-space character after the single
    human/AI switch. The predicted boundary is the start of the chunk at which
    the predicted label flips, taking the flip nearest to the true boundary;
    with no flip it is the nearer text end (0 or len(text)). The error counts
    the words that start between the two offsets. Boundary values are NaN
    when no text has one boundary.
    """
    if not (len(texts) == len(true_ai_spans) == len(scored)):
        raise ValueError("texts, true_ai_spans and scored must have the same length")
    tp = fp = fn = 0
    per_text_f1: list[float] = []
    errors: list[int] = []
    for text, spans, row in zip(texts, true_ai_spans, scored):
        n = len(text)
        chunks = sorted(row["chunks"], key=lambda c: (c[0], c[1]))
        nonspace = _nonspace_mask(text)
        true_mask = ai_char_mask(n, spans)
        pred_mask = ai_char_mask(n, [(s, e) for s, e, p in chunks if p >= threshold])
        t_tp = int(np.sum(true_mask & pred_mask & nonspace))
        t_fp = int(np.sum(~true_mask & pred_mask & nonspace))
        t_fn = int(np.sum(true_mask & ~pred_mask & nonspace))
        tp, fp, fn = tp + t_tp, fp + t_fp, fn + t_fn
        denom = 2 * t_tp + t_fp + t_fn
        per_text_f1.append(2 * t_tp / denom if denom else 1.0)

        boundary = _single_boundary(true_mask, nonspace)
        if boundary is None:
            continue
        flips = [chunks[j][0] for j in range(1, len(chunks))
                 if (chunks[j][2] >= threshold) != (chunks[j - 1][2] >= threshold)]
        candidates = flips or [0, n]
        word_starts = np.array([m.start() for m in _WORD.finditer(text)], dtype=np.int64)
        errors.append(min(_words_between(word_starts, boundary, c) for c in candidates))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "n_texts": len(texts),
        "char_precision": float(precision),
        "char_recall": float(recall),
        "char_f1": float(f1),
        "macro_f1": float(np.mean(per_text_f1)) if per_text_f1 else math.nan,
        "n_boundary_texts": len(errors),
        "boundary_error_mean": float(np.mean(errors)) if errors else math.nan,
        "boundary_error_median": float(np.median(errors)) if errors else math.nan,
    }


# --------------------------------------------------------------------------- #
# Highlighting
# --------------------------------------------------------------------------- #
def highlight(text: str, chunks: Sequence[Sequence[float]],
              threshold: float = AI_THRESHOLD) -> list[dict]:
    """
    Merge neighbouring chunks with the same label into display spans.

    Returns [{"start", "end", "p", "label": "ai" | "human"}, ...] in text order;
    a merged span runs from its first chunk's start to its last chunk's end
    (clipped to the text), and its p is the length-weighted mean chunk score.
    """
    out: list[dict] = []
    weights: list[float] = []
    n = len(text)
    for s, e, p in sorted(chunks, key=lambda c: (c[0], c[1])):
        s, e, p = max(0, int(s)), min(n, int(e)), float(p)
        label = "ai" if p >= threshold else "human"
        w = float(max(0, e - s))
        if out and out[-1]["label"] == label:
            last = out[-1]
            total = weights[-1] + w
            last["p"] = ((last["p"] * weights[-1] + p * w) / total if total
                         else (last["p"] + p) / 2)
            last["end"] = max(last["end"], e)
            weights[-1] = total
        else:
            out.append({"start": s, "end": e, "p": p, "label": label})
            weights.append(w)
    return out
