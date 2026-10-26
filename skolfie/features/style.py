"""
skolfie/features/style.py
-------------------------
A compact, fast and explainable set of StyloAI-style stylometric features
for the AI-vs-human LightGBM classifier. It replaces the 128-feature
`stylometric.py` extractor for modelling (that module stays for the app).

Design rules:
    * Pure Python regexes and word lists plus numpy: no NLTK, no spaCy, no
      downloads, fully deterministic.
    * Length-normalised only. Every feature is a rate per 100 words, a
      share, a mean or a coefficient of variation. There is no raw word,
      sentence or character count, because the corpus sources differ in
      length distribution and the model must not learn the source.
      Vocabulary richness and repetition are measured in fixed-size word
      windows (MATTR-50, hapax per 50-word window, n-gram repetition per
      100-word window) so they do not drift with text length.
    * No layout features (newlines, paragraphs, markdown). Newlines are
      only used as sentence boundaries.
    * Every value is finite for any input; undefined values are 0.0.

Feature groups (41 features, all prefixed "sty_"):
    1. Lexical         word length, long words, MATTR, hapax, Yule's K, Flesch
    2. Word classes    function words, pronouns, determiners, conjunctions,
                       prepositions, auxiliaries, negations, hedges, discourse
                       markers, informal markers, contractions
    3. Sentences       mean length, burstiness (CV), short/long share,
                       sentence-start diversity, capitalised starts
    4. Punctuation     , ; : ! ? quotes parentheses dashes ellipses
    5. Characters      digits, uppercase, ALL-CAPS words, non-ASCII
    6. Repetition      repeated word bigrams / trigrams

Words are runs of letters, optionally joined by an apostrophe or hyphen
("don't", "well-known"); numbers are not words. Word lists are matched on
the lowercased word, and contractions on their head ("I'm" -> "i" plus an
auxiliary, "don't" -> "do" plus a negation).

Use `style_features(text)` for one text (float64 vector ordered like
`STYLE_FEATURES`) and `style_matrix(texts, n_jobs=...)` for a corpus
(float32 matrix). `FEATURE_DESCRIPTIONS` holds one plain-English sentence
per feature for user-facing explanations.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# Feature names and descriptions (the order here is the vector order)
# --------------------------------------------------------------------------
_FEATURES: tuple[tuple[str, str], ...] = (
    # 1. Lexical
    ("sty_word_len_mean", "Average word length in letters."),
    ("sty_word_len_std", "Standard deviation of word length in letters."),
    ("sty_long_word_share", "Share of words with 7 or more letters."),
    ("sty_mattr50", "Share of distinct words in a sliding 50-word window, averaged "
                    "(moving-average type/token ratio)."),
    ("sty_hapax50", "Share of words that occur only once within a 50-word window, averaged "
                    "over all windows."),
    ("sty_yule_k", "Yule's K vocabulary repetitiveness (higher means a narrower, more "
                   "repetitive vocabulary)."),
    ("sty_flesch", "Flesch reading ease score (higher means easier to read)."),
    # 2. Word classes
    ("sty_function_word_share", "Share of words that are function words (the, of, and, it...)."),
    ("sty_pron_1st", "First-person pronouns (I, me, we, our...) per 100 words."),
    ("sty_pron_2nd", "Second-person pronouns (you, your...) per 100 words."),
    ("sty_pron_3rd", "Third-person pronouns (he, she, it, they...) per 100 words."),
    ("sty_determiners", "Articles and determiners (the, a, this, each...) per 100 words."),
    ("sty_coord_conj", "Coordinating conjunctions (and, but, or, so...) per 100 words."),
    ("sty_prepositions", "Prepositions (of, in, with, through...) per 100 words."),
    ("sty_aux_modals", "Auxiliary and modal verbs (is, have, can, should...) per 100 words."),
    ("sty_negations", "Negations (not, never, no, n't...) per 100 words."),
    ("sty_hedges", "Hedging words (perhaps, maybe, likely, seems...) per 100 words."),
    ("sty_discourse_markers", "Formal discourse markers (however, moreover, overall, "
                              "\"it is important to note\"...) per 100 words."),
    ("sty_informal", "Intensifiers and informal words (really, very, lol, gonna...) "
                     "per 100 words."),
    ("sty_contractions", "Contractions (don't, it's, I'm...) per 100 words."),
    # 3. Sentences
    ("sty_sent_len_mean", "Average sentence length in words."),
    ("sty_sent_len_cv", "Variation of sentence length (standard deviation divided by mean); "
                        "low values mean uniform sentences."),
    ("sty_short_sent_share", "Share of sentences with fewer than 8 words."),
    ("sty_long_sent_share", "Share of sentences with more than 30 words."),
    ("sty_sent_start_diversity", "Share of distinct sentence-opening words within windows of "
                                 "10 sentences."),
    ("sty_sent_cap_share", "Share of sentences that start with a capital letter."),
    # 4. Punctuation
    ("sty_commas", "Commas per 100 words."),
    ("sty_semicolons", "Semicolons per 100 words."),
    ("sty_colons", "Colons per 100 words."),
    ("sty_exclamations", "Exclamation marks per 100 words."),
    ("sty_questions", "Question marks per 100 words."),
    ("sty_quotes", "Quotation marks (double and single) per 100 words."),
    ("sty_parentheses", "Parentheses per 100 words."),
    ("sty_dashes", "Dashes (en dash, em dash, double or spaced hyphen) per 100 words."),
    ("sty_ellipses", "Ellipses (... or the ellipsis character) per 100 words."),
    # 5. Characters
    ("sty_digit_share", "Share of non-space characters that are digits."),
    ("sty_upper_share", "Share of letters that are uppercase."),
    ("sty_allcaps_words", "Fully capitalised words of 2 or more letters per 100 words."),
    ("sty_non_ascii_share", "Share of non-space characters outside plain ASCII."),
    # 6. Repetition
    ("sty_bigram_repeat", "Share of word pairs that repeat within a 100-word window."),
    ("sty_trigram_repeat", "Share of three-word sequences that repeat within a 100-word window."),
)

STYLE_FEATURES: tuple[str, ...] = tuple(name for name, _ in _FEATURES)
FEATURE_DESCRIPTIONS: dict[str, str] = dict(_FEATURES)
_D = len(STYLE_FEATURES)

# --------------------------------------------------------------------------
# Word lists (lowercase)
# --------------------------------------------------------------------------
_PRON_1ST = frozenset("i me my mine myself we us our ours ourselves".split())
_PRON_2ND = frozenset("you your yours yourself yourselves ya u ur y".split())
_PRON_3RD = frozenset(
    "he him his himself she her hers herself it its itself "
    "they them their theirs themselves".split()
)
_DETERMINERS = frozenset(
    "the a an this that these those each every some any another either neither "
    "both all such".split()
)
_COORD_CONJ = frozenset("and but or nor yet so".split())
_PREPOSITIONS = frozenset(
    "about above across after against along amid among amongst around as at before "
    "behind below beneath beside besides between beyond by despite down during except "
    "for from in inside into like near of off on onto out outside over past per "
    "regarding since through throughout till to toward towards under underneath unlike "
    "until up upon via with within without".split()
)
_AUX_MODALS = frozenset(
    "am is are was were be been being have has had having do does did will would "
    "shall should can could may might must ought need".split()
)
_NEGATIONS = frozenset("not no never nor none nobody nothing nowhere neither cannot".split())
_HEDGES = frozenset(
    "perhaps maybe might possibly probably likely unlikely seems seem seemed seemingly "
    "appear appears appeared apparently arguably somewhat presumably potentially "
    "roughly approximately generally typically usually often suggest suggests "
    "suggested tend tends".split()
)
_INFORMAL = frozenset(
    "really very so totally extremely super quite pretty absolutely literally actually "
    "basically honestly seriously definitely just lol lmao omg haha hahaha gonna wanna "
    "gotta kinda sorta yeah yep yup nope ok okay dude btw idk tbh imo imho stuff "
    "awesome cool".split()
)
_OTHER_FUNCTION = frozenset(
    "if because while although though unless whether when whenever where wherever why "
    "how who whom whose which what than then there here also once".split()
)
_FUNCTION_WORDS = (
    _PRON_1ST | _PRON_2ND | _PRON_3RD | _DETERMINERS | _COORD_CONJ | _PREPOSITIONS
    | _AUX_MODALS | _NEGATIONS | _OTHER_FUNCTION
) - {"ya", "u", "ur", "y"}

# AI-typical discourse markers: single words are looked up per word,
# phrases are counted on the space-joined lowercase word sequence.
_DISCOURSE_WORDS = frozenset(
    "however moreover furthermore additionally overall ultimately importantly notably "
    "consequently therefore thus hence nevertheless nonetheless conversely similarly "
    "likewise specifically essentially significantly crucially indeed".split()
)
_DISCOURSE_PHRASES = (
    "it is important to note", "it's important to note", "it is important to remember",
    "it is worth noting", "it's worth noting", "it is essential", "it is crucial",
    "in conclusion", "in summary", "to summarize", "to sum up", "in addition",
    "as a result", "on the other hand", "in other words", "first and foremost",
    "in particular", "for example", "for instance", "in today's",
)
_DISCOURSE_PATTERNS = tuple(f" {p} " for p in _DISCOURSE_PHRASES)

# Category index per word -> one dict lookup per word in the hot loop.
(_C_FUNC, _C_P1, _C_P2, _C_P3, _C_DET, _C_CC, _C_PREP, _C_AUX, _C_NEG, _C_HEDGE,
 _C_INF, _C_DISC) = range(12)
_N_CATS = 12


def _build_lexicon() -> dict[str, tuple[int, ...]]:
    groups = (
        (_C_FUNC, _FUNCTION_WORDS), (_C_P1, _PRON_1ST), (_C_P2, _PRON_2ND),
        (_C_P3, _PRON_3RD), (_C_DET, _DETERMINERS), (_C_CC, _COORD_CONJ),
        (_C_PREP, _PREPOSITIONS), (_C_AUX, _AUX_MODALS), (_C_NEG, _NEGATIONS),
        (_C_HEDGE, _HEDGES), (_C_INF, _INFORMAL), (_C_DISC, _DISCOURSE_WORDS),
    )
    lex: dict[str, list[int]] = {}
    for idx, words in groups:
        for w in words:
            lex.setdefault(w, []).append(idx)
    return {w: tuple(v) for w, v in lex.items()}


_LEXICON = _build_lexicon()

# Contractions. "n't" heads that differ from the verb, clitics that are
# auxiliaries, and the unambiguous "'s" contractions (other "'s" are
# treated as possessives).
_NT_HEADS = {"wo": "will", "ca": "can", "sha": "shall", "ai": "is"}
_CLITIC_AUX = frozenset(("re", "ve", "ll", "d", "m"))
_S_CONTRACTIONS = frozenset(
    "it's that's he's she's there's here's what's who's where's when's why's how's "
    "let's everyone's something's nothing's".split()
)

# --------------------------------------------------------------------------
# Precompiled regexes and constants
# --------------------------------------------------------------------------
_LETTER = r"[^\W\d_]"
_WORD_RE = re.compile(rf"{_LETTER}+(?:['’\-]{_LETTER}+)*")
_SENT_SPLIT_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+|\n\s*")
_UPPER_RE = re.compile(r"[A-ZÀ-ÖØ-Þ]")
_DIGIT_RE = re.compile(r"\d")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_DOT_RUN_RE = re.compile(r"\.{2,}")
_HYPHEN_RUN_RE = re.compile(r"-{2,}")
_DOUBLE_QUOTE_CHARS = "\"“”„«»"
_SINGLE_QUOTE_CHARS = "'‘’"

_VOWELS = frozenset("aeiouy")
_WINDOW = 50
_REPEAT_WINDOW = 100
_REPEAT_STRIDE = 50
_SENT_WINDOW = 10


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@lru_cache(maxsize=1 << 16)
def _syllables(word: str) -> int:
    """Vowel-group syllable heuristic (lowercase input), at least 1."""
    count = 0
    prev = False
    for ch in word:
        vowel = ch in _VOWELS
        if vowel and not prev:
            count += 1
        prev = vowel
    if count > 1 and word.endswith("e") and not word.endswith("le"):
        count -= 1
    return count if count > 0 else 1


def _window_richness(words: list[str], w: int = _WINDOW) -> tuple[float, float]:
    """MATTR and mean hapax share over sliding windows of `w` words.

    Texts shorter than `w` fall back to plain TTR and hapax/N.
    """
    n = len(words)
    if n == 0:
        return 0.0, 0.0
    if n <= w:
        c = Counter(words)
        hapax = sum(1 for v in c.values() if v == 1)
        return len(c) / n, hapax / n
    counts: dict[str, int] = {}
    types = hapax = 0
    for x in words[:w]:
        c = counts.get(x, 0)
        if c == 0:
            types += 1
            hapax += 1
        elif c == 1:
            hapax -= 1
        counts[x] = c + 1
    sum_t, sum_h = types, hapax
    for i in range(w, n):
        old = words[i - w]
        c = counts[old]
        if c == 1:
            types -= 1
            hapax -= 1
        elif c == 2:
            hapax += 1
        counts[old] = c - 1
        new = words[i]
        c = counts.get(new, 0)
        if c == 0:
            types += 1
            hapax += 1
        elif c == 1:
            hapax -= 1
        counts[new] = c + 1
        sum_t += types
        sum_h += hapax
    denom = (n - w + 1) * w
    return sum_t / denom, sum_h / denom


def _yules_k(words: list[str]) -> float:
    """Yule's K = 10^4 * (sum_i f_i^2 - N) / N^2 over word frequencies f_i."""
    n = len(words)
    if n == 0:
        return 0.0
    s2 = sum(f * f for f in Counter(words).values())
    return 1e4 * (s2 - n) / (n * n)


def _window_starts(n: int, w: int, stride: int) -> list[int]:
    if n <= w:
        return [0]
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts


def _repeat_share(ids: list[int], order: int) -> float:
    """Mean share of n-gram tokens whose type repeats, over 100-word windows."""
    n = len(ids)
    if n < order:
        return 0.0
    m = 1_000_003
    if order == 2:
        grams = [a * m + b for a, b in zip(ids, ids[1:])]
    else:
        grams = [(a * m + b) * m + c for a, b, c in zip(ids, ids[1:], ids[2:])]
    per_window = _REPEAT_WINDOW - order + 1
    total = 0.0
    k = 0
    for s in _window_starts(n, _REPEAT_WINDOW, _REPEAT_STRIDE):
        chunk = grams[s:s + per_window]
        rep = sum(v for v in Counter(chunk).values() if v > 1)
        total += rep / len(chunk)
        k += 1
    return total / k


def _sent_start_diversity(firsts: list[str]) -> float:
    """Distinct sentence-opening words per sentence, in sliding 10-sentence windows."""
    n = len(firsts)
    if n == 0:
        return 0.0
    if n <= _SENT_WINDOW:
        return len(set(firsts)) / n
    total = 0
    for s in range(n - _SENT_WINDOW + 1):
        total += len(set(firsts[s:s + _SENT_WINDOW]))
    return total / ((n - _SENT_WINDOW + 1) * _SENT_WINDOW)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def style_features(text: str) -> np.ndarray:
    """Return the style feature vector (float64, ordered like STYLE_FEATURES)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    f = dict.fromkeys(STYLE_FEATURES, 0.0)

    # ---- characters --------------------------------------------------------
    n_nonspace = len("".join(text.split()))
    if n_nonspace:
        f["sty_digit_share"] = len(_DIGIT_RE.findall(text)) / n_nonspace
        f["sty_non_ascii_share"] = len(_NON_ASCII_RE.findall(text)) / n_nonspace

    # ---- sentences and words (one word regex pass, per sentence) -----------
    tokens: list[str] = []
    sent_lens: list[int] = []
    firsts: list[str] = []
    capitalised = 0
    for sent in _SENT_SPLIT_RE.split(text):
        words = _WORD_RE.findall(sent)
        if not words:
            continue
        tokens.extend(words)
        sent_lens.append(len(words))
        first = words[0]
        firsts.append(first.lower())
        if first[0].isupper():
            capitalised += 1
    n = len(tokens)
    if n == 0:
        return np.fromiter(f.values(), dtype=np.float64, count=_D)
    per100 = 100.0 / n

    # ---- per-word pass -----------------------------------------------------
    lower: list[str] = []
    cats = [0] * _N_CATS
    len_sum = len_sq = long_words = syl_sum = allcaps = apostrophes = contractions = 0
    lexicon_get = _LEXICON.get
    for tok in tokens:
        lw = tok.lower()
        n_let = len(lw)
        if "-" in lw:
            n_let -= lw.count("-")
        head = lw
        if "'" in lw or "’" in lw:
            lw = lw.replace("’", "'")
            n_apos = lw.count("'")
            n_let -= n_apos
            apostrophes += n_apos
            if lw.endswith("n't"):
                head = lw[:-3]
                head = _NT_HEADS.get(head, head)
                cats[_C_NEG] += 1
                contractions += 1
            else:
                head, _, tail = lw.partition("'")
                if tail in _CLITIC_AUX or lw in _S_CONTRACTIONS:
                    cats[_C_AUX] += 1
                    contractions += 1
        lower.append(lw)
        len_sum += n_let
        len_sq += n_let * n_let
        if n_let >= 7:
            long_words += 1
        syl_sum += _syllables(lw)
        if n_let >= 2 and tok.isupper():
            allcaps += 1
        hit = lexicon_get(head)
        if hit is not None:
            for c in hit:
                cats[c] += 1

    mean_len = len_sum / n
    f["sty_word_len_mean"] = mean_len
    f["sty_word_len_std"] = max(len_sq / n - mean_len * mean_len, 0.0) ** 0.5
    f["sty_long_word_share"] = long_words / n
    f["sty_mattr50"], f["sty_hapax50"] = _window_richness(lower)
    f["sty_yule_k"] = _yules_k(lower)
    f["sty_allcaps_words"] = allcaps * per100
    # Every letter belongs to some word token, so len_sum is the letter count.
    if len_sum:
        f["sty_upper_share"] = len(_UPPER_RE.findall(text)) / len_sum

    joined = " " + " ".join(lower) + " "
    n_phrases = sum(joined.count(p) for p in _DISCOURSE_PATTERNS)
    f["sty_function_word_share"] = cats[_C_FUNC] / n
    f["sty_pron_1st"] = cats[_C_P1] * per100
    f["sty_pron_2nd"] = cats[_C_P2] * per100
    f["sty_pron_3rd"] = cats[_C_P3] * per100
    f["sty_determiners"] = cats[_C_DET] * per100
    f["sty_coord_conj"] = cats[_C_CC] * per100
    f["sty_prepositions"] = cats[_C_PREP] * per100
    f["sty_aux_modals"] = cats[_C_AUX] * per100
    f["sty_negations"] = cats[_C_NEG] * per100
    f["sty_hedges"] = cats[_C_HEDGE] * per100
    f["sty_discourse_markers"] = (cats[_C_DISC] + n_phrases) * per100
    f["sty_informal"] = cats[_C_INF] * per100
    f["sty_contractions"] = contractions * per100

    # ---- sentence statistics -----------------------------------------------
    n_sent = len(sent_lens)
    lens = np.asarray(sent_lens, dtype=np.float64)
    mean_s = n / n_sent
    f["sty_sent_len_mean"] = mean_s
    f["sty_sent_len_cv"] = float(lens.std()) / mean_s
    f["sty_short_sent_share"] = float((lens < 8).sum()) / n_sent
    f["sty_long_sent_share"] = float((lens > 30).sum()) / n_sent
    f["sty_sent_start_diversity"] = _sent_start_diversity(firsts)
    f["sty_sent_cap_share"] = capitalised / n_sent
    flesch = 206.835 - 1.015 * mean_s - 84.6 * (syl_sum / n)
    f["sty_flesch"] = min(max(flesch, -50.0), 120.0)

    # ---- punctuation -------------------------------------------------------
    f["sty_commas"] = text.count(",") * per100
    f["sty_semicolons"] = text.count(";") * per100
    f["sty_colons"] = text.count(":") * per100
    f["sty_exclamations"] = text.count("!") * per100
    f["sty_questions"] = text.count("?") * per100
    n_quotes = sum(text.count(ch) for ch in _DOUBLE_QUOTE_CHARS)
    # Single quote marks minus the word-internal apostrophes.
    n_quotes += max(sum(text.count(ch) for ch in _SINGLE_QUOTE_CHARS) - apostrophes, 0)
    f["sty_quotes"] = n_quotes * per100
    f["sty_parentheses"] = (text.count("(") + text.count(")")) * per100
    n_dashes = text.count("–") + text.count("—") + text.count(" - ")
    if "--" in text:
        n_dashes += len(_HYPHEN_RUN_RE.findall(text))
    f["sty_dashes"] = n_dashes * per100
    n_ellipses = text.count("…")
    if ".." in text:
        n_ellipses += len(_DOT_RUN_RE.findall(text))
    f["sty_ellipses"] = n_ellipses * per100

    # ---- repetition --------------------------------------------------------
    vocab: dict[str, int] = {}
    ids = [vocab.setdefault(w, len(vocab)) for w in lower]
    f["sty_bigram_repeat"] = _repeat_share(ids, 2)
    f["sty_trigram_repeat"] = _repeat_share(ids, 3)

    out = np.fromiter(f.values(), dtype=np.float64, count=_D)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _matrix_chunk(texts: Sequence[str]) -> np.ndarray:
    out = np.empty((len(texts), _D), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = style_features(t)
    return out


def style_matrix(texts: Sequence[str], n_jobs: int = 1, chunk_size: int = 2000) -> np.ndarray:
    """Feature matrix of shape (len(texts), len(STYLE_FEATURES)), float32.

    `n_jobs != 1` processes chunks of `chunk_size` texts in parallel with
    joblib (-1 = all cores). Results are identical to the serial path.
    """
    texts = list(texts)
    chunk_size = max(int(chunk_size), 1)
    if not texts:
        return np.zeros((0, _D), dtype=np.float32)
    if n_jobs == 1 or len(texts) <= chunk_size:
        return _matrix_chunk(texts)
    from joblib import Parallel, delayed

    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_matrix_chunk)(c) for c in chunks)
    return np.vstack(parts)
