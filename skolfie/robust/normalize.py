"""
skolfie/robust/normalize.py
----------------------------
Roadmap step 9, the defence against the Unicode attacks: every model maps a
text through `normalize_for_scoring()` before scoring it (and before
training on it), so look-alike letters and invisible characters no longer
change what the models see.

    1. Invisible format characters are deleted: zero-width space, LRM / RLM,
       bidi embeddings and overrides, word joiner and the invisible math
       operators, BOM, soft hyphen, Mongolian vowel separator, combining
       grapheme joiner. ZWJ / ZWNJ are kept only between two non-ASCII,
       non-Latin characters (emoji sequences, Indic and Arabic-script words
       need them) and deleted elsewhere, e.g. inside Latin words.
    2. Letters and digits with a compatibility form are folded to it (NFKC
       per character): fullwidth "ｔｅｘｔ", mathematical "𝐭𝐞𝐱𝐭", ligatures
       such as "ﬁ". Everything else keeps its form (see `_fold`), then the
       text is NFC-composed, like the corpus.
    3. Cyrillic, Greek and IPA look-alikes of Latin letters (`CONFUSABLES`)
       become the Latin letter, but only in words that are mostly Latin
       (a word with any other non-Latin letter is never touched):
         * a word with Latin letters, at least as many as its look-alikes
           ("tеxt" with a Cyrillic е, "IТ");
         * a word of (mostly) Cyrillic look-alikes, when the two nearest
           other words on each side are Latin, at least two in all (the
           article "a" written as Cyrillic "а"; "А cat sat").
       So Russian, Ukrainian, Serbian or Greek text, a Russian word with a
       Latin typo, and Greek letters used as symbols ("the α level") stay
       as they are.

`obfuscation_stats(text)` counts what step 1 deleted ("invisible", except a
leading BOM) and how many look-alike letters steps 2 and 3 mapped to plain
Latin letters ("lookalikes"; ligatures and other multi-letter folds are not
counted). Ordinary text has neither, so a non-zero count is itself a hint
that someone tried to hide the text from a detector; `obfuscation_note()`
words it for the explanations and the app.

Plain-ASCII text (most of the corpus) is returned at once. Otherwise it is
a few `str.translate` calls and regex passes, plus a loop over the words
for texts that contain a look-alike. The result is idempotent, and
non-string input gives "" like `skolfie.data.corpus.clean_text`.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from functools import lru_cache
from typing import Any

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
_INVISIBLE = dict.fromkeys([
    0x00AD,                     # soft hyphen
    0x034F,                     # combining grapheme joiner
    0x180E,                     # Mongolian vowel separator
    0x200B, 0x200E, 0x200F,     # zero-width space, LRM, RLM
    *range(0x202A, 0x202F),     # bidi embeddings / overrides
    *range(0x2060, 0x2065),     # word joiner, invisible math operators
    0xFEFF,                     # BOM / zero-width no-break space
])
_JOINER_RE = re.compile("[\u200c\u200d]")  # ZWNJ, ZWJ

# Look-alike -> Latin letter. Wider than the attack's own `HOMOGLYPHS` map on
# purpose, so the defence is not tuned to our own test. Lowercase Cyrillic
# в к м н т look like small capitals and map to the nearest Latin letter.
CONFUSABLES: dict[str, str] = {
    # Cyrillic, lowercase
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "һ": "h", "н": "h",
    "і": "i", "ј": "j", "к": "k", "ӏ": "l", "м": "m", "о": "o", "р": "p",
    "ԛ": "q", "ѕ": "s", "т": "t", "ѵ": "v", "ԝ": "w", "х": "x", "у": "y",
    "ү": "y",
    # Cyrillic, uppercase
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ӏ": "I",
    "Ј": "J", "К": "K", "М": "M", "О": "O", "Р": "P", "Ԛ": "Q", "Ѕ": "S",
    "Т": "T", "Ѵ": "V", "Ԝ": "W", "Х": "X", "У": "Y", "Ү": "Y",
    # Greek, lowercase
    "α": "a", "ϲ": "c", "ε": "e", "ι": "i", "ϳ": "j", "κ": "k", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "ν": "v", "χ": "x",
    # Greek, uppercase
    "Α": "A", "Β": "B", "Ϲ": "C", "Ε": "E", "Η": "H", "Ι": "I", "Ϳ": "J",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y",
    "Χ": "X", "Ζ": "Z",
    # Latin IPA letters
    "ɑ": "a", "ɡ": "g",
}
_CONFUSABLE_TABLE = str.maketrans(CONFUSABLES)
_CONFUSABLE_RE = re.compile("[" + "".join(CONFUSABLES) + "]")
_GREEK = frozenset(c for c in CONFUSABLES if "Ͱ" <= c <= "Ͽ")

# A word: letters, and the combining marks that belong to them (so that the
# final NFC composition never joins two words).
_MARKS = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
_WORD_RE = re.compile(rf"[^\W\d_](?:[^\W\d_]|[{_MARKS}])*")


def _is_latin(c: str) -> bool:
    """A Latin letter: ASCII, Latin-1 / Extended-A / Extended-B, Extended Additional."""
    if c.isascii():
        return c.isalpha()
    return ("\u00c0" <= c <= "\u024f" or "\u1e00" <= c <= "\u1eff") and c.isalpha()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _joiner(m: re.Match[str]) -> str:
    """Keep a ZWJ / ZWNJ only between two non-ASCII, non-Latin characters."""
    s, i = m.string, m.start()
    before = s[i - 1] if i > 0 else " "
    after = s[i + 1] if i + 1 < len(s) else " "
    keep = all(not c.isascii() and not _is_latin(c) for c in (before, after))
    return m.group() if keep else ""


@lru_cache(maxsize=8192)
def _fold(c: str) -> str | None:
    """NFKC form of a letter or decimal digit that has one, else None.

    Why not NFKC on the whole text: it would also rewrite characters the
    style features and the models were trained on, e.g. "…" -> "..."
    (sty_ellipses, sty_non_ascii_share), "²" -> "2" and "½" -> "1⁄2" (digits,
    word splits), no-break and other Unicode spaces, "™" -> "TM" and
    fullwidth punctuation. Letters and digits are what the evasion tricks
    disguise, so only they are folded; the rest of the text keeps its form.
    """
    if not (c.isalpha() or c.isdecimal()):
        return None
    folded = unicodedata.normalize("NFKC", c)
    return folded if folded != c else None


def _fold_letters(text: str) -> tuple[str, int]:
    table: dict[int, str] = {}
    lookalikes = 0
    for c in set(text):
        if c.isascii():
            continue
        folded = _fold(c)
        if folded is None:
            continue
        table[ord(c)] = folded
        # A disguised letter, not a ligature or other multi-letter fold; Latin-1's
        # ordinal indicators ("1º") are ordinary text.
        if len(folded) == 1 and folded.isascii() and ord(c) >= 0x100:
            lookalikes += text.count(c)
    return (text.translate(table) if table else text), lookalikes


def _map_confusables(text: str) -> tuple[str, int]:
    words = list(_WORD_RE.finditer(text))
    n = len(words)
    latin = [False] * n      # a Latin word, for the candidates' neighbourhood
    mapped = [False] * n
    candidates = []          # Cyrillic look-alike words that need Latin neighbours
    for i, m in enumerate(words):
        w = m.group()
        if w.isascii():
            latin[i] = True
            continue
        n_latin = n_conf = n_other = 0
        greek = False
        for c in w:
            if c in CONFUSABLES:
                n_conf += 1
                greek = greek or c in _GREEK
            elif _is_latin(c):
                n_latin += 1
            elif c.isalpha():
                n_other += 1
        if n_other:
            continue
        if n_latin and n_latin >= n_conf:
            latin[i] = True
            mapped[i] = n_conf > 0
        elif n_conf and not greek:
            candidates.append(i)

    # A candidate is mapped when the two nearest other words on each side (fewer
    # at the ends of the text, but at least two in all) are Latin. Other
    # candidates are skipped, so a run of them ("i.е. а mix") is judged as one,
    # and mapping never changes another candidate's neighbours: normalising
    # twice maps nothing more.
    if candidates:
        skip = set(candidates)
        others = [i for i in range(n) if i not in skip]
        for i in candidates:
            k = bisect.bisect_left(others, i)
            near = others[max(0, k - 2):k + 2]
            if len(near) >= 2 and all(latin[j] for j in near):
                mapped[i] = True

    if not any(mapped):
        return text, 0
    parts, pos, count = [], 0, 0
    for i, m in enumerate(words):
        if mapped[i]:
            w = m.group()
            count += sum(c in CONFUSABLES for c in w)
            parts += [text[pos:m.start()], w.translate(_CONFUSABLE_TABLE)]
            pos = m.end()
    parts.append(text[pos:])
    return "".join(parts), count


def _normalize(text: str) -> tuple[str, int, int]:
    """(normalised text, invisible characters deleted, look-alike letters mapped)."""
    if text.isascii():
        return text, 0, 0
    bom = text.startswith("\ufeff")
    out = text.translate(_INVISIBLE)
    if "\u200c" in out or "\u200d" in out:
        out = _JOINER_RE.sub(_joiner, out)
    invisible = len(text) - len(out) - bom
    out = unicodedata.normalize("NFC", out)
    out, lookalikes = _fold_letters(out)
    if _CONFUSABLE_RE.search(out):
        out, n_mapped = _map_confusables(out)
        lookalikes += n_mapped
    return unicodedata.normalize("NFC", out), invisible, lookalikes


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def normalize_for_scoring(text: Any) -> str:
    """The text as the models see it (see the module docstring); "" for non-strings."""
    if not isinstance(text, str):
        return ""
    return _normalize(text)[0]


def obfuscation_stats(text: Any) -> dict[str, int]:
    """{"invisible": n, "lookalikes": n}: what `normalize_for_scoring` deletes / maps."""
    if not isinstance(text, str):
        return {"invisible": 0, "lookalikes": 0}
    _, invisible, lookalikes = _normalize(text)
    return {"invisible": invisible, "lookalikes": lookalikes}


def obfuscation_note(stats: dict[str, Any] | None) -> str:
    """A plain-text warning for non-zero `obfuscation_stats`, else ""."""
    stats = stats or {}
    invisible, lookalikes = int(stats.get("invisible") or 0), int(stats.get("lookalikes") or 0)
    found = [f"{n:,} {what}{'' if n == 1 else 's'}" for n, what in (
        (invisible, "invisible character"), (lookalikes, "look-alike letter")) if n > 0]
    if not found:
        return ""
    return (f"The text contains {' and '.join(found)} (e.g. zero-width spaces, Cyrillic letters "
            "that look Latin). They were removed or replaced by plain letters before scoring; "
            "ordinary text rarely has them, so they may be an attempt to evade detection.")
