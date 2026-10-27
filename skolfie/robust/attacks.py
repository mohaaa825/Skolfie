"""
skolfie/robust/attacks.py
--------------------------
Roadmap step 9: evasion attacks on AI-text detection. Each attack rewrites a
list of texts and has a stable name; `get_attack(name)` returns its
`AttackSpec` and `apply_attack(name, frame, ctx)` runs it.

    name            what it does                                       needs    applied to
    typo_<pct>      character typos (swap / drop / keyboard-neighbour  -        ai + human
                    insert or substitute) in <pct>% of the words of
                    three or more letters
    synonym_<pct>   replaces <pct>% of the content words (not stop     -        ai + human
                    words) by a synonym; WordNet when available, else
                    the small bundled `SYNONYM_TABLE`
    homoglyph       swaps one letter for a Cyrillic look-alike in 20%  -        ai + human
                    of the words (`homoglyph_<pct>` for another rate)
    zero_width      inserts a zero-width space inside 20% of the words -        ai + human
                    (`zero_width_<pct>`)
    t5_paraphrase   paraphrases every sentence with a local T5 model   GPU      ai
                    (default `T5_MODEL`), keeping the text's layout
    llm_paraphrase  asks an LLM to paraphrase the whole text           API key  ai
    humanizer       asks an LLM to rewrite it so it reads as human     API key  ai
                    (varied sentence length, contractions, asides)
    temp_low/high   fresh generation at temperature 0.2 / 1.3: the     API key  ai
                    original prompt for "fresh:" ids, else a
                    continuation of the text's opening sentences
    human_edit_<pct>  splice, standing in for human-edited AI text     -        ai
                    (no such data exists): <pct>% of the sentences
                    are replaced by sentences of human texts of the
                    same source and domain

"ai + human" (`applies_to="both"`) attacks are also applied to human text,
to check that the false-positive rate does not explode under noise that
says nothing about authorship. Paraphrase-style and regeneration attacks
only make sense for AI text.

Everything is deterministic given the seed: each text gets its own random
generator seeded from (seed, attack name, text), so a text is attacked the
same way whatever else is in the batch. The LLM attacks go through
`skolfie.data.generation.providers` (use the `fake` provider in tests), the
T5 attack takes any callable `list[str] -> list[str]` as its paraphraser
(`T5Paraphraser` lazily loads transformers). An attack returns None for a
text it could not attack (an LLM refusal, no donor text for a splice); the
caller drops those.
"""

from __future__ import annotations

import hashlib
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from skolfie.mixed.chunking import split_sentences

T5_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
TEMPERATURES = {"temp_low": 0.2, "temp_high": 1.3}
LLM_TEMPERATURE = 0.7  # paraphrase / humanizer
MIN_LLM_WORDS = 20     # shorter LLM outputs are treated as failures

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttackSpec:
    name: str
    family: str
    description: str
    applies_to: str = "ai"     # "ai", or "both" (also applied to human text)
    needs_gpu: bool = False    # runs on CPU, but far too slowly for a real sample
    needs_api: bool = False    # needs an LLM provider (API key)
    rate: float | None = None  # share of words / sentences changed, where it applies
    params: dict[str, Any] = field(default_factory=dict)


_FAMILIES: dict[str, dict[str, Any]] = {
    "typo": {"applies_to": "both", "default": None,
             "description": "character typos in {pct}% of the words"},
    "synonym": {"applies_to": "both", "default": None,
                "description": "synonyms for {pct}% of the content words"},
    "homoglyph": {"applies_to": "both", "default": 20,
                  "description": "Cyrillic look-alike letter in {pct}% of the words"},
    "zero_width": {"applies_to": "both", "default": 20,
                   "description": "zero-width space inside {pct}% of the words"},
    "human_edit": {"applies_to": "ai", "default": None,
                   "description": "{pct}% of the sentences replaced by human sentences of "
                                  "the same source and domain (splice, not real human edits)"},
}
_FIXED: dict[str, AttackSpec] = {
    "t5_paraphrase": AttackSpec(
        "t5_paraphrase", "paraphrase", "every sentence paraphrased by a T5 model",
        needs_gpu=True),
    "llm_paraphrase": AttackSpec(
        "llm_paraphrase", "paraphrase", "whole text paraphrased by an LLM", needs_api=True),
    "humanizer": AttackSpec(
        "humanizer", "paraphrase", "LLM rewrite asked to sound human", needs_api=True),
    "temp_low": AttackSpec(
        "temp_low", "regenerate", f"fresh LLM generation at temperature "
        f"{TEMPERATURES['temp_low']}", needs_api=True,
        params={"temperature": TEMPERATURES["temp_low"]}),
    "temp_high": AttackSpec(
        "temp_high", "regenerate", f"fresh LLM generation at temperature "
        f"{TEMPERATURES['temp_high']}", needs_api=True,
        params={"temperature": TEMPERATURES["temp_high"]}),
}
_RATED = re.compile(r"^(typo|synonym|homoglyph|zero_width|human_edit)(?:_(\d{1,3}))?$")


def get_attack(name: str) -> AttackSpec:
    """The spec of an attack name, e.g. "typo_10", "homoglyph" or "t5_paraphrase"."""
    if name in _FIXED:
        return _FIXED[name]
    m = _RATED.match(name)
    if not m:
        raise ValueError(f"unknown attack {name!r}; known: {', '.join(ATTACKS)} "
                         "(rates are percentages, e.g. typo_10)")
    family, pct = m.group(1), m.group(2)
    info = _FAMILIES[family]
    if pct is None:
        if info["default"] is None:
            raise ValueError(f"attack {name!r} needs a rate in percent, e.g. {family}_10")
        pct = info["default"]
    pct = int(pct)
    if not 0 <= pct <= 100 or (family == "human_edit" and not 0 < pct < 100):
        raise ValueError(f"attack {name!r}: rate out of range")
    return AttackSpec(name, family, info["description"].format(pct=pct),
                      applies_to=info["applies_to"], rate=pct / 100)


# The default suite, in report order.
ATTACKS: dict[str, AttackSpec] = {}
for _name in ("typo_5", "typo_15", "synonym_15", "homoglyph", "zero_width", "t5_paraphrase",
              "llm_paraphrase", "humanizer", "temp_low", "temp_high", "human_edit_30"):
    ATTACKS[_name] = get_attack(_name)
PARAPHRASE_ATTACKS = ("t5_paraphrase", "llm_paraphrase", "humanizer")


def text_rng(seed: int, name: str, text: str) -> random.Random:
    """A generator that depends only on (seed, attack name, text)."""
    digest = hashlib.sha256(f"{seed}\x00{name}\x00{text}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _pick(rng: random.Random, candidates: Sequence[int], rate: float) -> list[int]:
    """round(rate * n) of the candidates (at least one if rate > 0 and any exist)."""
    if not candidates or rate <= 0:
        return []
    k = min(len(candidates), max(1, round(rate * len(candidates))))
    return sorted(rng.sample(list(candidates), k))


def _replace_spans(text: str, edits: Sequence[tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) edits; spans must not overlap."""
    out, pos = [], 0
    for start, end, new in sorted(edits):
        out += [text[pos:start], new]
        pos = end
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Typos
# --------------------------------------------------------------------------- #
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
KEYBOARD_NEIGHBOURS: dict[str, str] = {}
for _r, _row in enumerate(_KEYBOARD_ROWS):
    for _c, _ch in enumerate(_row):
        _near = [_row[j] for j in (_c - 1, _c + 1) if 0 <= j < len(_row)]
        for _dr in (-1, 1):
            if 0 <= _r + _dr < len(_KEYBOARD_ROWS):
                _other = _KEYBOARD_ROWS[_r + _dr]
                _near += [_other[j] for j in (_c - 1, _c, _c + 1) if 0 <= j < len(_other)]
        KEYBOARD_NEIGHBOURS[_ch] = "".join(dict.fromkeys(_near))

_TYPO_WORD = re.compile(r"[A-Za-z]{3,}")


def _match_case(ch: str, like: str) -> str:
    return ch.upper() if like.isupper() else ch


def typo_word(word: str, rng: random.Random) -> str:
    """One typo: swap two neighbours, drop, insert or substitute a keyboard neighbour."""
    i = rng.randrange(1, len(word))  # keep the first letter, as most real typos do
    op = rng.choice(("swap", "drop", "insert", "substitute"))
    j = i + 1 if i + 1 < len(word) else i - 1  # swap partner, never the first letter
    a, b = min(i, j), max(i, j)
    if op == "swap" and word[a].lower() != word[b].lower():
        return word[:a] + word[b] + word[a] + word[b + 1:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    near = KEYBOARD_NEIGHBOURS.get(word[i].lower(), "e")
    new = _match_case(rng.choice(near), word[i])
    if op == "insert":
        return word[:i] + new + word[i:]
    return word[:i] + new + word[i + 1:]  # substitute (and swap of a doubled letter)


def typo_text(text: str, rate: float, rng: random.Random) -> str:
    words = list(_TYPO_WORD.finditer(text))
    chosen = _pick(rng, range(len(words)), rate)
    return _replace_spans(text, [(words[i].start(), words[i].end(),
                                  typo_word(words[i].group(), rng)) for i in chosen])


# --------------------------------------------------------------------------- #
# Synonyms
# --------------------------------------------------------------------------- #
# Fallback when WordNet is not installed: common content words (base forms),
# AI-favoured ones included. Synonyms are close enough in most contexts.
SYNONYM_TABLE: dict[str, tuple[str, ...]] = {
    "important": ("key", "major", "significant"), "crucial": ("vital", "key", "critical"),
    "significant": ("major", "notable", "considerable"), "essential": ("vital", "necessary"),
    "various": ("different", "several", "many"), "numerous": ("many", "countless"),
    "additional": ("extra", "further"), "however": ("but", "still", "yet"),
    "additionally": ("also", "besides", "moreover"), "furthermore": ("also", "besides"),
    "moreover": ("also", "besides"), "therefore": ("so", "thus", "hence"),
    "ultimately": ("finally", "in the end"), "overall": ("in general", "on the whole"),
    "utilize": ("use", "employ"), "use": ("employ", "apply"), "help": ("aid", "assist"),
    "enhance": ("improve", "boost"), "improve": ("better", "boost"),
    "foster": ("encourage", "promote"), "leverage": ("use", "exploit"),
    "provide": ("give", "offer", "supply"), "ensure": ("make sure", "guarantee"),
    "demonstrate": ("show", "prove"), "show": ("reveal", "display"),
    "require": ("need", "demand"), "obtain": ("get", "acquire"), "get": ("obtain", "gain"),
    "begin": ("start", "commence"), "start": ("begin", "launch"), "end": ("finish", "close"),
    "allow": ("let", "permit", "enable"), "create": ("make", "build", "produce"),
    "make": ("create", "produce"), "build": ("construct", "create"),
    "consider": ("weigh", "think about"), "think": ("believe", "suppose"),
    "believe": ("think", "feel"), "explore": ("examine", "investigate"),
    "examine": ("inspect", "study"), "understand": ("grasp", "comprehend"),
    "include": ("contain", "cover"), "increase": ("rise", "grow", "raise"),
    "reduce": ("cut", "lower", "decrease"), "change": ("alter", "shift"),
    "big": ("large", "huge"), "large": ("big", "sizable"), "small": ("little", "tiny"),
    "little": ("small", "slight"), "good": ("fine", "decent", "great"),
    "great": ("excellent", "terrific"), "bad": ("poor", "awful"), "fast": ("quick", "rapid"),
    "quick": ("fast", "swift"), "slow": ("sluggish", "unhurried"), "hard": ("difficult", "tough"),
    "difficult": ("hard", "tough"), "easy": ("simple", "effortless"), "simple": ("easy", "plain"),
    "new": ("fresh", "novel"), "old": ("aged", "former"), "happy": ("glad", "cheerful"),
    "sad": ("unhappy", "gloomy"), "strong": ("powerful", "sturdy"), "weak": ("feeble", "frail"),
    "clear": ("plain", "obvious"), "obvious": ("clear", "evident"), "main": ("chief", "primary"),
    "complex": ("complicated", "intricate"), "effective": ("efficient", "successful"),
    "innovative": ("inventive", "original"), "comprehensive": ("thorough", "complete"),
    "robust": ("sturdy", "strong", "solid"), "unique": ("distinctive", "singular"),
    "diverse": ("varied", "mixed"), "vital": ("crucial", "essential"),
    "particular": ("specific", "certain"), "entire": ("whole", "complete"),
    "approach": ("method", "way"), "method": ("technique", "approach"),
    "aspect": ("feature", "facet"), "benefit": ("advantage", "gain"),
    "challenge": ("difficulty", "problem"), "problem": ("issue", "difficulty"),
    "issue": ("matter", "problem"), "goal": ("aim", "objective"), "result": ("outcome", "effect"),
    "outcome": ("result", "consequence"), "impact": ("effect", "influence"),
    "effect": ("impact", "result"), "idea": ("notion", "concept"), "way": ("manner", "means"),
    "people": ("folks", "individuals"), "person": ("individual", "someone"),
    "child": ("kid", "youngster"), "children": ("kids", "youngsters"), "job": ("work", "post"),
    "home": ("house", "dwelling"), "house": ("home", "residence"), "car": ("vehicle", "auto"),
    "city": ("town", "metropolis"), "road": ("street", "route"), "area": ("region", "zone"),
    "world": ("globe", "earth"), "society": ("community", "public"), "money": ("cash", "funds"),
    "time": ("period", "while"), "moment": ("instant", "second"), "story": ("tale", "account"),
    "friend": ("pal", "companion"), "study": ("research", "analysis"),
    "research": ("study", "investigation"), "landscape": ("scene", "terrain"),
    "insight": ("understanding", "perception"), "journey": ("trip", "voyage"),
    "really": ("truly", "very"), "very": ("extremely", "quite"), "often": ("frequently",),
    "usually": ("normally", "generally"), "quickly": ("rapidly", "swiftly"),
    "finally": ("eventually", "at last"), "maybe": ("perhaps", "possibly"),
    "perhaps": ("maybe", "possibly"), "almost": ("nearly", "virtually"),
    "especially": ("particularly", "notably"), "probably": ("likely", "presumably"),
}

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _stop_words() -> frozenset[str]:
    from skolfie.features.stylometric import FUNCTION_WORDS

    extra = ("am", "is", "are", "was", "were", "be", "been", "being", "does", "doing", "done",
             "having", "should", "shall", "must", "might", "very", "just", "also", "too",
             "here", "where", "why", "our", "ours", "us", "me", "mine", "yours", "its",
             "those", "any", "such", "only", "own", "same", "both", "few", "most", "nor")
    return frozenset(w.lower() for w in (*FUNCTION_WORDS, *extra))


class TableSynonyms:
    """Synonyms from a dict (the bundled `SYNONYM_TABLE` by default)."""

    name = "table"

    def __init__(self, table: dict[str, Sequence[str]] | None = None) -> None:
        self.table = {k.lower(): tuple(v) for k, v in (table or SYNONYM_TABLE).items()}

    def __call__(self, word: str) -> Sequence[str]:
        return self.table.get(word, ())


class WordNetSynonyms:
    """WordNet lemmas of the word's most common sense; base forms only, single words."""

    name = "wordnet"

    def __init__(self, max_candidates: int = 3) -> None:
        from nltk.corpus import wordnet

        wordnet.synsets("test")  # raises LookupError if the data is missing
        self.wn = wordnet
        self.max_candidates = max_candidates
        self._cache: dict[str, tuple[str, ...]] = {}

    def __call__(self, word: str) -> Sequence[str]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        out: list[str] = []
        synsets = self.wn.synsets(word)
        if synsets and self.wn.morphy(word, synsets[0].pos()) == word:
            pos = synsets[0].pos()
            counts: dict[str, int] = {}
            for syn in [s for s in synsets if s.pos() == pos][:2]:
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name != word and name.isalpha():
                        counts[name] = max(counts.get(name, 0), lemma.count())
            out = sorted(counts, key=lambda w: (-counts[w], w))[: self.max_candidates]
        self._cache[word] = tuple(out)
        return self._cache[word]


def load_synonyms(source: str = "auto", download: bool = False) -> Callable[[str], Sequence[str]]:
    """"wordnet", "table", or "auto" (WordNet if installed or downloadable, else the table)."""
    if source == "table":
        return TableSynonyms()
    if source not in ("auto", "wordnet"):
        raise ValueError(f"unknown synonym source {source!r}; use auto, wordnet or table")
    try:
        return WordNetSynonyms()
    except LookupError:
        if download:
            import nltk

            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                return WordNetSynonyms()
            except Exception:  # no network: fall through
                pass
        if source == "wordnet":
            raise
        return TableSynonyms()


def _match_word_case(new: str, like: str) -> str:
    if like.isupper() and len(like) > 1:
        return new.upper()
    if like[0].isupper():
        return new[0].upper() + new[1:]
    return new


def synonym_text(text: str, rate: float, rng: random.Random,
                 synonyms: Callable[[str], Sequence[str]],
                 stop_words: frozenset[str] | None = None) -> str:
    """Replace `rate` of the content words by a synonym (among those that have one)."""
    stop = _stop_words() if stop_words is None else stop_words
    words = [m for m in _WORD.finditer(text) if m.group().lower() not in stop
             and "'" not in m.group()]
    n_target = round(rate * len(words))
    options = [(m, synonyms(m.group().lower())) for m in words]
    options = [(m, s) for m, s in options if s]
    if n_target <= 0 or not options:
        return text
    chosen = rng.sample(options, min(n_target, len(options)))
    return _replace_spans(text, [(m.start(), m.end(), _match_word_case(rng.choice(list(s)),
                                                                        m.group()))
                                 for m, s in chosen])


# --------------------------------------------------------------------------- #
# Unicode tricks
# --------------------------------------------------------------------------- #
HOMOGLYPHS: dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о", "p": "р",
    "x": "х", "y": "у", "s": "ѕ", "j": "ј",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}
ZERO_WIDTH = "​"
_ANY_WORD = re.compile(r"\S+")


def homoglyph_text(text: str, rate: float, rng: random.Random) -> str:
    """In `rate` of the words that have one, swap one letter for its Cyrillic twin."""
    words = [m for m in _ANY_WORD.finditer(text) if any(c in HOMOGLYPHS for c in m.group())]
    edits = []
    for i in _pick(rng, range(len(words)), rate):
        m = words[i]
        pos = rng.choice([j for j, c in enumerate(m.group()) if c in HOMOGLYPHS])
        edits.append((m.start() + pos, m.start() + pos + 1, HOMOGLYPHS[m.group()[pos]]))
    return _replace_spans(text, edits)


def zero_width_text(text: str, rate: float, rng: random.Random) -> str:
    """Insert a zero-width space inside `rate` of the words of two or more characters."""
    words = [m for m in _ANY_WORD.finditer(text) if len(m.group()) >= 2]
    edits = []
    for i in _pick(rng, range(len(words)), rate):
        m = words[i]
        pos = m.start() + rng.randrange(1, len(m.group()))
        edits.append((pos, pos, ZERO_WIDTH))
    return _replace_spans(text, edits)


def undo_unicode_tricks(text: str) -> str:
    """Remove zero-width spaces and map the homoglyphs above back to Latin."""
    back = {v: k for k, v in HOMOGLYPHS.items()}
    return "".join(back.get(c, c) for c in text.replace(ZERO_WIDTH, ""))


# --------------------------------------------------------------------------- #
# Human-edit splice
# --------------------------------------------------------------------------- #
class DonorPool:
    """Human texts grouped by (source, domain), sentence-split lazily."""

    def __init__(self, texts: Sequence[str], sources: Sequence[str],
                 domains: Sequence[str]) -> None:
        self.texts = list(texts)
        self.groups: dict[tuple[str, str], list[int]] = {}
        for i, key in enumerate(zip(map(str, sources), map(str, domains))):
            self.groups.setdefault(key, []).append(i)
        self._sents: dict[int, list[str]] = {}

    def sentences(self, i: int) -> list[str]:
        if i not in self._sents:
            t = self.texts[i]
            self._sents[i] = [t[s:e] for s, e in split_sentences(t)]
        return self._sents[i]


def human_edit_text(text: str, source: str, domain: str, share: float, rng: random.Random,
                    donors: DonorPool) -> str | None:
    """Replace `share` of the sentences (at least one, never all) by human sentences.

    The human sentences are consecutive runs from random human texts of the
    same source and domain. None if the text has fewer than two sentences or
    there is no donor.
    """
    spans = split_sentences(text)
    pool = [i for i in donors.groups.get((str(source), str(domain)), [])
            if donors.texts[i] != text and donors.sentences(i)]
    if len(spans) < 2 or not pool:
        return None
    k = min(len(spans) - 1, max(1, round(share * len(spans))))
    chosen = sorted(rng.sample(range(len(spans)), k))
    supply: list[str] = []
    edits = []
    for j in chosen:
        if not supply:
            sents = donors.sentences(rng.choice(pool))
            supply = sents[rng.randrange(len(sents)):]
        edits.append((spans[j][0], spans[j][1], supply.pop(0)))
    return _replace_spans(text, edits)


# --------------------------------------------------------------------------- #
# T5 paraphrase
# --------------------------------------------------------------------------- #
class T5Paraphraser:
    """Sentence paraphraser around a seq2seq checkpoint; loads lazily, batches by length.

    Defaults follow the model card of `T5_MODEL` (prefix "paraphrase: ", beam
    search with a strong repetition penalty, which pushes the wording away
    from the input). fp32 on purpose: T5 is unstable in fp16.
    """

    def __init__(self, model_name: str = T5_MODEL, device: str | None = None,
                 batch_size: int = 32, max_input_tokens: int = 256, max_new_tokens: int = 128,
                 prefix: str = "paraphrase: ", log: Callable[[str], Any] = print,
                 **generate_kwargs: Any) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.prefix = prefix
        self.log = log
        self.generate_kwargs = {"num_beams": 5, "repetition_penalty": 10.0,
                                "no_repeat_ngram_size": 2, "max_new_tokens": max_new_tokens,
                                **generate_kwargs}
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self.log(f"[t5] loaded {self.model_name} on {self.device}")

    def __call__(self, sentences: Sequence[str]) -> list[str]:
        import torch

        if self._model is None:
            self._load()
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        out = [""] * len(sentences)
        n_batches = (len(order) + self.batch_size - 1) // self.batch_size
        for b, lo in enumerate(range(0, len(order), self.batch_size)):
            idx = order[lo:lo + self.batch_size]
            enc = self._tokenizer([self.prefix + sentences[i] for i in idx], return_tensors="pt",
                                  padding=True, truncation=True,
                                  max_length=self.max_input_tokens).to(self.device)
            with torch.inference_mode():
                gen = self._model.generate(**enc, **self.generate_kwargs)
            for i, t in zip(idx, self._tokenizer.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip()
            if (b + 1) % 50 == 0 or b + 1 == n_batches:
                self.log(f"[t5] {b + 1}/{n_batches} batches")
        return out


def paraphrase_texts(texts: Sequence[str], paraphraser: Callable[[list[str]], list[str]],
                     min_words: int = 3) -> list[str]:
    """Paraphrase every sentence of every text in one call to `paraphraser`.

    Each sentence is replaced in place, so paragraph breaks survive. Sentences
    under `min_words` words, and empty paraphrases, keep the original.
    """
    jobs: list[tuple[int, int, int]] = []
    sentences: list[str] = []
    for t, text in enumerate(texts):
        for s, e in split_sentences(text):
            if len(text[s:e].split()) >= min_words:
                jobs.append((t, s, e))
                sentences.append(text[s:e])
    new = list(paraphraser(sentences)) if sentences else []
    if len(new) != len(sentences):
        raise ValueError(f"paraphraser returned {len(new)} outputs for {len(sentences)} inputs")
    edits: list[list[tuple[int, int, str]]] = [[] for _ in texts]
    for (t, s, e), p in zip(jobs, new):
        if p and p.strip():
            edits[t].append((s, e, " ".join(p.split())))
    return [_replace_spans(text, ed) for text, ed in zip(texts, edits)]


# --------------------------------------------------------------------------- #
# LLM attacks
# --------------------------------------------------------------------------- #
PARAPHRASE_PROMPT = (
    "Paraphrase the text below. Keep its meaning, its format and roughly its length (about "
    "{n_words} words), but reword it throughout.\n\n{output}\n\nText:\n\"\"\"\n{text}\n\"\"\""
)
HUMANIZER_PROMPT = (
    "Rewrite the text below so that it reads as if a person wrote it, not an AI. Vary the "
    "sentence length (mix short and long sentences), use contractions, allow an informal "
    "aside or two, and drop stock phrases and tidy summaries. Keep the meaning and roughly "
    "the length (about {n_words} words).\n\n{output}\n\nText:\n\"\"\"\n{text}\n\"\"\""
)
CONTINUE_PROMPT = (
    "Here is the opening of a {domain} text:\n\n\"\"\"\n{opening}\n\"\"\"\n\nContinue the text "
    "from where it stops, in the same style, for about {n_words} more words. Reply with the "
    "continuation only, without repeating the opening."
)
OPENING_WORDS = 40

_PREAMBLE = re.compile(
    r"^(?:(?:sure|certainly|of course|absolutely)\b[^\n]*|"
    r"(?:here(?:'|’)?s|here is|below is)\b[^\n]*:)\s*\n", re.IGNORECASE)
_REFUSAL = re.compile(r"^(?:i(?:'|’)m sorry|i can(?:'|’)?no?t|i cannot|as an ai\b)",
                      re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_llm_output(raw: str | None, min_words: int = MIN_LLM_WORDS) -> str | None:
    """Strip a leading "Sure, here is..." line and quote fences; None for refusals/too short."""
    if not raw:
        return None
    text = _THINK.sub("", raw).replace("\r\n", "\n").strip()
    text = _PREAMBLE.sub("", text, count=1).strip()
    text = text.strip('"').strip()
    if not text or _REFUSAL.match(text) or len(text.split()) < min_words:
        return None
    return text


def fresh_prompt(orig_id: str) -> str | None:
    """The original prompt of a "fresh:<generator>:<prompt-id>:<idx>" text, if it parses."""
    from skolfie.data.generation.prompts import _STYLES_BY_ID, _TEMPLATES_BY_ID, build_prompt

    parts = str(orig_id).rsplit(":", 2)
    if len(parts) != 3 or not parts[0].startswith("fresh:"):
        return None
    m = re.match(r"^([^.]+)\.([^.]*)\.([^.]+)\.w(\d+)\.t(\d+)$", parts[1])
    if not m or m.group(1) not in _TEMPLATES_BY_ID or m.group(3) not in _STYLES_BY_ID:
        return None
    template = _TEMPLATES_BY_ID[m.group(1)]
    codes = [c for c in m.group(2).split("-") if c]
    names = list(template["slots"])
    if len(codes) != len(names):
        return None
    slots = {}
    for name, code in zip(names, codes):
        if code[:-2] != name[:3] or not code[-2:].isdigit():
            return None
        slots[name] = int(code[-2:])
    try:
        return build_prompt(m.group(1), slots, m.group(3), int(m.group(4)),
                            int(m.group(5)) / 10).text
    except (IndexError, KeyError):
        return None


def regeneration_prompt(orig_id: str, text: str, domain: str) -> tuple[str, str, str]:
    """(prompt, kind, prefix): the original prompt of fresh texts ("original", no prefix),
    else a request to continue the text's opening sentences ("continuation"; the opening
    is the prefix of the attacked text)."""
    prompt = fresh_prompt(orig_id)
    if prompt is not None:
        return prompt, "original", ""
    spans = split_sentences(text)
    target = min(OPENING_WORDS, len(text.split()) / 3)
    end, n = 0, 0
    for k, (s, e) in enumerate(spans):  # at least one sentence, never all of several
        if k and (n >= target or k == len(spans) - 1):
            break
        end = e
        n += len(text[s:e].split())
    opening = text[:end]
    remaining = max(MIN_LLM_WORDS, len(text.split()) - n)
    return (CONTINUE_PROMPT.format(domain=domain, opening=opening, n_words=remaining),
            "continuation", opening)


def run_prompts(provider: Any, prompts: Sequence[str], *, temperature: float,
                max_tokens: Sequence[int], workers: int = 8,
                log: Callable[[str], Any] = print) -> list[str | None]:
    """Call the provider for every prompt (threaded); None for failed or refused outputs.

    Auth and unknown-model errors (HTTP 401/403/404) abort the whole run.
    """
    from skolfie.data.generation.providers import FATAL_STATUSES, ProviderError

    def call(i: int) -> str | None:
        try:
            raw = provider.generate(prompts[i], temperature=temperature,
                                    max_tokens=int(max_tokens[i]))
        except ProviderError as exc:
            if exc.status in FATAL_STATUSES:
                raise
            log(f"[llm] request {i} failed: {exc}")
            return None
        return raw

    out: list[str | None] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for raw in pool.map(call, range(len(prompts))):  # in order; a fatal error re-raises
            out.append(raw)
            if len(out) % 100 == 0 or len(out) == len(prompts):
                log(f"[llm] {len(out)}/{len(prompts)} done")
    return out


def temperature_problem(provider: Any, temperature: float) -> str | None:
    """Why `provider` cannot sample at `temperature`, or None if it can."""
    if not getattr(provider, "send_temperature", True):
        return (f"{getattr(provider, 'model_name', 'the model')} does not accept a temperature "
                "(the provider drops it), so a temperature attack would be meaningless")
    if getattr(provider, "provider_name", "") == "anthropic" and temperature > 1:
        return "the Anthropic API accepts temperatures up to 1 only"
    return None


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
@dataclass
class AttackContext:
    """What the attacks may need; leave out what the chosen attacks do not use."""

    seed: int = 0
    provider: Any = None                                       # LLM attacks
    paraphraser: Callable[[list[str]], list[str]] | None = None  # t5_paraphrase
    synonyms: Callable[[str], Sequence[str]] | None = None       # synonym_*
    donors: DonorPool | None = None                              # human_edit_*
    workers: int = 8
    log: Callable[[str], Any] = print
    # Per attack, per text: e.g. {"prompt": "continuation"}; filled by LLM attacks.
    meta: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _llm_max_tokens(texts: Sequence[str]) -> list[int]:
    from skolfie.data.generation.prompts import max_tokens_for

    return [max_tokens_for(max(50, len(t.split()))) for t in texts]


def apply_attack(name: str, frame: Any, ctx: AttackContext) -> list[str | None]:
    """Attack every row of `frame` (columns id, text, source, domain); None = not attacked."""
    spec = get_attack(name)
    texts = [str(t) for t in frame["text"]]
    rngs = [text_rng(ctx.seed, name, t) for t in texts]

    if spec.family == "typo":
        return [typo_text(t, spec.rate, r) for t, r in zip(texts, rngs)]
    if spec.family == "synonym":
        syn = ctx.synonyms or TableSynonyms()
        stop = _stop_words()
        return [synonym_text(t, spec.rate, r, syn, stop) for t, r in zip(texts, rngs)]
    if spec.family == "homoglyph":
        return [homoglyph_text(t, spec.rate, r) for t, r in zip(texts, rngs)]
    if spec.family == "zero_width":
        return [zero_width_text(t, spec.rate, r) for t, r in zip(texts, rngs)]
    if spec.family == "human_edit":
        if ctx.donors is None:
            raise ValueError(f"{name} needs a donor pool of human texts (ctx.donors)")
        return [human_edit_text(t, s, d, spec.rate, r, ctx.donors)
                for t, s, d, r in zip(texts, frame["source"], frame["domain"], rngs)]
    if name == "t5_paraphrase":
        if ctx.paraphraser is None:
            ctx.paraphraser = T5Paraphraser(log=ctx.log)
        return list(paraphrase_texts(texts, ctx.paraphraser))

    if ctx.provider is None:
        raise ValueError(f"{name} needs an LLM provider (ctx.provider)")
    from skolfie.data.generation.prompts import OUTPUT_INSTRUCTION

    if spec.family == "paraphrase":
        template = PARAPHRASE_PROMPT if name == "llm_paraphrase" else HUMANIZER_PROMPT
        prompts = [template.format(n_words=len(t.split()), text=t, output=OUTPUT_INSTRUCTION)
                   for t in texts]
        raw = run_prompts(ctx.provider, prompts, temperature=LLM_TEMPERATURE,
                          max_tokens=_llm_max_tokens(texts), workers=ctx.workers, log=ctx.log)
        ctx.meta[name] = [{"model": ctx.provider.model_name} for _ in texts]
        return [clean_llm_output(r) for r in raw]

    temperature = spec.params["temperature"]
    problem = temperature_problem(ctx.provider, temperature)
    if problem:
        raise ValueError(f"{name}: {problem}")
    built = [regeneration_prompt(i, t, d) for i, t, d in zip(frame["id"], texts, frame["domain"])]
    raw = run_prompts(ctx.provider, [p for p, _, _ in built], temperature=temperature,
                      max_tokens=_llm_max_tokens(texts), workers=ctx.workers, log=ctx.log)
    problem = temperature_problem(ctx.provider, temperature)
    if problem:  # the provider learnt mid-run that the model rejects temperatures
        raise ValueError(f"{name}: {problem}")
    ctx.meta[name] = [{"model": ctx.provider.model_name, "temperature": temperature,
                       "prompt": kind} for _, kind, _ in built]
    out = []
    for (_, _, prefix), r in zip(built, raw):
        text = clean_llm_output(r)
        out.append(None if text is None else (f"{prefix} {text}" if prefix else text))
    return out
