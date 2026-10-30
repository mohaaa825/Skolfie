"""
skolfie/explain/reasons.py
---------------------------
Plain-language reasons for the feature model's (Model B's) features, for the
explanations of roadmap step 8.

Every feature in `STYLE_FEATURES`, `LM_FEATURES`, `BINO_FEATURES` and `DELTA_FEATURES` has an
entry with

    group     the psycholinguistic dimension it measures, following the
              framing of Opara (2025): lexical diversity, readability,
              function words, word choice, sentence structure, burstiness,
              predictability, punctuation, characters, repetition, plus the
              Delta word profile
    title     a short name ("vocabulary variety")
    low/high  what a low and a high value means in plain words
    ai_side   which side is usually the AI side ("low" or "high"), as
              expected from the literature

`reason(name, direction, ...)` builds the sentence shown to the user:
"low vocabulary variety, typical of AI text". `direction` is where the
feature pushed the verdict ("ai" or "human", from the sign of its SHAP
value). Whether the value counts as low or high comes from comparing it
with `typical`, the feature's median on reference texts, when given; a
tree model can learn the unexpected side, and the phrase then still
describes the text truthfully. Without `typical` the side is inferred from
`direction` and `ai_side`, i.e. the expected mapping is assumed. Unknown
feature names get a generic phrase built from the name.

"typical of AI text" is a claim about AI texts in general, so with the
human and AI medians known it is only made when the value sits on the
`direction` side of them (`matches_typical`). Otherwise the tree model
learned something the medians do not show (an interaction, or a feature
whose medians are equal), and the reason says only what the model did:
"many words used only once; the model counts this toward AI". A count
feature that most texts lack (both medians 0) reads "uses semicolons (most
texts have none)" instead of "many semicolons".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

DIRECTIONS = ("ai", "human")
_SUFFIX = {"ai": "typical of AI text", "human": "typical of human writing"}
_MODEL_SUFFIX = {"ai": "the model counts this toward AI",
                 "human": "the model counts this toward human"}


@dataclass(frozen=True)
class Reason:
    group: str
    title: str
    low: str
    high: str
    ai_side: Optional[str]  # "low", "high" or None (no expected side)


def _r(group: str, title: str, low: str, high: str, ai_side: Optional[str]) -> Reason:
    return Reason(group, title, low, high, ai_side)


REASONS: dict[str, Reason] = {
    # ---- style: lexical -------------------------------------------------- #
    "sty_word_len_mean": _r("word choice", "average word length",
                            "short, everyday words", "long words on average", "high"),
    "sty_word_len_std": _r("word choice", "spread of word lengths",
                           "words of similar length", "a wide mix of short and long words",
                           "low"),
    "sty_long_word_share": _r("word choice", "share of long words",
                              "few long words", "many long words (7+ letters)", "high"),
    "sty_mattr50": _r("lexical diversity", "vocabulary variety",
                      "low vocabulary variety", "varied vocabulary", "low"),
    "sty_hapax50": _r("lexical diversity", "words used only once",
                      "few one-off words, the same words come back",
                      "many words used only once", "low"),
    "sty_yule_k": _r("lexical diversity", "vocabulary repetitiveness (Yule's K)",
                     "little repetition of the same words", "repetitive vocabulary", "high"),
    "sty_flesch": _r("readability", "reading ease (Flesch)",
                     "dense, hard-to-read prose", "plain, easy-to-read prose", "low"),
    # ---- style: word classes --------------------------------------------- #
    "sty_function_word_share": _r("function words", "share of function words",
                                  "few function words (the, of, it...)",
                                  "many function words (the, of, it...)", "low"),
    "sty_pron_1st": _r("function words", "first-person pronouns",
                       "little use of I / we", "frequent I / we", "low"),
    "sty_pron_2nd": _r("function words", "second-person pronouns",
                       "rarely addresses the reader as you", "often addresses the reader as you",
                       "low"),
    "sty_pron_3rd": _r("function words", "third-person pronouns",
                       "few he / she / they / it", "many he / she / they / it", "low"),
    "sty_determiners": _r("function words", "articles and determiners",
                          "few articles and determiners (the, a, this)",
                          "many articles and determiners (the, a, this)", "high"),
    "sty_coord_conj": _r("function words", "joining words",
                         "few joining words (and, but, or)", "many joining words (and, but, or)",
                         "high"),
    "sty_prepositions": _r("function words", "prepositions",
                           "few prepositions (of, in, with)", "many prepositions (of, in, with)",
                           "high"),
    "sty_aux_modals": _r("function words", "auxiliary and modal verbs",
                         "few auxiliary and modal verbs (is, can, should)",
                         "many auxiliary and modal verbs (is, can, should)", "low"),
    "sty_negations": _r("function words", "negations",
                        "few negations (not, never)", "many negations (not, never)", "low"),
    "sty_hedges": _r("word choice", "hedging words",
                     "little hedging", "much hedging (perhaps, likely, typically)", "high"),
    "sty_discourse_markers": _r("word choice", "formal linking phrases",
                                "few formal linking phrases",
                                "many formal linking phrases (however, moreover, overall)",
                                "high"),
    "sty_informal": _r("word choice", "informal words and intensifiers",
                       "few informal words or intensifiers",
                       "informal words and intensifiers (really, lol, gonna)", "low"),
    "sty_contractions": _r("word choice", "contractions",
                           "few contractions (don't, it's)", "many contractions (don't, it's)",
                           "low"),
    # ---- style: sentences ------------------------------------------------ #
    "sty_sent_len_mean": _r("sentence structure", "average sentence length",
                            "short sentences", "long sentences", "high"),
    "sty_sent_len_cv": _r("burstiness", "variation in sentence length",
                          "uniform sentence lengths", "sentence lengths that vary a lot",
                          "low"),
    "sty_short_sent_share": _r("burstiness", "share of short sentences",
                               "few short sentences", "many short sentences (under 8 words)",
                               "low"),
    "sty_long_sent_share": _r("burstiness", "share of very long sentences",
                              "few very long sentences",
                              "many very long sentences (over 30 words)", "low"),
    "sty_sent_start_diversity": _r("sentence structure", "variety of sentence openings",
                                   "sentences that often open the same way",
                                   "varied sentence openings", "low"),
    "sty_sent_cap_share": _r("sentence structure", "capitalised sentence starts",
                             "sentences that often start in lower case",
                             "consistently capitalised sentences", "high"),
    # ---- style: punctuation ---------------------------------------------- #
    "sty_commas": _r("punctuation", "commas", "few commas", "many commas", "high"),
    "sty_semicolons": _r("punctuation", "semicolons", "few semicolons", "many semicolons", "low"),
    "sty_colons": _r("punctuation", "colons", "few colons", "many colons", "high"),
    "sty_exclamations": _r("punctuation", "exclamation marks",
                           "few exclamation marks", "many exclamation marks", "low"),
    "sty_questions": _r("punctuation", "question marks",
                        "few questions", "many question marks", "low"),
    "sty_quotes": _r("punctuation", "quotation marks",
                     "few quotation marks", "many quotation marks", "low"),
    "sty_parentheses": _r("punctuation", "parentheses", "few brackets", "many brackets", "low"),
    "sty_dashes": _r("punctuation", "dashes", "few dashes", "many dashes", "low"),
    "sty_ellipses": _r("punctuation", "ellipses", "few or no ellipses (...)",
                       "many ellipses (...)", "low"),
    # ---- style: characters ----------------------------------------------- #
    "sty_digit_share": _r("characters", "digits", "few digits", "many digits", "low"),
    "sty_upper_share": _r("characters", "capital letters",
                          "few capital letters", "many capital letters", "low"),
    "sty_allcaps_words": _r("characters", "ALL-CAPS words",
                            "few ALL-CAPS words", "many ALL-CAPS words", "low"),
    "sty_non_ascii_share": _r("characters", "special characters",
                              "plain characters only", "many special or accented characters",
                              "low"),
    # ---- style: repetition ----------------------------------------------- #
    "sty_bigram_repeat": _r("repetition", "repeated word pairs",
                            "few repeated word pairs", "many repeated word pairs", "high"),
    "sty_trigram_repeat": _r("repetition", "repeated three-word phrases",
                             "few repeated phrases", "many repeated three-word phrases", "high"),
    # ---- GPT-2 language-model features ----------------------------------- #
    "lm_mean_nll": _r("predictability", "perplexity",
                      "highly predictable wording (low perplexity)",
                      "surprising word choices (high perplexity)", "low"),
    "lm_std_nll": _r("burstiness", "word-to-word variation in predictability",
                     "evenly predictable from word to word",
                     "predictability that jumps from word to word", "low"),
    "lm_p90_nll": _r("predictability", "most surprising words",
                     "hardly any surprising words", "some very surprising words", "low"),
    "lm_sent_mean_nll_std": _r("burstiness", "sentence-to-sentence variation in predictability",
                               "every sentence about equally predictable",
                               "predictability that varies from sentence to sentence", "low"),
    "lm_sent_nll_cv": _r("burstiness", "relative variation in predictability",
                         "evenly predictable sentences",
                         "sentences that differ a lot in predictability", "low"),
    "lm_mean_log_rank": _r("predictability", "rank of the actual words",
                           "words that match the language model's top guesses",
                           "words that often differ from the language model's top guesses",
                           "low"),
    "lm_top1_frac": _r("predictability", "words the model guessed first",
                       "few words are the language model's first guess",
                       "many words are the language model's first guess", "high"),
    "lm_top10_frac": _r("predictability", "words in the model's top 10",
                        "many words outside the language model's top-10 guesses",
                        "most words are among the language model's top-10 guesses", "high"),
    "lm_mean_entropy": _r("predictability", "openness of the next word",
                          "context that makes the next word easy to guess",
                          "context that leaves the next word open", "low"),
    "lm_nll_minus_entropy": _r("predictability", "surprise versus expected surprise",
                               "words less surprising than the language model expects",
                               "words more surprising than the language model expects", "low"),
    # ---- Binoculars (modern observer / performer LM pair) ----------------- #
    "bino_obs_log_ppl": _r("predictability", "perplexity under a modern language model",
                           "highly predictable to a modern language model",
                           "surprising to a modern language model", "low"),
    "bino_x_ppl": _r("predictability", "how open the context leaves the next word "
                     "(cross-perplexity)",
                     "context that makes any continuation predictable",
                     "context that leaves many continuations open", "low"),
    "bino_score": _r("predictability", "Binoculars score",
                     "wording more predictable than its context explains (low Binoculars score)",
                     "wording less predictable than its context explains "
                     "(high Binoculars score)", "low"),
    "bino_perf_log_ppl": _r("predictability", "perplexity under a chat model",
                            "highly predictable to a chat-tuned language model",
                            "surprising to a chat-tuned language model", "low"),
    "bino_obs_nll_std": _r("burstiness", "word-to-word variation in predictability "
                           "(modern language model)",
                           "evenly predictable from word to word",
                           "predictability that jumps from word to word", "low"),
    # ---- Delta word profile ---------------------------------------------- #
    "delta_cos_human": _r("word profile", "distance to human common-word use",
                          "common words used much like human writing",
                          "common words used unlike human writing", "high"),
    "delta_cos_ai": _r("word profile", "distance to AI common-word use",
                       "common words used much like AI text",
                       "common words used unlike AI text", "low"),
    "delta_cos_diff": _r("word profile", "common-word use, AI versus human (Cosine Delta)",
                         "common words used more like human writing",
                         "common words used more like AI text", "high"),
    "delta_bur_diff": _r("word profile", "common-word use, AI versus human (Burrows' Delta)",
                         "common words used more like human writing",
                         "common words used more like AI text", "high"),
}


def _generic(name: str) -> Reason:
    title = name.split("_", 1)[-1].replace("_", " ") if "_" in name else name
    return Reason("other", title, f"low {title}", f"high {title}", None)


def lookup(name: str) -> Reason:
    """The entry for a feature name; a generic one built from the name if unknown."""
    return REASONS.get(name) or _generic(name)


def _missing(value: Optional[float]) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def value_side(name: str, direction: str, value: Optional[float] = None,
               typical: Optional[float] = None) -> Optional[str]:
    """"low" or "high": against `typical` when both are known, else the expected side."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if not _missing(value) and not _missing(typical) and value != typical:
        return "high" if value > typical else "low"
    ai_side = lookup(name).ai_side
    if ai_side is None:
        return None
    if direction == "ai":
        return ai_side
    return "low" if ai_side == "high" else "high"


def matches_typical(direction: str, value: Optional[float], typical_human: Optional[float],
                    typical_ai: Optional[float]) -> Optional[bool]:
    """
    Whether `value` sits on the `direction` side of the human and AI medians
    (closer to that class's median); None when that cannot be told (a value or
    median is missing, or the two medians are equal).
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if _missing(value) or _missing(typical_human) or _missing(typical_ai):
        return None
    if typical_human == typical_ai:
        return None
    closer = "human" if abs(value - typical_human) < abs(value - typical_ai) else "ai"
    return closer == direction


def reason(name: str, direction: str, value: Optional[float] = None,
           typical: Optional[float] = None, typical_human: Optional[float] = None,
           typical_ai: Optional[float] = None) -> str:
    """One plain-language sentence, e.g. "low vocabulary variety, typical of AI text"."""
    entry = lookup(name)
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if value is not None and _missing(value):
        # LightGBM routes missing values down a learned branch, so they can still count.
        return f"{entry.title} could not be measured (text too short), {_SUFFIX[direction]}"
    has_medians = not (_missing(typical_human) or _missing(typical_ai))
    if has_medians and typical_human == typical_ai == 0 and not _missing(value) and value > 0:
        phrase = f"uses {entry.title} (most texts have none)"
    else:
        side = value_side(name, direction, value, typical)
        phrase = {"low": entry.low, "high": entry.high}.get(side or "", entry.title)
    if has_medians and not matches_typical(direction, value, typical_human, typical_ai):
        return f"{phrase}; {_MODEL_SUFFIX[direction]}"
    return f"{phrase}, {_SUFFIX[direction]}"
