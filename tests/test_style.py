"""Unit tests for skolfie.features.style (compact StyloAI-style features)."""

import numpy as np
import pytest

from skolfie.features.style import (
    FEATURE_DESCRIPTIONS,
    STYLE_FEATURES,
    style_features,
    style_matrix,
)

IDX = {name: i for i, name in enumerate(STYLE_FEATURES)}

EDGE_CASES = [
    "",
    "   \n\t  \n\n ",
    "...",
    "!!! ??? ;;; --- ()",
    "word",
    "Word.",
    "1234567890 42 3.14",
    "😀🎉🔥 👍",
    "Это тест. Ceci n'est pas une pipe. 这是一个测试。",
    "ALL CAPS SHOUTING HERE!!!",
    "a",
    "'''\"\"\"",
    "-- - — –",
    "\n\nHello\n\nworld\n\n",
    "word " * 5000,
]

# 12 sentences, about 150 words; used for the length-invariance test.
BASE_TEXT = (
    "The history of the city is long and complicated. When the first settlers arrived, "
    "they found a river valley that was already home to several communities. I remember "
    "my grandmother telling me stories about the old market; she said it smelled of bread, "
    "fish and diesel. Honestly, I don't know how much of that was true! However, it is "
    "important to note that records from the period are incomplete. Some historians argue "
    "that the population doubled between 1850 and 1880 -- others think the growth was "
    "slower. Why does this matter? Because the way we tell the story shapes how we plan "
    "for the future (and who gets a say in it). Maybe that sounds dramatic. My teacher "
    "always said history is an argument, not a list of dates. Moreover, every family "
    "remembers a different version of events. We're still arguing about it today, "
    "and perhaps we always will be."
)


def feats(text: str) -> dict[str, float]:
    return dict(zip(STYLE_FEATURES, style_features(text)))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_names_unique_and_prefixed():
    assert len(STYLE_FEATURES) == len(set(STYLE_FEATURES))
    assert all(name.startswith("sty_") for name in STYLE_FEATURES)
    assert 35 <= len(STYLE_FEATURES) <= 42


def test_descriptions_for_every_feature():
    assert set(FEATURE_DESCRIPTIONS) == set(STYLE_FEATURES)
    for name, desc in FEATURE_DESCRIPTIONS.items():
        assert isinstance(desc, str) and desc.strip(), name
        assert desc.rstrip().endswith("."), name


def test_no_raw_length_features():
    banned = ("count", "n_words", "n_sent", "n_chars", "newline", "paragraph")
    for name in STYLE_FEATURES:
        assert not any(b in name for b in banned), name


def test_shape_and_dtype():
    vec = style_features(BASE_TEXT)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(STYLE_FEATURES),)
    assert vec.dtype == np.float64


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", EDGE_CASES)
def test_edge_cases_finite(text):
    vec = style_features(text)
    assert vec.shape == (len(STYLE_FEATURES),)
    assert np.isfinite(vec).all()


@pytest.mark.parametrize("text", ["", "   \n ", "...", "😀🎉", "123 456"])
def test_no_words_gives_zero_word_features(text):
    f = feats(text)
    for name in ("sty_word_len_mean", "sty_mattr50", "sty_commas", "sty_sent_len_mean",
                 "sty_flesch", "sty_function_word_share", "sty_bigram_repeat"):
        assert f[name] == 0.0, name


def test_none_is_treated_as_empty():
    assert np.array_equal(style_features(None), style_features(""))


def test_seed_texts_finite(seed_texts):
    for texts in seed_texts.values():
        mat = style_matrix(texts)
        assert np.isfinite(mat).all()


def test_shares_are_bounded(seed_texts):
    shares = [n for n in STYLE_FEATURES if n.endswith("_share")] + [
        "sty_mattr50", "sty_hapax50", "sty_bigram_repeat", "sty_trigram_repeat",
        "sty_sent_start_diversity",
    ]
    mat = style_matrix(seed_texts["human"] + seed_texts["ai"] + EDGE_CASES)
    for name in shares:
        col = mat[:, IDX[name]]
        assert (col >= 0).all() and (col <= 1).all(), name


def test_flesch_is_clipped():
    f = feats("Antidisestablishmentarianism incomprehensibilities " * 30)
    assert -50.0 <= f["sty_flesch"] <= 120.0
    assert f["sty_flesch"] == -50.0


def test_deterministic():
    assert np.array_equal(style_features(BASE_TEXT), style_features(BASE_TEXT))


# ---------------------------------------------------------------------------
# Known values on hand-built texts
# ---------------------------------------------------------------------------
def test_simple_sentence_values():
    f = feats("The cat sat on mats.")
    assert f["sty_word_len_mean"] == pytest.approx(15 / 5)
    assert f["sty_long_word_share"] == 0.0
    assert f["sty_mattr50"] == pytest.approx(1.0)  # shorter than window: plain TTR
    assert f["sty_hapax50"] == pytest.approx(1.0)
    assert f["sty_yule_k"] == pytest.approx(0.0)
    assert f["sty_determiners"] == pytest.approx(20.0)  # "the"
    assert f["sty_prepositions"] == pytest.approx(20.0)  # "on"
    assert f["sty_function_word_share"] == pytest.approx(2 / 5)
    assert f["sty_sent_len_mean"] == pytest.approx(5.0)
    assert f["sty_short_sent_share"] == pytest.approx(1.0)
    assert f["sty_sent_cap_share"] == pytest.approx(1.0)
    assert f["sty_commas"] == 0.0


def test_commas_per_100_words():
    # 10 words, 3 commas -> 30 per 100 words.
    f = feats("Red, green, blue, and also yellow were the colours chosen.")
    assert f["sty_commas"] == pytest.approx(30.0)


def test_punctuation_rates():
    f = feats("Wait... what?! Really; yes: no (maybe) — fine - ok -- “sure” 'quote'")
    # 10 words
    assert f["sty_ellipses"] == pytest.approx(100 / 10)
    assert f["sty_questions"] == pytest.approx(100 / 10)
    assert f["sty_exclamations"] == pytest.approx(100 / 10)
    assert f["sty_semicolons"] == pytest.approx(100 / 10)
    assert f["sty_colons"] == pytest.approx(100 / 10)
    assert f["sty_parentheses"] == pytest.approx(200 / 10)
    assert f["sty_dashes"] == pytest.approx(300 / 10)
    assert f["sty_quotes"] == pytest.approx(400 / 10)


def test_contractions_and_negations():
    # 9 words: don't, I'm, it's, won't are contractions; don't, won't, not negations.
    f = feats("I don't think I'm wrong, it's not. Won't work.")
    words = 9
    assert f["sty_contractions"] == pytest.approx(4 * 100 / words)
    assert f["sty_negations"] == pytest.approx(3 * 100 / words)
    assert f["sty_pron_1st"] == pytest.approx(2 * 100 / words)
    assert f["sty_pron_3rd"] == pytest.approx(1 * 100 / words)


def test_possessive_is_not_contraction_or_quote():
    f = feats("John's car is red.")
    assert f["sty_contractions"] == 0.0
    assert f["sty_quotes"] == 0.0


def test_curly_apostrophe_contraction():
    assert feats("I don’t know.")["sty_contractions"] == pytest.approx(100 / 3)


def test_discourse_markers_including_phrases():
    text = ("However, it is important to note that the results vary. Moreover, in "
            "conclusion, the method works.")
    # 16 words; markers: however, it is important to note, moreover, in conclusion.
    assert feats(text)["sty_discourse_markers"] == pytest.approx(4 * 100 / 16)


def test_hedges_and_informal():
    f = feats("Maybe it is likely. Lol that is really gonna work.")
    # 10 words; hedges: maybe, likely; informal: lol, really, gonna.
    assert f["sty_hedges"] == pytest.approx(20.0)
    assert f["sty_informal"] == pytest.approx(30.0)


def test_pronoun_classes():
    f = feats("You and your friends met them and her.")
    # 8 words
    assert f["sty_pron_2nd"] == pytest.approx(200 / 8)
    assert f["sty_pron_3rd"] == pytest.approx(200 / 8)
    assert f["sty_coord_conj"] == pytest.approx(200 / 8)


def test_equal_sentences_have_zero_cv():
    f = feats("The dog ran home. A cat sat down. We all went out.")
    assert f["sty_sent_len_cv"] == pytest.approx(0.0)
    assert f["sty_sent_len_mean"] == pytest.approx(4.0)


def test_unequal_sentences_have_positive_cv():
    f = feats("Stop. " + "This sentence is much longer than the first one ever was. ")
    assert f["sty_sent_len_cv"] > 0.5


def test_long_and_short_sentence_shares():
    long_sent = " ".join(["word"] * 35) + "."
    f = feats(f"Short one. {long_sent} Another short one.")
    assert f["sty_short_sent_share"] == pytest.approx(2 / 3)
    assert f["sty_long_sent_share"] == pytest.approx(1 / 3)


def test_sentence_start_diversity_and_capitals():
    f = feats("The sun rose. The day began. the birds sang. A dog barked.")
    assert f["sty_sent_start_diversity"] == pytest.approx(2 / 4)  # "the", "a"
    assert f["sty_sent_cap_share"] == pytest.approx(3 / 4)


def test_newlines_split_sentences():
    f = feats("A heading without a full stop\nThe body text follows here.")
    assert f["sty_sent_len_mean"] == pytest.approx(5.5)


def test_character_features():
    f = feats("NASA and the FBI met in 2024. Café.")
    # Non-space chars: NASA(4) and(3) the(3) FBI(3) met(3) in(2) 2024.(5) Café.(5) = 28
    assert f["sty_digit_share"] == pytest.approx(4 / 28)
    assert f["sty_non_ascii_share"] == pytest.approx(1 / 28)
    # Letters: 4+3+3+3+3+2+4 = 22; uppercase: NASA(4) FBI(3) C(1) = 8
    assert f["sty_upper_share"] == pytest.approx(8 / 22)
    assert f["sty_allcaps_words"] == pytest.approx(2 * 100 / 7)


def test_repetition_features():
    f = feats("the cat sat the cat sat")
    # bigrams: the-cat x2, cat-sat x2, sat-the x1 -> 4/5 repeated
    assert f["sty_bigram_repeat"] == pytest.approx(4 / 5)
    # trigrams: the-cat-sat x2, cat-sat-the, sat-the-cat -> 2/4
    assert f["sty_trigram_repeat"] == pytest.approx(2 / 4)
    assert feats("one two three four")["sty_bigram_repeat"] == 0.0


def test_mattr_window_on_long_text():
    words = [f"w{i}" for i in range(200)]
    text = " ".join("x" + w[1:].translate(str.maketrans("0123456789", "abcdefghij"))
                    for w in words)
    f = feats(text)
    assert f["sty_mattr50"] == pytest.approx(1.0)
    assert f["sty_hapax50"] == pytest.approx(1.0)
    assert feats("same " * 200)["sty_mattr50"] == pytest.approx(1 / 50)


def test_length_invariance_of_rates():
    base = feats(BASE_TEXT)
    long = feats(" ".join([BASE_TEXT] * 4))
    # Yule's K depends on the exact frequency spectrum, which duplication changes.
    skip = {"sty_yule_k"}
    for name in STYLE_FEATURES:
        if name in skip:
            continue
        assert long[name] == pytest.approx(base[name], rel=0.05, abs=0.02), name


# ---------------------------------------------------------------------------
# Matrix API
# ---------------------------------------------------------------------------
def test_style_matrix_matches_stacked_features(seed_texts):
    texts = seed_texts["human"][:5] + seed_texts["ai"][:5] + EDGE_CASES[:5]
    mat = style_matrix(texts)
    assert mat.shape == (len(texts), len(STYLE_FEATURES))
    assert mat.dtype == np.float32
    expected = np.vstack([style_features(t) for t in texts]).astype(np.float32)
    assert np.array_equal(mat, expected)


def test_style_matrix_parallel_matches_serial(seed_texts):
    texts = (seed_texts["human"] + seed_texts["ai"] + EDGE_CASES) * 2
    serial = style_matrix(texts, n_jobs=1)
    parallel = style_matrix(texts, n_jobs=2, chunk_size=7)
    assert np.array_equal(serial, parallel)


def test_style_matrix_empty():
    mat = style_matrix([])
    assert mat.shape == (0, len(STYLE_FEATURES))
    assert mat.dtype == np.float32
    assert style_matrix([], n_jobs=2).shape == (0, len(STYLE_FEATURES))


def test_style_matrix_accepts_tuple_and_generator():
    texts = ("One text here.", "Another one!")
    assert np.array_equal(style_matrix(texts), style_matrix(t for t in texts))
