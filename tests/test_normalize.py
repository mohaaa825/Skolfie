"""
tests/test_normalize.py
------------------------
Roadmap step 9, the defence (skolfie/robust/normalize.py): the Unicode
attacks are undone, real Cyrillic / Greek / emoji text is left alone, the
obfuscation counts, and the models score an attacked text like the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from skolfie.robust import attacks as A
from skolfie.robust.normalize import (
    CONFUSABLES,
    normalize_for_scoring,
    obfuscation_note,
    obfuscation_stats,
)
from tests.test_evaluate import make_corpus

TEXT = (
    "A committee will consider the important proposal tomorrow morning. However, numerous "
    "members believe it is a crucial step for the whole society! I hope so, Jo. Critics "
    "argue that the plan is too expensive and too slow.\nIn the end, a vote will show "
    "whether the approach works: OK, XML, CPU, i.e. a mix of cases."
)

UNTOUCHED = [
    "Привет, как дела? Я в Москве. А у нас всё хорошо, и сок очень вкусный.",
    "В Linux есть всё, а в Windows нет. Это IT и ПО.",
    "Київ є столицею України, і там живе багато людей.",
    "Ја сам у Београду и свако јутро пијем кафу.",
    "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας. Ο καιρός και η θάλασσα.",
    "ΚΑΙ ΤΟ ΟΡΟΣ ΕΙΝΑΙ ΚΑΛΟ.",
    "We set the α level to 0.05 and report the τ statistic and χ² test.",
    "The rocket reached Mach 3 at −40 °C … then 20 m² in ½ hour™.",
    "Café, naïve, Zürich, São Paulo, Ærøskøbing, Łódź, Dvořák.",
    "Family 👨\u200d👩\u200d👧 and flag 🏳\ufe0f\u200d🌈 emoji.",
    "می\u200cخواهم بروم",  # Persian: ZWNJ between letters
    "क\u094d\u200dष and ক\u09cd\u200cষ",  # Indic: ZWJ / ZWNJ after a virama
    "Ten-year-old 「日本語」 text 中文 and 한국어.",
    "Ordinary ASCII text, with punctuation -- and \"quotes\"...",
]


def _attacked(text: str, rate: float, seed: int) -> str:
    glyph = A.homoglyph_text(text, rate, A.text_rng(seed, "homoglyph", text))
    return A.zero_width_text(glyph, rate, A.text_rng(seed, "zero_width", text))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate", [0.2, 0.5, 1.0])
def test_attacks_round_trip(rate: float) -> None:
    texts = [TEXT, *make_corpus(n_per_cell=2, seed=3)["text"]]
    for seed in range(5):
        for text in texts:
            assert text.isascii()
            for attacked in (A.homoglyph_text(text, rate, A.text_rng(seed, "h", text)),
                             A.zero_width_text(text, rate, A.text_rng(seed, "z", text)),
                             _attacked(text, rate, seed)):
                assert normalize_for_scoring(attacked) == text


def test_confusables_cover_the_attack_map() -> None:
    # The table is wider than the attack's own map, and agrees with it.
    assert all(CONFUSABLES[v] == k for k, v in A.HOMOGLYPHS.items())
    assert len(CONFUSABLES) > 2 * len(A.HOMOGLYPHS)
    assert all(v.isascii() and v.isalpha() and len(v) == 1 for v in CONFUSABLES.values())


@pytest.mark.parametrize("text", UNTOUCHED)
def test_real_text_is_untouched(text: str) -> None:
    assert normalize_for_scoring(text) == text
    assert obfuscation_stats(text) == {"invisible": 0, "lookalikes": 0}


def test_lookalikes_only_in_latin_words() -> None:
    # One Cyrillic letter in a Latin word, a Greek omicron, an IPA script g.
    assert normalize_for_scoring("Тhe tеxt is gοod and biɡ.") == "The text is good and big."
    # A word of Cyrillic look-alikes between Latin words: the article "a".
    assert normalize_for_scoring("This is а test.") == "This is a test."
    assert normalize_for_scoring("а cat sat") == "a cat sat"
    # ... but not between Russian words, nor next to only one Latin word.
    assert normalize_for_scoring("Это а там") == "Это а там"
    assert normalize_for_scoring("В Linux") == "В Linux"
    # A Russian word with a Latin typo keeps its Cyrillic letters.
    assert normalize_for_scoring("мой кoмпьютер") == "мой кoмпьютер"
    # Mostly look-alikes with one Latin letter: only between Latin words.
    assert normalize_for_scoring("we drink сoк daily") == "we drink cok daily"
    assert normalize_for_scoring("мы пьём сoк утром") == "мы пьём сoк утром"
    # Greek letters standing alone are symbols, not disguised Latin.
    assert normalize_for_scoring("the ο and ν cases") == "the ο and ν cases"


def test_invisible_characters_and_joiners() -> None:
    invisible = ["\u00ad", "\u034f", "\u180e", "\u200b", "\u200e", "\u200f", "\u202a",
                 "\u202b", "\u202c", "\u202d", "\u202e", "\u2060", "\u2061", "\u2062",
                 "\u2063", "\u2064", "\ufeff"]
    for ch in invisible:
        assert normalize_for_scoring(f"wo{ch}rd and{ch} more") == "word and more"
    # ZWJ / ZWNJ go inside and next to Latin words, stay between emoji or non-Latin letters.
    assert normalize_for_scoring("wo\u200drd wo\u200crd") == "word word"
    assert normalize_for_scoring("\u200dstart end\u200d") == "start end"
    assert normalize_for_scoring("x 👍\u200d🔥 y") == "x 👍\u200d🔥 y"
    assert obfuscation_stats("x\u200d\u200by\u00ad") == {"invisible": 3, "lookalikes": 0}


def test_compatibility_folding_is_limited_to_letters_and_digits() -> None:
    disguised = "ｆｕｌｌｗｉｄｔｈ １２３ 𝐛𝐨𝐥𝐝 𝘪𝘵𝘢𝘭𝘪𝘤"
    assert normalize_for_scoring(disguised) == "fullwidth 123 bold italic"
    assert normalize_for_scoring("ﬁne ﬂow") == "fine flow"
    # Punctuation, symbols, superscripts, fractions and spaces keep their form.
    kept = "Wait… x² ½ 1st™ a\u00a0b ！ «quote» — dash"
    assert normalize_for_scoring(kept) == kept
    # NFC like the corpus: a decomposed accent is composed.
    assert normalize_for_scoring("cafe\u0301") == "café"
    assert normalize_for_scoring("ｃａｆｅ\u0301") == "café"
    stats = obfuscation_stats("ｆｕｌｌ ﬁne")
    assert stats == {"invisible": 0, "lookalikes": 4}  # the ligature is not a disguise


@pytest.mark.parametrize("text", [
    *UNTOUCHED, _attacked(TEXT, 0.3, 0), _attacked(TEXT, 1.0, 1),
    "е\u0301 tе\u0301ѕt сафе ａ\u0301 \u200d\u200d 👍\u200d", "а о cat dog", "а the о dog",
    "ϰ ϲ Ϲat", "\ufeffBOM text", "",
])
def test_idempotent(text: str) -> None:
    once = normalize_for_scoring(text)
    assert normalize_for_scoring(once) == once
    assert obfuscation_stats(once) == {"invisible": 0, "lookalikes": 0}


def test_non_string_input() -> None:
    for value in (None, float("nan"), 5, b"bytes"):
        assert normalize_for_scoring(value) == ""
        assert obfuscation_stats(value) == {"invisible": 0, "lookalikes": 0}


def test_obfuscation_stats_and_note() -> None:
    assert obfuscation_stats("tеxt\u200b wоrd") == {"invisible": 1, "lookalikes": 2}
    # A leading BOM (from a file's encoding) is removed but not counted.
    assert normalize_for_scoring("\ufeffHello") == "Hello"
    assert obfuscation_stats("\ufeffHello\ufeff") == {"invisible": 1, "lookalikes": 0}
    attacked = _attacked(TEXT, 0.2, 0)
    stats = obfuscation_stats(attacked)
    assert stats["invisible"] == attacked.count(A.ZERO_WIDTH)
    assert stats["lookalikes"] == sum(c in CONFUSABLES for c in attacked) > 0

    assert obfuscation_note({"invisible": 0, "lookalikes": 0}) == obfuscation_note(None) == ""
    note = obfuscation_note({"invisible": 1, "lookalikes": 2})
    assert note.startswith("The text contains 1 invisible character and 2 look-alike letters")
    assert obfuscation_note({"invisible": 1200, "lookalikes": 0}).startswith(
        "The text contains 1,200 invisible characters (")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus():
    corpus = make_corpus(n_per_cell=12, seed=1)
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    attacked = [_attacked(t, 0.2, i) for i, t in enumerate(texts)]
    assert all(a != t for a, t in zip(attacked, texts))
    return texts, labels, attacked


def test_feature_model_scores_attacked_text_like_the_original(corpus, tmp_path) -> None:
    pytest.importorskip("lightgbm")
    from skolfie.models.feature_model import FeatureModel

    texts, labels, attacked = corpus
    kwargs = dict(use_lm=False, delta_words=50, delta_folds=3, n_jobs=1, log=lambda m: None,
                  lgbm_params={"n_estimators": 60})
    fm = FeatureModel(cache_dir=tmp_path, **kwargs).fit(texts, labels)
    n_cached = len(fm._caches["style"])
    p = fm.predict_proba(texts)
    np.testing.assert_allclose(fm.predict_proba(attacked), p)
    np.testing.assert_allclose(fm.featurize(attacked), fm.featurize(texts))
    # The cache is keyed by the normalised text: the attacked copies are hits.
    assert len(fm._caches["style"]) == n_cached
    # Training on attacked texts gives the same model.
    fm_attacked = FeatureModel(**kwargs).fit(attacked, labels)
    np.testing.assert_allclose(fm_attacked.predict_proba(texts), p)


def test_baselines_and_stack_score_attacked_text_like_the_original(corpus, tmp_path) -> None:
    import joblib

    from skolfie.models.baselines import LengthBaseline, TfidfLRBaseline
    from skolfie.models.stacker import StackedModel, Stacker, operating_points

    texts, labels, attacked = corpus
    paths = {}
    for name, model in (("length", LengthBaseline()), ("tfidf", TfidfLRBaseline(n_jobs=1))):
        model.fit(texts, labels)
        np.testing.assert_allclose(model.predict_proba(attacked), model.predict_proba(texts))
        paths[name] = tmp_path / f"{name}.joblib"
        joblib.dump(model, paths[name])
    P = np.column_stack([joblib.load(paths[n]).predict_proba(texts) for n in paths])
    stacker = Stacker().fit(P, labels)
    stacked = StackedModel(paths, stacker, operating_points(labels, stacker.predict_proba(P)))
    np.testing.assert_allclose(stacked.predict_proba(attacked), stacked.predict_proba(texts))


def test_app_shows_the_obfuscation_warning() -> None:
    from skolfie.app.gradio_app import render_result_html
    from skolfie.robust.normalize import obfuscation_note

    result = {"label": "ai", "headline": "Likely AI-generated", "p_ai": 0.9, "warnings": []}
    assert "Hidden characters" not in render_result_html(result, "some text")
    note = obfuscation_note({"invisible": 4, "lookalikes": 0})
    result["warnings"] = [{"kind": "obfuscation", "message": note}]
    page = render_result_html(result, "some text")
    assert "Hidden characters" in page and "The text contains 4 invisible characters" in page
