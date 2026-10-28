"""
tests/test_explain.py
----------------------
Roadmap step 8: plain-language reasons, SHAP values for the feature model,
the Explainer's output, the HTML / Markdown views and scripts/explain.py end
to end on the synthetic evaluation corpus.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pytest

from skolfie.explain.reasons import REASONS, lookup, matches_typical, reason, value_side
from skolfie.features.delta_features import DELTA_FEATURES
from skolfie.features.lm import LM_FEATURES
from skolfie.features.style import STYLE_FEATURES
from skolfie.mixed.chunking import chunk_spans
from skolfie.mixed.detector import ChunkRule
from skolfie.models.baselines import LengthBaseline, TfidfLRBaseline
from tests.test_evaluate import make_corpus, make_splits
from tests.test_stacker import _load_script

ALL_FEATURES = (*STYLE_FEATURES, *LM_FEATURES, *DELTA_FEATURES)
CHUNKER = partial(chunk_spans, target_words=20, min_words=8)


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #
def test_every_feature_has_a_reason() -> None:
    assert set(ALL_FEATURES) == set(REASONS)
    for name in ALL_FEATURES:
        entry = REASONS[name]
        assert entry.ai_side in ("low", "high")
        assert entry.group and entry.title and entry.low and entry.high
        assert entry.low != entry.high


def test_reason_wording() -> None:
    assert reason("sty_mattr50", "ai") == "low vocabulary variety, typical of AI text"
    assert reason("sty_mattr50", "human") == "varied vocabulary, typical of human writing"
    # A known value: the side comes from the value, not the expected mapping.
    assert reason("sty_mattr50", "ai", value=0.9, typical=0.5) == (
        "varied vocabulary, typical of AI text")
    assert value_side("lm_mean_nll", "human", value=1.0, typical=2.0) == "low"
    assert "could not be measured" in reason("lm_mean_nll", "ai", value=float("nan"))
    # Unknown names fall back to a phrase built from the name.
    assert lookup("foo_bar_baz").title == "bar baz"
    assert reason("foo_bar_baz", "ai", value=3.0, typical=1.0) == "high bar baz, typical of AI text"
    assert reason("foo_bar_baz", "human").endswith("typical of human writing")
    with pytest.raises(ValueError):
        reason("sty_mattr50", "maybe")


def test_reason_only_claims_typical_when_the_medians_back_it() -> None:
    # Value on the AI side of the medians: a claim about AI texts in general.
    assert reason("lm_mean_nll", "ai", value=2.4, typical=3.0, typical_human=3.4,
                  typical_ai=2.7) == ("highly predictable wording (low perplexity), "
                                      "typical of AI text")
    # Pushed towards human although the value is on the AI side (seen in the step 8 review):
    # say only what the model did.
    assert reason("sty_hapax50", "human", value=0.58, typical=0.64, typical_human=0.65,
                  typical_ai=0.63) == ("few one-off words, the same words come back; "
                                       "the model counts this toward human")
    # A count most texts lack (both medians 0): "uses", not "many".
    assert reason("sty_semicolons", "human", value=0.1, typical=0.0, typical_human=0.0,
                  typical_ai=0.0) == ("uses semicolons (most texts have none); "
                                      "the model counts this toward human")
    assert reason("sty_semicolons", "ai", value=0.0, typical=0.0, typical_human=0.0,
                  typical_ai=0.0) == "few semicolons; the model counts this toward AI"
    assert matches_typical("ai", 2.4, 3.4, 2.7) is True
    assert matches_typical("human", 2.4, 3.4, 2.7) is False
    assert matches_typical("ai", 0.0, 0.0, 0.0) is None
    assert matches_typical("ai", None, 3.4, 2.7) is None
    with pytest.raises(ValueError):
        matches_typical("maybe", 1.0, 2.0, 3.0)


# --------------------------------------------------------------------------- #
# SHAP on the feature model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained() -> dict:
    pytest.importorskip("lightgbm")
    from skolfie.models.feature_model import FeatureModel

    corpus = make_corpus(n_per_cell=12, seed=1)
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    fm = FeatureModel(use_lm=False, delta_words=50, delta_folds=3, n_jobs=1,
                      log=lambda m: None, lgbm_params={"n_estimators": 60}).fit(texts, labels)
    return {"texts": texts, "labels": labels, "features": fm,
            "length": LengthBaseline().fit(texts, labels)}


def test_shap_values_sum_to_raw_score(trained) -> None:
    from skolfie.explain.contributions import explain_features, shap_values

    fm, texts = trained["features"], trained["texts"]
    X = fm.featurize(texts)
    contrib, base = shap_values(fm, X)
    assert contrib.shape == X.shape and base.shape == (len(texts),)
    raw = fm.clf.predict(X, raw_score=True)
    np.testing.assert_allclose(base + contrib.sum(axis=1), raw, atol=1e-6)

    out = explain_features(fm, X=X, top_k=5)
    np.testing.assert_allclose([o["raw_score"] for o in out], raw, atol=1e-6)
    np.testing.assert_allclose([o["p_ai"] for o in out], fm.predict_proba(texts), atol=1e-6)
    names = fm.feature_names_
    for i, o in enumerate(out):
        rows = o["top_features"]
        assert len(rows) == 5
        mags = [abs(r["contribution"]) for r in rows]
        assert mags == sorted(mags, reverse=True)
        # The top 5 are the 5 largest of all contributions.
        assert mags[-1] >= np.sort(np.abs(contrib[i]))[-5] - 1e-12
        for r in rows:
            j = names.index(r["feature"])
            assert r["contribution"] == pytest.approx(contrib[i, j])
            assert r["value"] == pytest.approx(float(X[i, j]))
            assert r["direction"] == ("ai" if r["contribution"] > 0 else "human")
            assert r["reason"].endswith("AI text" if r["direction"] == "ai"
                                        else "human writing")
            assert r["typical_human"] is None and r["typical_ai"] is None


def test_feature_reference_and_errors(trained) -> None:
    from skolfie.explain.contributions import explain_features, feature_reference, shap_values

    fm, texts, labels = trained["features"], trained["texts"], trained["labels"]
    ref = feature_reference(fm, texts[:40], labels[:40])
    assert set(ref) == set(fm.feature_names_)
    assert set(ref["sty_mattr50"]) == {"median", "human", "ai"}
    row = explain_features(fm, texts[:1], top_k=3, reference=ref)[0]["top_features"][0]
    assert row["typical_human"] == ref[row["feature"]]["human"]

    with pytest.raises(ValueError):
        shap_values(fm, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        explain_features(fm)
    with pytest.raises(ValueError):
        explain_features(fm, texts[:1], top_k=0)
    assert shap_values(fm, np.zeros((0, len(fm.feature_names_))))[0].shape == (
        0, len(fm.feature_names_))
    rf = type(fm)(classifier="rf")
    rf.clf = object()
    with pytest.raises(NotImplementedError):
        shap_values(rf, np.zeros((1, 1)))


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #
def test_explainer_output_shape(trained, tmp_path: Path) -> None:
    from skolfie.explain.explainer import Explainer

    joblib.dump(trained["features"], tmp_path / "features.joblib")
    joblib.dump(trained["length"], tmp_path / "length.joblib")
    explainer = Explainer(tmp_path / "length.joblib", tmp_path / "features.joblib",
                          chunk_rule=ChunkRule(lo=0.45, hi=0.55), chunker=CHUNKER,
                          batch_size=4)
    texts = trained["texts"][:10] + [trained["texts"][0] + " " + trained["texts"][-1], ""]
    out = explainer.explain_many(texts)
    assert len(out) == len(texts)
    json.loads(json.dumps(out, allow_nan=False))  # plain JSON as it is

    p = trained["length"].predict_proba(texts)
    for ex, text, p_ai in zip(out, texts, p):
        assert set(ex) == {"label", "verdict", "p_ai", "chunks", "highlights", "top_features",
                           "feature_model"}
        assert ex["verdict"] == ("ai" if p_ai >= 0.5 else "human")
        assert ex["label"] in (ex["verdict"], "mixed")
        assert ex["p_ai"] == pytest.approx(p_ai)
        assert len(ex["top_features"]) == 5
        assert set(ex["top_features"][0]) == {"feature", "group", "value", "typical_human",
                                              "typical_ai", "contribution", "direction",
                                              "matches_typical", "reason"}
        assert set(ex["feature_model"]) == {"p_ai", "raw_score", "base_value", "agrees"}
        assert ex["feature_model"]["agrees"] == (
            (ex["feature_model"]["p_ai"] >= 0.5) == (ex["verdict"] == "ai"))
        for c in ex["chunks"]:
            assert 0 <= c["start"] <= c["end"] <= len(text)
            assert c["label"] == ("ai" if c["p"] >= 0.5 else "human")
        assert all(h["label"] in ("ai", "human") for h in ex["highlights"])
    # Mixed only where the chunk rule says so.
    scored_mixed = [ex["label"] == "mixed" for ex in out]
    rule_says = [len(ex["chunks"]) >= 2 and max(c["p"] for c in ex["chunks"]) >= 0.55
                 and min(c["p"] for c in ex["chunks"]) <= 0.45 for ex in out]
    assert scored_mixed == rule_says
    assert explainer.explain(texts[0]) == out[0]
    info = explainer.info()
    assert info["chunk_rule"]["lo"] == 0.45 and info["verdict_model"] == "length"


def test_explainer_mixed_text(trained) -> None:
    from skolfie.explain.explainer import Explainer

    texts, labels = trained["texts"], trained["labels"]
    human = max((t for t, y in zip(texts, labels) if y == "human"), key=len)
    ai = max((t for t, y in zip(texts, labels) if y == "ai"), key=len)
    text = human + " " + ai
    explainer = Explainer(trained["length"], trained["features"],
                          chunk_scorer=trained["features"], chunk_rule=ChunkRule(0.3, 0.7),
                          chunker=CHUNKER)
    ex = explainer.explain(text)
    assert ex["label"] == "mixed" and ex["verdict"] in ("human", "ai")
    # The heatmap finds the boundary: human chunks first, then AI chunks.
    assert [h["label"] for h in ex["highlights"]] == ["human", "ai"]
    boundary = ex["highlights"][1]["start"]
    assert boundary == len(human) + 1
    assert text[boundary:] == ai


def test_explainer_finds_features_and_verdicts(trained) -> None:
    from skolfie.explain.explainer import Explainer
    from skolfie.models.stacker import StackedModel, Stacker

    fm, texts = trained["features"], trained["texts"][:6]
    # The feature model explains itself; no chunk rule -> never Mixed.
    out = Explainer(fm, chunker=CHUNKER).explain_many(texts)
    assert all(ex["label"] in ("human", "ai") for ex in out)
    for ex in out:
        assert ex["feature_model"]["p_ai"] == pytest.approx(ex["p_ai"])

    # A stacked model: verdicts from its cut-offs (with an Uncertain band).
    class _Stacked(StackedModel):
        def _member(self, name):
            return type("Loaded", (), {"obj": fm, "predict_proba": fm.predict_proba})()

    cutoffs = {"ai_threshold": 0.9, "human_threshold": 0.1}
    stacker = Stacker().fit(np.array([[0.1], [0.2], [0.8], [0.9]]), [0, 0, 1, 1])
    stacked = _Stacked({"features": "unused.joblib"}, stacker, cutoffs)
    explainer = Explainer(stacked, chunker=CHUNKER)
    assert explainer.feature_model is fm
    out = explainer.explain_many(texts)
    for ex in out:
        expected = ("ai" if ex["p_ai"] > 0.9 else "human" if ex["p_ai"] < 0.1 else "uncertain")
        assert ex["verdict"] == ex["label"] == expected

    with pytest.raises(ValueError):
        Explainer(trained["length"])  # no feature model to explain with
    with pytest.raises(ValueError):
        Explainer(fm, top_k=0)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fake_explanation(text: str) -> dict:
    return {
        "label": "mixed", "verdict": "ai", "p_ai": 0.91,
        "chunks": [{"start": 0, "end": 10, "p": 0.05, "label": "human"},
                   {"start": 11, "end": len(text), "p": 0.97, "label": "ai"}],
        "highlights": [],
        "top_features": [{"feature": "sty_mattr50", "group": "lexical diversity", "value": 0.4,
                          "typical_human": 0.7, "typical_ai": 0.5, "contribution": 1.25,
                          "direction": "ai",
                          "reason": "low vocabulary variety, <b>typical</b> of AI text"},
                         {"feature": "x|y", "group": "other", "value": None,
                          "typical_human": None, "typical_ai": None, "contribution": -0.5,
                          "direction": "human", "reason": "a | b"}],
        "feature_model": {"p_ai": 0.8, "raw_score": 1.4, "base_value": 0.1, "agrees": True},
    }


def test_render_notes_when_the_feature_model_disagrees() -> None:
    from skolfie.explain.explainer import feature_model_agrees
    from skolfie.explain.render import disagreement_note, render_html, render_markdown

    assert feature_model_agrees("ai", 0.37) is False and feature_model_agrees("ai", 0.9)
    assert feature_model_agrees("human", 0.2) and feature_model_agrees("uncertain", 0.9) is None
    text = "Some text that the transformer finds very AI-like."
    ex = _fake_explanation(text)
    assert disagreement_note(ex) == ""
    ex["feature_model"].update(p_ai=0.37, agrees=False)
    note = disagreement_note(ex)
    assert note.startswith("The style features alone point to human (P(ai) 0.37")
    assert note in render_html([ex], [text])
    assert f"> {note}" in render_markdown([ex], [text], cases=[{"id": "a"}])


def test_render_html_escapes() -> None:
    from skolfie.explain.render import render_html

    text = "<script>alert('x')</script> & then some AI text."
    page = render_html([_fake_explanation(text)], [text], captions=["<i>case</i>"])
    assert page.startswith("<!doctype html>")
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<i>case</i>" not in page and "&lt;i&gt;case&lt;/i&gt;" in page
    assert "<b>typical</b>" not in page
    assert "http" not in page  # self-contained: no external assets
    assert page.count('class="chunk"') == 2
    assert "rgba(214, 39, 40" in page and "rgba(31, 119, 180" in page
    with pytest.raises(ValueError):
        render_html([], [text])


def test_render_markdown_review_sheet() -> None:
    from skolfie.explain.render import md_escape, render_markdown

    text = "First | part.\nSecond <part> here."
    md = render_markdown([_fake_explanation(text)], [text],
                         cases=[{"id": "s1:a|b", "true_label": "ai"}])
    assert "reasons plausible? (y/n)" in md and "highlight plausible? (y/n)" in md
    assert "s1:a\\|b" in md and "<part>" not in md and "&lt;b&gt;typical" in md
    checklist = [ln for ln in md.splitlines() if ln.startswith("| 1 | s1:a")]
    assert len(checklist) == 1 and checklist[0].endswith("|  |  |")
    assert md_escape("a|b\nc") == "a\\|b c"
    with pytest.raises(ValueError):
        render_markdown([_fake_explanation(text)], [text], cases=[])


# --------------------------------------------------------------------------- #
# scripts/explain.py
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("lightgbm")
    from skolfie.models.feature_model import FeatureModel

    root = tmp_path_factory.mktemp("explaindata")
    corpus = make_corpus(n_per_cell=24, seed=7)
    splits = make_splits(corpus, seed=7)
    corpus.to_parquet(root / "corpus.parquet", index=False)
    splits.to_parquet(root / "splits.parquet", index=False)
    train = corpus[corpus["id"].isin(splits.loc[splits["split"] == "train", "id"])]
    texts, labels = train["text"].tolist(), train["label"].tolist()
    joblib.dump(TfidfLRBaseline(n_jobs=1).fit(texts, labels), root / "tfidf.joblib")
    joblib.dump(FeatureModel(use_lm=False, delta_words=50, delta_folds=3, n_jobs=1,
                             log=lambda m: None, lgbm_params={"n_estimators": 60})
                .fit(texts, labels), root / "features.joblib")
    (root / "mixed_report.json").write_text(json.dumps({
        "rules": {"chunk": {"params": {"lo": 0.3, "hi": 0.7, "min_chunks": 2,
                                       "min_share": 0.0}}},
        "chunking": {"target_words": 20, "min_words": 8}}))
    return {"root": root, "corpus": root / "corpus.parquet", "splits": root / "splits.parquet"}


def _argv(data: dict[str, Path], *extra: str) -> list[str]:
    root = data["root"]
    return ["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
            "--model", str(root / "tfidf.joblib"),
            "--features-model", str(root / "features.joblib"), *extra]


def test_explain_script_end_to_end(data, capsys) -> None:
    out = data["root"] / "out"
    script = _load_script("explain")
    argv = _argv(data, "--n", "8", "--reference-n", "20", "--batch-size", "3",
                 "--mixed-report", str(data["root"] / "mixed_report.json"),
                 "--out-dir", str(out))
    assert script.main(argv) == 0
    assert "review.md" in capsys.readouterr().out

    result = json.loads((out / "explanations.json").read_text())
    assert result["n"] == 8 and len(result["cases"]) == 8
    assert sorted(c["true_label"] for c in result["cases"]) == ["ai"] * 4 + ["human"] * 4
    assert result["chunking"] == {"target_words": 20, "min_words": 8}
    assert result["explainer"]["chunk_rule"]["hi"] == 0.7
    assert result["reference"]["split"] == "val"
    assert 0 < result["reference"]["n"] <= 20  # the actual count; val is small here
    for case in result["cases"]:
        assert case["label"] in ("human", "ai", "mixed")
        assert len(case["top_features"]) == 5
        assert all(f["typical_human"] is not None for f in case["top_features"])
        assert case["text"] and case["chunks"]
    reference = json.loads((out / "reference.json").read_text())
    assert "sty_mattr50" in reference

    md = (out / "review.md").read_text()
    assert md.count("|  |  |") == 8
    assert all(c["id"] in md for c in result["cases"])
    page = (out / "review.html").read_text()
    assert page.count('<section class="case">') == 8

    # Same seed, same sample; a different seed, a different one.
    again = data["root"] / "again"
    assert script.main(_argv(data, "--n", "8", "--reference-n", "0", "--chunk-lo", "0.2",
                             "--chunk-hi", "0.9", "--out-dir", str(again))) == 0
    result2 = json.loads((again / "explanations.json").read_text())
    assert [c["id"] for c in result2["cases"]] == [c["id"] for c in result["cases"]]
    assert result2["reference"] is None and not (again / "reference.json").exists()
    assert result2["explainer"]["chunk_rule"]["lo"] == 0.2


def test_explain_script_with_stacked_model(data) -> None:
    """The production setup: a saved StackedModel whose feature member is explained."""
    from skolfie.models.stacker import StackedModel, Stacker

    root = data["root"]
    members = {"features": root / "features.joblib", "tfidf": root / "tfidf.joblib"}
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    P = np.clip(np.column_stack([y * 0.6 + rng.uniform(0, 0.4, 40)] * 2), 0.01, 0.99)
    stacked = StackedModel(members, Stacker().fit(P, y),
                           {"ai_threshold": 0.8, "human_threshold": 0.2})
    joblib.dump(stacked, root / "stacked.joblib")

    script = _load_script("explain")
    loaded = joblib.load(root / "stacked.joblib")
    features = script.reuse_or_load(loaded, root / "stacked.joblib", members["features"])
    assert features is loaded._member("features").obj
    assert script.reuse_or_load(loaded, root / "stacked.joblib", root / "stacked.joblib") is loaded

    out = root / "stacked-out"
    assert script.main(["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
                        "--model", str(root / "stacked.joblib"), "--n", "6",
                        "--reference-n", "0", "--out-dir", str(out)]) == 0
    result = json.loads((out / "explanations.json").read_text())
    assert result["explainer"]["feature_model"] == "features-nolm"
    assert result["explainer"]["cutoffs"]["ai_threshold"] == 0.8
    for case in result["cases"]:
        expected = ("ai" if case["p_ai"] > 0.8 else "human" if case["p_ai"] < 0.2
                    else "uncertain")
        assert case["label"] == case["verdict"] == expected
        assert len(case["top_features"]) == 5


def test_explain_script_usage_errors(data) -> None:
    script = _load_script("explain")
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--split", "nope"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--chunk-lo", "0.8", "--chunk-hi", "0.2"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--mixed-report", str(data["root"] / "missing.json")))
    with pytest.raises(SystemExit):  # tfidf has no feature model inside
        script.main(["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
                     "--model", str(data["root"] / "tfidf.joblib"), "--reference-n", "0"])


def test_stratified_ids() -> None:
    import pandas as pd

    df = pd.DataFrame({"label": ["human"] * 3 + ["ai"] * 10},
                      index=[f"h{i}" for i in range(3)] + [f"a{i}" for i in range(10)])
    script = _load_script("explain")
    ids = script.stratified_ids(df, 8, seed=0)
    assert len(ids) == len(set(ids)) == 8
    assert sum(i.startswith("h") for i in ids) == 3  # topped up with AI texts
    assert script.stratified_ids(df, 8, seed=0) == ids
    assert len(script.stratified_ids(df, 50, seed=0)) == 13
