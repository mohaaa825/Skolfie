"""
tests/test_eval_fairness.py
----------------------------
Roadmap step 10: Wilson intervals, per-group rates, the "far above average"
flag rule, the groupings and their wording, and scripts/eval_fairness.py end
to end on the synthetic evaluation corpus with a fixture fairness-sets file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from skolfie.eval.fairness import (
    LENGTH_LABELS,
    PROFICIENCY_LABELS,
    FlagRule,
    cap_per_stratum,
    describe_group,
    evaluate_fairness,
    external_groupings,
    flag_groups,
    group_rates,
    known_limitations,
    length_bucket,
    proficiency_band,
    short_pct,
    threshold_for_fpr,
    wilson_interval,
)
from skolfie.models.baselines import LengthBaseline, TfidfLRBaseline
from tests.test_evaluate import _AI_WORDS, _HUMAN_WORDS, _text, make_corpus, make_splits
from tests.test_stacker import _load_script


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #
def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    low, high = wilson_interval(1, 100)
    assert low == pytest.approx(0.00177, abs=1e-5) and high == pytest.approx(0.05448, abs=1e-5)
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    assert wilson_interval(10, 10) == (pytest.approx(0.7225, abs=1e-4), 1.0)
    assert all(math.isnan(x) for x in wilson_interval(0, 0))
    # A wider interval at a higher confidence.
    assert wilson_interval(5, 10, z=2.576)[0] < wilson_interval(5, 10)[0]
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)


def test_group_rates() -> None:
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b", None]})
    p = [0.9, 0.2, 0.71, 0.1, 0.7, 0.99]
    rows = group_rates(df, p, threshold=0.7, by="g")
    assert [r["group"] for r in rows] == ["a", "b"]  # the row without a group is left out
    a, b = rows
    assert (a["n"], a["k"], b["n"], b["k"]) == (3, 2, 2, 0)  # 0.7 is not above 0.7
    assert a["rate"] == pytest.approx(2 / 3)
    assert (a["ci_low"], a["ci_high"]) == pytest.approx(wilson_interval(2, 3))
    # Labels as an array, in category order.
    cats = pd.Categorical(["z", "y", "z", "y", "z", "y"], categories=["z", "y", "x"])
    rows = group_rates(df, p, 0.5, by=cats)
    assert [r["group"] for r in rows] == ["z", "y"]
    with pytest.raises(ValueError):
        group_rates(df, p[:3], 0.5, by="g")


def test_threshold_for_fpr() -> None:
    p = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
    t = threshold_for_fpr(p, 0.05)
    assert np.mean(p > t) <= 0.05
    assert np.mean(p > t - 0.011) > 0.05  # the lowest such cut-off on this grid
    with pytest.raises(ValueError):
        threshold_for_fpr([], 0.01)


# --------------------------------------------------------------------------- #
# Flag rule
# --------------------------------------------------------------------------- #
def _row(n: int, k: int) -> dict:
    low, high = wilson_interval(k, n)
    return {"group": f"{k}/{n}", "n": n, "k": k, "rate": k / n, "ci_low": low, "ci_high": high}


def test_flag_rule_edge_cases() -> None:
    ref = 0.01
    rows = flag_groups([_row(200, 24),   # 12%: 12x, CI 7.9-17% clears 1%
                        _row(20, 10),    # 50% but n < 30
                        _row(30, 1),     # 3.3%: 3x and +2.3 pp, but CI 0.6-17% reaches 1%
                        _row(1000, 15),  # 1.5%: CI clears 1% but not 2x
                        _row(5000, 0),
                        _row(30, 30)], ref)
    assert [r["status"] for r in rows] == ["far_above", "too_small", "above_uncertain", "ok",
                                           "ok", "far_above"]
    assert rows[0]["far_above"] and not rows[2]["far_above"]
    assert rows[0]["ratio"] == pytest.approx(12.0)
    # 2x but under the absolute margin: 0.4% vs 0.2% is not a finding.
    assert flag_groups([_row(100_000, 400)], 0.002)[0]["status"] == "ok"
    # A stricter rule.
    strict = FlagRule(min_ratio=20, min_excess=0.0, min_n=10)
    assert flag_groups([_row(200, 24)], ref, strict)[0]["status"] == "ok"
    assert flag_groups([_row(20, 10)], ref, strict)[0]["status"] == "far_above"
    # A zero reference: any clear hit is far above, the ratio is infinite.
    zero = flag_groups([_row(100, 10), _row(100, 0)], 0.0)
    assert zero[0]["status"] == "far_above" and zero[0]["ratio"] == math.inf
    assert zero[1]["status"] == "ok"
    with pytest.raises(ValueError):
        flag_groups([_row(10, 1)], float("nan"))
    with pytest.raises(ValueError):
        FlagRule(min_ratio=0.5)
    assert "2x" in FlagRule().describe() and "30" in FlagRule().describe()


# --------------------------------------------------------------------------- #
# Groupings
# --------------------------------------------------------------------------- #
def test_length_buckets() -> None:
    got = length_bucket([0, 49, 50, 99, 100, 199, 200, 399, 400, 5000, np.nan])
    assert list(got.categories) == list(LENGTH_LABELS)
    assert [None if pd.isna(x) else x for x in got] == [
        "<50", "<50", "50-99", "50-99", "100-199", "100-199", "200-399", "200-399", "400+",
        "400+", None]


def test_proficiency_bands() -> None:
    got = proficiency_band([1.0, 2.0, 2.5, 3.0, 3.5, 5.0, 2.75, np.nan])
    assert list(got.categories) == list(PROFICIENCY_LABELS)
    low, mid, high = PROFICIENCY_LABELS
    assert [None if pd.isna(x) else x for x in got] == [low, low, mid, mid, high, high, mid,
                                                        None]


def test_cap_per_stratum() -> None:
    df = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 3, "v": range(13)})
    out = cap_per_stratum(df, [df["a"]], 4, seed=0)
    assert (out["a"] == "x").sum() == 4 and (out["a"] == "y").sum() == 3
    assert list(out.index) == sorted(out.index)  # original order kept
    assert out.equals(cap_per_stratum(df, [df["a"]], 4, seed=0))
    assert cap_per_stratum(df, [df["a"]], None, seed=0) is df


def fairness_sets(seed: int = 0, toefl_words: list[str] | None = None) -> pd.DataFrame:
    """A fixture in the schema of scripts/build_fairness_sets.py's fairness_sets.parquet.

    TOEFL essays use AI-like wording by default, so a detector trained on the
    synthetic corpus flags them.
    """
    rng = np.random.default_rng(seed)
    specs = [  # set, group, writer, edited_by_ai, n, words
        ("toefl_real", "non-native: TOEFL essays", "non-native", False, 40,
         toefl_words or _AI_WORDS),
        ("college_real", "native: US college essays", "native", False, 40, _HUMAN_WORDS),
        ("toefl_gpt4polished", "non-native, GPT-4-polished: TOEFL essays", "non-native", True,
         20, _AI_WORDS),
        ("college_gpt3", "AI: ChatGPT-3.5 college essays", "ai", False, 20, _AI_WORDS),
        ("ellipse", "non-native: ELLIPSE learner essays", "non-native", False, 60,
         _HUMAN_WORDS),
    ]
    rows = []
    for name, group, writer, edited, n, words in specs:
        for i in range(n):
            text = _text(rng, list(words), int(rng.integers(3, 7)), words is _AI_WORDS)
            ellipse = name == "ellipse"
            rows.append({
                "id": f"fair:{name}-{i}", "text": text,
                "label": "ai" if writer == "ai" else "human",
                "generator": "chatgpt-3.5" if writer == "ai" else "human",
                "domain": "academic",
                "subdomain": name, "source": "fairness", "split_hint": "",
                "n_words": len(text.split()), "set": name, "group": group, "writer": writer,
                "edited_by_ai": edited,
                "proficiency": float(1 + (i % 9) * 0.5) if ellipse else np.nan,
                "grade": float(8 + i % 5) if ellipse else np.nan,
            })
    return pd.DataFrame(rows)


def test_external_groupings() -> None:
    ext = fairness_sets()
    groups = {g.key: g for g in external_groupings(ext)}
    assert set(groups) == {"ext_set", "ext_writer", "ext_length", "ellipse_proficiency",
                           "ellipse_grade", "ext_ai_edited", "ext_ai"}
    # Human tables never hold AI-edited or AI texts.
    assert len(groups["ext_set"].positions) == 40 + 40 + 60
    assert set(groups["ext_writer"].labels.dropna()) == {"non-native", "native"}
    assert len(groups["ext_ai_edited"].positions) == 20 and len(groups["ext_ai"].positions) == 20
    assert len(groups["ellipse_grade"].positions) == 60
    assert list(groups["ellipse_grade"].labels.cat.categories) == [f"grade {g}"
                                                                   for g in range(8, 13)]
    with pytest.raises(ValueError):
        external_groupings(ext.drop(columns=["writer"]))


# --------------------------------------------------------------------------- #
# Known limitations
# --------------------------------------------------------------------------- #
def test_known_limitations_wording() -> None:
    result = {"reference": {"rate": 0.01}, "tables": [
        {"key": "ext_writer", "kind": "human", "user_facing": True, "rows": [
            {**_row(100, 12), "group": "non-native", "status": "far_above"},
            {**_row(100, 1), "group": "native", "status": "ok"}]},
        {"key": "corpus_technical_length", "kind": "human", "user_facing": True, "rows": [
            {**_row(300, 60), "group": "<50", "status": "far_above"}]},
        {"key": "corpus_source", "kind": "human", "user_facing": False, "rows": [
            {**_row(300, 90), "group": "hc3", "status": "far_above"}]},
        {"key": "ext_ai", "kind": "ai", "user_facing": False, "rows": [
            {**_row(300, 90), "group": "gpt", "status": "far_above"}]},
    ]}
    items = known_limitations(result)
    assert [it["id"] for it in items] == ["corpus_technical_length:under-50",
                                         "ext_writer:non-native"]
    assert items[1]["message"] == (
        "Essays by non-native English writers are flagged as AI 12% of the time (vs 1% for "
        "human text in our test data). Treat an AI verdict on such text with caution.")
    assert items[0]["message"].startswith("Technical texts under 50 words are flagged as AI 20%")
    assert describe_group("corpus_length", "400+") == "Texts of 400 words or more"
    assert describe_group("corpus_length", "50-99") == "Texts of 50 to 99 words"
    assert describe_group("corpus_domain", "qa").startswith("Answers")
    assert describe_group("ellipse_grade", "grade 8") == "Essays by English learners in grade 8"
    assert describe_group("ext_set", "native: US college essays") == (
        "US college essays by native English writers")
    assert describe_group("ext_set", "Stanford CS224N abstracts") == (
        "Texts like the Stanford CS224N abstracts")
    assert short_pct(0.123) == "12%" and short_pct(0.015) == "1.5%" and short_pct(0.01) == "1%"


def test_evaluate_fairness_in_memory() -> None:
    human = pd.DataFrame({
        "text": ["word " * n for n in [10] * 60 + [120] * 60],
        "domain": ["technical"] * 60 + ["news"] * 60, "source": "s",
        "n_words": [10] * 60 + [120] * 60,
        "splits": [["test"]] * 100 + [["test", "test_short"]] * 20})
    p = np.r_[np.full(15, 0.9), np.full(45, 0.1), np.full(60, 0.1)]  # short technical: 25%
    res = evaluate_fairness(human, p, 0.5)
    assert res["reference"]["rate"] == pytest.approx(15 / 120)
    tables = {t["key"]: t for t in res["tables"]}
    split_rows = {r["group"]: r["n"] for r in tables["corpus_split"]["rows"]}
    assert split_rows == {"test": 120, "test_short": 20}
    tech = {r["group"]: r for r in tables["corpus_technical_length"]["rows"]}
    # 25% vs 12.5%: twice the reference, 12.5 pp above it, CI 15.8-37.5% clears it.
    assert tech["<50"]["k"] == 15 and tech["<50"]["status"] == "far_above"
    assert {it["id"] for it in res["known_limitations"]} == {
        "corpus_domain:technical", "corpus_length:under-50",
        "corpus_technical_length:under-50"}
    assert "features" not in res
    # Half as many hits: 2x and +2 pp still hold, but the interval reaches the reference.
    p[7:15] = 0.1
    res = evaluate_fairness(human, p, 0.5)
    assert res["known_limitations"] == [] and {
        r["status"] for t in res["tables"] for r in t["rows"]} == {"ok", "above_uncertain",
                                                                    "too_small"}


# --------------------------------------------------------------------------- #
# scripts/eval_fairness.py
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from skolfie.models.feature_model import FeatureModel
    from skolfie.models.stacker import StackedModel, Stacker

    root = tmp_path_factory.mktemp("fairnessdata")
    corpus = make_corpus(n_per_cell=40, seed=5)
    corpus["domain"] = corpus["domain"].replace({"casual": "technical"})
    splits = make_splits(corpus, seed=5)
    corpus.to_parquet(root / "corpus.parquet", index=False)
    splits.to_parquet(root / "splits.parquet", index=False)
    (root / "splits_manifest.json").write_text(json.dumps({"splits_version": "s-fair",
                                                           "corpus_version": "v-fair"}))
    train = corpus[corpus["id"].isin(splits.loc[splits["split"] == "train", "id"])]
    texts, labels = train["text"].tolist(), train["label"].tolist()
    joblib.dump(TfidfLRBaseline(n_jobs=1).fit(texts, labels), root / "tfidf.joblib")
    joblib.dump(LengthBaseline().fit(texts, labels), root / "length.joblib")
    members = {"tfidf": root / "tfidf.joblib", "length": root / "length.joblib"}
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20)
    P = np.clip(np.column_stack([y * 0.6 + rng.uniform(0, 0.4, 40)] * 2), 0.01, 0.99)
    joblib.dump(StackedModel(members, Stacker().fit(P, y),
                             {"ai_threshold": 0.6, "human_threshold": 0.3, "max_fpr": 0.01}),
                root / "stacked.joblib")
    # Unfitted is enough for the label-free style features.
    joblib.dump(FeatureModel(use_lm=False, n_jobs=1, log=lambda m: None),
                root / "features.joblib")
    fairness_sets().to_parquet(root / "fairness_sets.parquet", index=False)
    (root / "fairness_manifest.json").write_text(json.dumps({"n_rows": 180, "sources": {
        "liang2023": {"title": "GPT detectors are biased", "license": "MIT",
                      "citation": "Liang et al. (2023)", "url": "https://example.org/l"},
        "ellipse": {"title": "ELLIPSE", "license": "CC BY 4.0",
                    "citation": "Crossley et al. (2023)", "url": "https://example.org/e"}}}))
    return {"root": root, "corpus": root / "corpus.parquet", "splits": root / "splits.parquet",
            "sets": root / "fairness_sets.parquet"}


def _argv(data: dict[str, Path], *extra: str) -> list[str]:
    return ["--corpus", str(data["corpus"]), "--splits", str(data["splits"]), *extra]


def test_script_end_to_end_with_stacked_model(data, capsys) -> None:
    root = data["root"]
    out = root / "stacked-out"
    script = _load_script("eval_fairness")
    assert script.main(_argv(data, "--fairness-sets", str(data["sets"]),
                             "--model", f"stacked={root / 'stacked.joblib'}",
                             "--features-model", str(root / "features.joblib"),
                             "--feature-ai-n", "15", "--out-dir", str(out))) == 0
    assert "known_limitations.json" in capsys.readouterr().out

    report = json.loads((out / "fairness_report.json").read_text())
    res = report["models"]["stacked"]
    assert res["threshold"] == 0.6 and res["threshold_info"]["source"] == "model_cutoffs"
    assert report["eval_splits"] == ["test", "test_ood_domain"]
    assert report["corpus_version"] == "v-fair"
    keys = [t["key"] for t in res["tables"]]
    for key in ("corpus_split", "corpus_domain", "corpus_length", "corpus_technical_length",
                "corpus_source", "ext_set", "ext_writer", "ext_length", "ellipse_proficiency",
                "ellipse_grade", "ext_ai_edited", "ext_ai"):
        assert key in keys
    tables = {t["key"]: t for t in res["tables"]}
    assert {r["group"] for r in tables["corpus_domain"]["rows"]} == {"news", "technical",
                                                                     "academic"}
    assert {r["group"] for r in tables["corpus_split"]["rows"]} == {"test", "test_ood_domain"}
    ref = res["reference"]
    assert ref["n"] == sum(r["n"] for r in tables["corpus_domain"]["rows"])

    # The TOEFL fixture is written with AI-like words: flagged far above the reference.
    toefl = {r["group"]: r for r in tables["ext_set"]["rows"]}["non-native: TOEFL essays"]
    assert toefl["status"] == "far_above" and toefl["rate"] > 0.5
    assert all("status" not in r for r in tables["ext_ai"]["rows"])
    limits = json.loads((out / "known_limitations.json").read_text())
    assert limits["model"] == "stacked" and limits["threshold"] == 0.6
    msgs = [it["message"] for it in limits["limitations"]]
    assert any(m.startswith("TOEFL essays by non-native English writers are flagged as AI")
               for m in msgs)
    assert any(m.startswith("Essays by non-native English writers") for m in msgs)
    assert limits["limitations"] == res["known_limitations"]
    # Far above too, but confounded with the set: not a user-facing caveat.
    assert any(r["status"] == "far_above" for r in tables["ext_length"]["rows"])
    assert not any(it["grouping"] == "ext_length" for it in limits["limitations"])

    feats = res["features"]
    assert "sty_mattr50" in feats["names"] and "lm_mean_nll" not in feats["names"]
    assert feats["n_corpus_ai"] == 15 and feats["corpus_ai"]["sty_mattr50"] is not None
    assert "non-native: TOEFL essays" in feats["groups"]["ext_set"]

    md = (out / "fairness_report.md").read_text()
    for part in ("# Fairness check: stacked", "### Human false-positive rate by subgroup",
                 "### External sets: human writing", "### Human texts edited by AI",
                 "### AI texts: recall", "### Why: feature medians", "### Known limitations",
                 "## Data sources", "MIT", "CC BY 4.0", "**far above**", "@CAPS1"):
        assert part in md, part
    # Licences and citations come from the manifest next to the sets.
    assert report["sources_from"].endswith("fairness_manifest.json")
    assert report["fairness_sets"]["manifest"]["n_rows"] == 180


def test_script_threshold_from_val_and_missing_sets(data) -> None:
    root = data["root"]
    out = root / "compare-out"
    script = _load_script("eval_fairness")
    assert script.main(_argv(data, "--fairness-sets", str(root / "missing.parquet"),
                             "--model", f"tfidf={root / 'tfidf.joblib'}",
                             "--model", f"length={root / 'length.joblib'}",
                             "--max-fpr", "0.05", "--max-per-group", "10",
                             "--out-dir", str(out))) == 0
    report = json.loads((out / "fairness_report.json").read_text())
    assert list(report["models"]) == ["tfidf", "length"]
    assert "not found" in report["fairness_sets"]["note"]
    corpus = pd.read_parquet(data["corpus"])
    splits = pd.read_parquet(data["splits"])
    val_human = corpus[corpus["id"].isin(splits.loc[splits["split"] == "val", "id"])
                       & (corpus["label"] == "human")]
    tfidf = joblib.load(root / "tfidf.joblib")
    for name, res in report["models"].items():
        info = res["threshold_info"]
        assert info["source"] == "val_fpr" and info["split"] == "val"
        assert info["n_human"] == len(val_human) and info["fpr_on_split"] <= 0.05
        assert all(t["section"] == "corpus" for t in res["tables"])
        assert "features" not in res and "no feature model" in res["features_note"]
    p = tfidf.predict_proba(val_human["text"].tolist())
    assert report["models"]["tfidf"]["threshold"] == pytest.approx(threshold_for_fpr(p, 0.05))
    # The cap applies per domain x length bucket.
    assert all(r["n"] <= 20 for r in next(
        t for t in report["models"]["tfidf"]["tables"] if t["key"] == "corpus_domain")["rows"])
    md = (out / "fairness_report.md").read_text()
    assert "## Summary" in md and "Skipped:" in md
    assert json.loads((out / "known_limitations.json").read_text())["model"] == "tfidf"

    # A fixed threshold for every model; no --fairness-sets at all.
    out2 = root / "fixed-out"
    assert script.main(_argv(data, "--model", str(root / "tfidf.joblib"), "--threshold", "0.9",
                             "--no-features", "--out-dir", str(out2))) == 0
    res = json.loads((out2 / "fairness_report.json").read_text())["models"]["tfidf"]
    assert res["threshold"] == 0.9 and res["threshold_info"]["source"] == "given"
    assert res["external_note"] == "no --fairness-sets given"


def test_data_sources_fallbacks(tmp_path) -> None:
    script = _load_script("eval_fairness")
    sources, origin = script.data_sources({})
    assert origin.startswith("skolfie.") and sources
    assert all(isinstance(s, dict) and s.get("name") for s in sources)
    licences = " ".join(str(s.get("licence") or s.get("license")) for s in sources)
    assert "MIT" in licences and "CC BY 4.0" in licences
    path = tmp_path / "fairness_sets.parquet"
    assert script.fairness_manifest(path) == {}
    (tmp_path / "fairness_manifest.json").write_text(json.dumps({"sources": [
        {"name": "x", "license": "MIT"}]}))
    manifest = script.fairness_manifest(path)
    assert script.data_sources(manifest) == ([{"name": "x", "license": "MIT"}],
                                             str(tmp_path / "fairness_manifest.json"))


def test_script_usage_errors(data) -> None:
    root = data["root"]
    script = _load_script("eval_fairness")
    model = ["--model", str(root / "tfidf.joblib")]
    with pytest.raises(SystemExit):  # no --model
        script.main(_argv(data))
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--model", str(root / "nope.joblib")))
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--eval-splits", "test,nope"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--threshold-split", "nope"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--model", f"tfidf={root / 'length.joblib'}"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--threshold", "1.5"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--min-ratio", "0.5"))
    bad = root / "bad_sets.parquet"
    fairness_sets().drop(columns=["edited_by_ai"]).to_parquet(bad, index=False)
    with pytest.raises(SystemExit):
        script.main(_argv(data, *model, "--fairness-sets", str(bad)))
