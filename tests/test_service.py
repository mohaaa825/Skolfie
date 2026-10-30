"""
tests/test_service.py
----------------------
Roadmap step 11: `skolfie.service.Detector` and `make_bundle` with small fake
models saved the way Colab saves the real ones (a StackedModel whose members
are recorded at absolute Colab paths), so nothing needs torch, GPT-2 or the
network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pytest

from skolfie import service
from skolfie.models.stacker import StackedModel, Stacker
from skolfie.service import (
    DISCLAIMER,
    MAX_TEXT_CHARS,
    MIN_TEXT_WORDS,
    SHORT_TEXT_WORDS,
    Detector,
    english_share,
    looks_non_english,
    make_bundle,
)

# --------------------------------------------------------------------------- #
# Fake models (module level, so joblib can pickle them by reference)
# --------------------------------------------------------------------------- #
AI_WORDS = frozenset({"moreover", "furthermore", "crucial", "tapestry", "multifaceted",
                      "delve", "pivotal"})
FEATURES = ("sty_mattr50", "sty_hapax50", "sty_semicolons", "sty_word_len_mean",
            "sty_sent_len_mean")


def ai_share(text: str) -> float:
    words = [w.strip(".,;:!?").lower() for w in text.split()]
    return sum(w in AI_WORDS for w in words) / max(1, len(words))


class FakeScorer:
    """P(ai) from the share of 'AI words' (stands in for the transformer)."""

    def __init__(self, name: str = "transformer") -> None:
        self.name = name

    def predict_proba(self, texts):
        return np.array([min(0.98, 0.04 + 6 * ai_share(t)) for t in texts])


class FakeBooster:
    def predict(self, X, pred_contrib=False, num_iteration=None):
        X = np.asarray(X, dtype=np.float64)
        return np.column_stack([(X - 0.05) * 10, np.zeros(len(X))])


class FakeClf:
    best_iteration_ = None

    def __init__(self) -> None:
        self.booster_ = FakeBooster()


class FakeFeatureModel:
    """Looks like the LightGBM FeatureModel to `explain_features` (SHAP via the booster)."""

    name = "features"
    classifier = "lgbm"
    feature_names_ = list(FEATURES)

    def __init__(self) -> None:
        self.clf = FakeClf()

    def featurize(self, texts):
        return np.array([[ai_share(t) * (1 + 0.1 * k) for k in range(len(FEATURES))]
                         for t in texts], dtype=np.float64)

    def predict_proba(self, texts):
        raw = self.clf.booster_.predict(self.featurize(texts), pred_contrib=True).sum(axis=1)
        return 1 / (1 + np.exp(-raw))


class FakeBinoFeatureModel(FakeFeatureModel):
    """A feature model that reports a Binoculars pair, like FeatureModel(bino_observer=...)."""

    def language_models(self):
        return {"lm": "gpt2", "bino_observer": "Qwen/Qwen2.5-1.5B",
                "bino_performer": "Qwen/Qwen2.5-1.5B-Instruct", "bino_quantization": None,
                "bino_max_tokens": 256}


# --------------------------------------------------------------------------- #
# Texts
# --------------------------------------------------------------------------- #
HUMAN_SENTENCES = (
    "Last summer I worked at my uncle's bakery and it was harder than I thought it would be.",
    "We had to get up at four in the morning, and the ovens were already hot when we came in.",
    "My job was to shape the rolls, which sounds easy but my first ones looked like rocks.",
    "By the end of July I could do a whole tray in ten minutes without dropping any of them.",
    "What I learned most is that you have to keep going even when your hands are tired.",
)
AI_SENTENCES = (
    "Moreover, the crucial tapestry of modern education is multifaceted and pivotal.",
    "Furthermore, we must delve into the crucial and multifaceted tapestry of learning.",
    "Moreover, this pivotal journey offers a crucial lens on a multifaceted tapestry.",
)
SWEDISH = ("Skolan är en viktig plats för barn och ungdomar. Där lär de sig inte bara att läsa "
           "och skriva, utan också hur man samarbetar med andra människor. Många lärare menar "
           "att det svåraste med yrket är att hinna se varje elev, eftersom klasserna ofta är "
           "stora och tiden är begränsad. Därför behöver skolan mer resurser.")
GERMAN = ("Die Schule ist ein wichtiger Ort für Kinder und Jugendliche. Dort lernen sie nicht "
          "nur lesen und schreiben, sondern auch, wie man mit anderen Menschen "
          "zusammenarbeitet. Viele Lehrer sagen, dass das Schwierigste an ihrem Beruf ist, "
          "jeden Schüler zu sehen, weil die Klassen oft groß sind und die Zeit begrenzt ist.")
LIMITATIONS = [
    {"id": "toefl:non_native", "grouping": "toefl", "group": "non-native", "n": 91,
     "rate": 0.25, "ci_low": 0.17, "ci_high": 0.35, "reference_rate": 0.01,
     "message": "Essays by non-native English writers are flagged as AI 25% of the time."},
    {"id": "length:50_99", "grouping": "length", "group": "50-99", "n": 40, "rate": 0.1,
     "ci_low": 0.04, "ci_high": 0.2, "reference_rate": 0.01,
     "message": "Texts of 50-99 words are flagged as AI 10% of the time."},
]
COLAB = "/content/drive/MyDrive/skolfie/models"


def essay(sentences, n_words: int) -> str:
    out, k = [], 0
    while sum(len(s.split()) for s in out) < n_words:
        out.append(sentences[k % len(sentences)])
        k += 1
    return " ".join(out)


HUMAN_ESSAY = essay(HUMAN_SENTENCES, 200)
AI_ESSAY = essay(AI_SENTENCES, 200)
MIXED_ESSAY = essay(HUMAN_SENTENCES, 120) + "\n\n" + essay(AI_SENTENCES, 120)


def make_colab_dirs(root: Path, *, chunks: bool = True, reports: bool = True) -> dict[str, Path]:
    """models/ and reports/ as the Colab notebooks leave them on Drive."""
    models, rep = root / "models", root / "reports"
    models.mkdir(parents=True)
    rep.mkdir(parents=True)
    joblib.dump(FakeFeatureModel(), models / "features.joblib")
    joblib.dump(FakeScorer("transformer"), models / "transformer.joblib")
    if chunks:
        joblib.dump(FakeScorer("transformer-chunks"), models / "transformer-chunks.joblib")
    P = np.array([[0.05, 0.05], [0.1, 0.08], [0.9, 0.92], [0.97, 0.95]])
    stacked = StackedModel({"features": f"{COLAB}/features.joblib",
                            "transformer": f"{COLAB}/transformer.joblib"},
                           Stacker().fit(P, [0, 0, 1, 1]),
                           {"ai_threshold": 0.7, "human_threshold": 0.3})
    joblib.dump(stacked, models / "stacked.joblib")
    if reports:
        report = {"rules": {"chunk": {"params": {"lo": 0.2, "hi": 0.8, "min_chunks": 2,
                                                 "min_share": 0.0}}},
                  "chunking": {"target_words": 30, "min_words": 10}}
        for name in ("mixed-transformer-chunks", "mixed-transformer"):
            (rep / name).mkdir()
            (rep / name / "mixed_report.json").write_text(json.dumps(report))
        (rep / "explain").mkdir()
        (rep / "explain" / "reference.json").write_text(json.dumps(
            {f: {"median": 0.1, "human": 0.0, "ai": 0.3} for f in FEATURES}))
        (rep / "fairness" / "stacked").mkdir(parents=True)
        (rep / "fairness" / "stacked" / "known_limitations.json").write_text(
            json.dumps({"model": "stacked", "threshold": 0.7,  # eval_fairness.py's shape
                        "limitations": LIMITATIONS}))
    return {"root": root, "models": models, "reports": rep}


@pytest.fixture(scope="module")
def dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return make_colab_dirs(tmp_path_factory.mktemp("colab"))


@pytest.fixture(scope="module")
def detector(dirs) -> Detector:
    return Detector.from_dirs(dirs["models"], dirs["reports"])


def kinds(result: dict) -> list[str]:
    return [w["kind"] for w in result["warnings"]]


# --------------------------------------------------------------------------- #
# Language heuristic
# --------------------------------------------------------------------------- #
def test_language_heuristic() -> None:
    from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

    for text in (*HUMAN_TEXTS, *AI_TEXTS, HUMAN_ESSAY, AI_ESSAY):
        assert not looks_non_english(text), text[:60]
    assert english_share(HUMAN_ESSAY)[0] > 0.25
    assert looks_non_english(SWEDISH) and looks_non_english(GERMAN)
    assert english_share(GERMAN)[0] < 0.05
    assert not looks_non_english("Hej och välkommen")  # too few words to judge
    # Look-alike letters do not hide English function words.
    assert not looks_non_english(HUMAN_ESSAY.replace("e", "е"))


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #
def test_check_result_shape(detector) -> None:
    out = detector.check_many([HUMAN_ESSAY, AI_ESSAY, MIXED_ESSAY])
    json.loads(json.dumps(out, allow_nan=False))  # plain JSON as it is
    human, ai, mixed = out
    assert human["label"] == "human" and human["headline"] == "Likely written by a human"
    assert ai["label"] == "ai" and ai["headline"] == "Likely AI-generated"
    assert mixed["label"] == "mixed" and mixed["headline"].startswith("Mixed")
    assert human["p_ai"] < 0.3 < 0.7 < ai["p_ai"]
    for res, text in zip(out, [HUMAN_ESSAY, AI_ESSAY, MIXED_ESSAY]):
        assert res["n_words"] == len(text.split())
        assert list(res) ==["label", "headline", "p_ai", "n_words", "verdict", "chunks",
                             "highlights", "top_features", "feature_model", "obfuscation",
                             "reasons", "notes", "warnings", "disclaimer"]
        assert res["disclaimer"] == DISCLAIMER
        assert res["reasons"] == [f["reason"] for f in res["top_features"]]
        assert len(res["reasons"]) == 5
        assert "short_text" not in kinds(res) and "language" not in kinds(res)
    # The reference values reach the reasons ("typical of AI text").
    assert all(f["typical_ai"] == 0.3 for f in ai["top_features"])
    assert any("typical of AI text" in r for r in ai["reasons"])
    assert detector.check(HUMAN_ESSAY) == human


def test_too_short_and_short_text(detector) -> None:
    res = detector.check("Only a few words here, not enough to say anything at all.")
    assert res["label"] == "too_short" and res["p_ai"] is None
    assert res["headline"] == f"Too short to judge (needs at least {MIN_TEXT_WORDS} words)"
    assert res["chunks"] == [] and res["top_features"] == [] and res["reasons"] == []
    assert res["warnings"] == [] and res["disclaimer"] == DISCLAIMER
    assert detector.check("")["label"] == "too_short"

    short = essay(HUMAN_SENTENCES, 60)
    res = detector.check(short)
    assert res["label"] == "human" and kinds(res) == ["short_text"]
    assert f"{res['n_words']} words" in res["warnings"][0]["message"]
    assert MIN_TEXT_WORDS <= res["n_words"] < SHORT_TEXT_WORDS
    # too_short and scored texts mix in one batch, in order.
    labels = [r["label"] for r in detector.check_many(["one two", short, "three"])]
    assert labels == ["too_short", "human", "too_short"]


def test_fairness_warnings_only_on_ai_or_mixed(detector) -> None:
    human, ai, mixed = detector.check_many([HUMAN_ESSAY, AI_ESSAY, MIXED_ESSAY])
    assert "fairness" not in kinds(human)
    for res in (ai, mixed):
        msgs = [w["message"] for w in res["warnings"] if w["kind"] == "fairness"]
        assert msgs == [item["message"] for item in LIMITATIONS]


def test_obfuscation_and_language_warnings(detector) -> None:
    hidden = HUMAN_ESSAY.replace(" ", " ​", 20)
    res = detector.check(hidden)
    assert kinds(res) == ["obfuscation"] and res["obfuscation"]["invisible"] == 20
    assert res["warnings"][0]["message"].startswith("The text contains 20 invisible")
    res = detector.check(" ".join([SWEDISH] * 3))
    assert "language" in kinds(res)
    assert "English" in [w for w in res["warnings"] if w["kind"] == "language"][0]["message"]
    assert "language" in kinds(detector.check(GERMAN + " " + GERMAN))


def test_disagreement_note(detector) -> None:
    # The feature model alone says AI for every text with AI words; a verdict of human
    # with AI words in it is not possible here, so check a forged explanation.
    ex = detector.explainer.explain(HUMAN_ESSAY)
    ex["feature_model"] = {**ex["feature_model"], "p_ai": 0.9, "agrees": False}
    res = detector._result(HUMAN_ESSAY, 200, ex)
    assert len(res["notes"]) == 1 and res["notes"][0].startswith("The style features alone")


def test_text_limits(detector) -> None:
    with pytest.raises(ValueError, match="too long"):
        detector.check("word " * (MAX_TEXT_CHARS // 5 + 1))
    with pytest.raises(TypeError):
        detector.check(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_member_paths_are_remapped_and_reused(dirs, detector) -> None:
    info = detector.info()
    assert info["members"] == {"features": str(dirs["models"] / "features.joblib"),
                               "transformer": str(dirs["models"] / "transformer.joblib")}
    verdict = detector.explainer.verdict_model
    assert isinstance(verdict, StackedModel)
    # features.joblib is the stacker's member: loaded once, not twice.
    assert detector.explainer.feature_model is verdict._member("features").obj
    assert detector.explainer.chunk_scorer.name == "transformer-chunks"
    assert info["cutoffs"] == {"ai_threshold": 0.7, "human_threshold": 0.3}
    assert info["chunk_rule"]["lo"] == 0.2 and info["mixed_enabled"] is True
    assert info["chunking"] == {"target_words": 30, "min_words": 10}
    assert info["known_limitations"] == 2 and info["missing"] == []
    assert info["files"]["mixed_report"].endswith(
        "mixed-transformer-chunks/mixed_report.json")
    assert info["limits"]["max_text_chars"] == MAX_TEXT_CHARS
    assert "numpy" in info["versions"]
    json.dumps(info, allow_nan=False)


def test_info_names_the_language_models(dirs, detector, tmp_path: Path) -> None:
    # A feature model without language_models() (any duck-typed scorer): None, no note.
    assert detector.info()["language_models"] is None
    bino = make_colab_dirs(tmp_path / "bino")
    joblib.dump(FakeBinoFeatureModel(), bino["models"] / "features.joblib")
    det = Detector.from_dirs(bino["models"], bino["reports"])
    info = det.info()
    assert info["language_models"]["bino_observer"] == "Qwen/Qwen2.5-1.5B"
    assert info["language_models"]["bino_performer"] == "Qwen/Qwen2.5-1.5B-Instruct"
    note = [n for n in info["notes"] if "Binoculars" in n]
    assert note and "Qwen/Qwen2.5-1.5B + Qwen/Qwen2.5-1.5B-Instruct" in note[0]
    assert "GPU" in note[0]
    assert det.check(AI_ESSAY)["label"] in service.LABELS


def test_missing_optional_files(tmp_path: Path) -> None:
    d = make_colab_dirs(tmp_path, chunks=False, reports=False)
    det = Detector.from_dirs(d["models"])  # reports_dir defaults to ../reports
    info = det.info()
    missing = info["missing"]
    assert str(d["models"] / "transformer-chunks.joblib") in missing
    assert str(d["reports"] / "mixed-transformer" / "mixed_report.json") in missing
    assert str(d["reports"] / "explain" / "reference.json") in missing
    assert str(d["reports"] / "fairness" / "stacked" / "known_limitations.json") in missing
    assert info["mixed_enabled"] is False and info["known_limitations"] == 0
    # The chunk model falls back to transformer.joblib, the stacker's member.
    assert det.explainer.chunk_scorer is det.explainer.verdict_model._member("transformer").obj
    human, ai, mixed = det.check_many([HUMAN_ESSAY, AI_ESSAY, MIXED_ESSAY])
    assert mixed["label"] != "mixed" and mixed["chunks"]  # heatmap, but no Mixed
    assert "fairness" not in kinds(ai)
    assert all(f["typical_ai"] is None for f in ai["top_features"])  # no reference values


def test_missing_required_models(tmp_path: Path) -> None:
    d = make_colab_dirs(tmp_path)
    (d["models"] / "transformer.joblib").unlink()
    with pytest.raises(FileNotFoundError) as err:
        Detector.from_dirs(d["models"], d["reports"])
    msg = str(err.value)
    assert "transformer.joblib" in msg and "notebooks/transformer.ipynb" in msg
    assert COLAB in msg

    (d["models"] / "stacked.joblib").unlink()
    with pytest.raises(FileNotFoundError, match=r"stacked\.joblib.*notebooks/stack\.ipynb"):
        Detector.from_dirs(d["models"], d["reports"])
    with pytest.raises(FileNotFoundError, match="models folder not found"):
        Detector.from_dirs(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="bundle"):
        Detector.from_bundle(tmp_path)


def test_bundle_round_trip(dirs, detector, tmp_path: Path) -> None:
    src = make_colab_dirs(tmp_path / "colab")
    manifest_path = make_bundle(src["models"], src["reports"], tmp_path / "bundle")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"] == {
        "verdict": "stacked.joblib", "features": "features.joblib",
        "chunk": "transformer-chunks.joblib", "mixed_report": "mixed_report.json",
        "reference": "reference.json", "known_limitations": "known_limitations.json"}
    assert manifest["members"] == {"features": "features.joblib",
                                   "transformer": "transformer.joblib"}
    assert manifest["missing"] == []
    # Portable: move it and delete the originals.
    moved = tmp_path / "elsewhere" / "skolfie-bundle"
    shutil.move(str(tmp_path / "bundle"), str(moved))
    shutil.rmtree(tmp_path / "colab")
    det = Detector.from_bundle(moved)
    info = det.info()
    assert info["source"] == "bundle" and info["bundle_dir"] == str(moved)
    assert info["members"]["transformer"] == str(moved / "transformer.joblib")
    assert det.explainer.feature_model is det.explainer.verdict_model._member("features").obj
    texts = [HUMAN_ESSAY, AI_ESSAY, MIXED_ESSAY]
    assert det.check_many(texts) == detector.check_many(texts)


def test_bundle_with_missing_optional_files(tmp_path: Path) -> None:
    src = make_colab_dirs(tmp_path / "colab", chunks=False, reports=False)
    manifest = json.loads(make_bundle(src["models"], None, tmp_path / "b").read_text())
    assert manifest["files"]["chunk"] == "transformer.joblib"
    assert manifest["files"]["reference"] is None
    assert any(m.endswith("known_limitations.json") for m in manifest["missing"])
    det = Detector.from_bundle(tmp_path / "b")
    assert any(m.endswith("reference.json") for m in det.info()["missing"])
    assert det.check(AI_ESSAY)["label"] == "ai"


def test_bundle_rejects_paths_in_manifest(tmp_path: Path) -> None:
    src = make_colab_dirs(tmp_path / "colab")
    path = make_bundle(src["models"], src["reports"], tmp_path / "b")
    manifest = json.loads(path.read_text())
    manifest["files"]["verdict"] = "../colab/models/stacked.joblib"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="plain file name"):
        Detector.from_bundle(tmp_path / "b")


def test_mixed_report_helpers(tmp_path: Path) -> None:
    from skolfie.mixed.detector import chunk_rule_from_params, read_mixed_report

    path = tmp_path / "mixed_report.json"
    path.write_text(json.dumps({"rules": {"chunk": {"params": {"lo": 0.1, "hi": 0.9,
                                                               "extra": 1}}}}))
    report = read_mixed_report(path)
    assert report == {"rule": {"lo": 0.1, "hi": 0.9, "extra": 1}, "target_words": None,
                      "min_words": None}
    assert chunk_rule_from_params(report["rule"]).hi == 0.9
    with pytest.raises(ValueError):
        chunk_rule_from_params({"lo": 0.9, "hi": 0.1})
    path.write_text("{}")
    with pytest.raises(ValueError, match="no rules.chunk.params"):
        read_mixed_report(path)
    with pytest.raises(FileNotFoundError):
        read_mixed_report(tmp_path / "missing.json")


def test_constants() -> None:
    assert service.MIN_TEXT_WORDS < service.SHORT_TEXT_WORDS
    assert set(service.HEADLINES) == set(service.LABELS)
    assert "sole basis" in DISCLAIMER


@pytest.mark.parametrize("data", [
    {"model": "stacked", "limitations": [{"message": "m1"}, {"message": " "}]},
    {"known_limitations": [{"message": "m1"}]},
    [{"message": "m1"}, {"no": "message"}],
])
def test_known_limitations_file_shapes(data) -> None:
    assert [i["message"] for i in service._limitations(data, Path("k.json"))] == ["m1"]
    with pytest.raises(ValueError, match="list of items"):
        service._limitations({"model": "stacked"}, Path("k.json"))
