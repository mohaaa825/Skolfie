"""
tests/test_stacker.py
----------------------
Roadmap step 6: the calibrated stacker, the Human / Uncertain / AI cut-offs,
the StackedModel scorer, and scripts/stack.py end to end on the synthetic
evaluation corpus (with the length and TF-IDF baselines as members).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from skolfie.eval.metrics import expected_calibration_error
from skolfie.models.baselines import LengthBaseline
from skolfie.models.stacker import (
    StackedModel,
    Stacker,
    logit,
    operating_points,
    verdict_table,
    verdicts,
)
from tests.test_evaluate import make_corpus, make_splits

REPO_ROOT = Path(__file__).resolve().parent.parent


def _members(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two noisy, miscalibrated member scores for a balanced label vector."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = np.where(y == 1, 1.0, -1.0)
    a = 1 / (1 + np.exp(-(3 * z + rng.normal(0, 2.0, n))))   # overconfident
    b = 1 / (1 + np.exp(-(0.5 * z + rng.normal(0, 0.6, n))))  # underconfident
    return np.column_stack([a, b]), y


# --------------------------------------------------------------------------- #
# Stacker
# --------------------------------------------------------------------------- #
def test_logit_is_finite_at_the_edges() -> None:
    out = logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(out).all() and out[1] == 0 and out[0] == pytest.approx(-out[2])


@pytest.mark.parametrize("calibration", ["platt", "isotonic"])
def test_stacker_beats_members_and_is_calibrated(calibration: str) -> None:
    from sklearn.metrics import roc_auc_score

    P, y = _members()
    P_test, y_test = _members(seed=1)
    stack = Stacker(calibration=calibration).fit(P, y)
    p = stack.predict_proba(P_test)
    assert p.shape == (len(y_test),) and ((p >= 0) & (p <= 1)).all()
    best_member = max(roc_auc_score(y_test, P_test[:, j]) for j in range(2))
    assert roc_auc_score(y_test, p) >= best_member - 0.002
    assert expected_calibration_error(y_test, p) < 0.03
    assert expected_calibration_error(y_test, P_test[:, 1]) > 0.1  # that member was not


def test_stacker_oof_and_weights() -> None:
    P, y = _members(n=1000)
    stack = Stacker(folds=4).fit(P, y)
    oof = stack.oof_predict(P, y)
    assert oof.shape == (1000,) and not np.allclose(oof, stack.predict_proba(P))
    w = stack.weights(["a", "b"])
    assert set(w) == {"a", "b", "intercept"} and w["a"] > 0 and w["b"] > 0


def test_stacker_errors() -> None:
    P, y = _members(n=50)
    with pytest.raises(ValueError):
        Stacker(calibration="beta")
    with pytest.raises(ValueError):
        Stacker(folds=1)
    with pytest.raises(RuntimeError):
        Stacker().predict_proba(P)
    with pytest.raises(ValueError):
        Stacker().fit(P, np.zeros(50))
    with pytest.raises(ValueError):
        Stacker().fit(P[:, :0], y)
    bad = P.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        Stacker().fit(bad, y)


# --------------------------------------------------------------------------- #
# Cut-offs
# --------------------------------------------------------------------------- #
def test_operating_points_respect_targets() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    p = np.r_[rng.beta(2, 5, 1000), rng.beta(5, 2, 1000)]
    cut = operating_points(y, p, max_fpr=0.02, max_miss=0.10)
    assert cut["human_threshold"] <= cut["ai_threshold"]
    t = verdict_table(y, p, cut)
    assert t["human_texts"]["ai"] <= 0.02 and t["ai_texts"]["human"] <= 0.10
    assert t["human_texts"]["n"] == 1000
    for row in t.values():
        assert sum(row[v] for v in ("human", "uncertain", "ai")) == pytest.approx(1)
    assert t["human_texts"]["uncertain"] > 0  # overlapping scores leave a band


def test_operating_points_without_overlap_have_no_band() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    cut = operating_points(y, p, max_fpr=0.01, max_miss=0.01)
    assert cut["human_threshold"] <= cut["ai_threshold"] == 0.2
    v = verdicts(p, cut)
    assert list(v) == ["human", "human", "uncertain", "ai", "ai", "ai"]
    with pytest.raises(ValueError):
        operating_points(y, p, max_fpr=0.0)
    with pytest.raises(ValueError):
        operating_points([0, 0], [0.1, 0.2])


# --------------------------------------------------------------------------- #
# StackedModel
# --------------------------------------------------------------------------- #
def test_stacked_model_round_trip(tmp_path: Path) -> None:
    corpus = make_corpus(n_per_cell=10)
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    paths = {}
    for name, C in (("a", 1.0), ("b", 0.01)):
        paths[name] = tmp_path / f"{name}.joblib"
        joblib.dump(LengthBaseline(C=C).fit(texts, labels), paths[name])
    member_scores = np.column_stack([joblib.load(paths[n]).predict_proba(texts) for n in "ab"])
    stack = Stacker().fit(member_scores, labels)
    cut = operating_points(labels, stack.predict_proba(member_scores))
    model = StackedModel(paths, stack, cut, {"fit_split": "val"})
    assert model.pretrained and model.name == "stacked"
    p = model.predict_proba(texts)
    np.testing.assert_allclose(p, stack.predict_proba(member_scores))
    assert set(model.predict(texts)) <= {"human", "uncertain", "ai"}
    assert model.predict_proba([]).shape == (0,)
    assert model.params()["fit_split"] == "val"

    joblib.dump(model, tmp_path / "stacked.joblib")
    loaded = joblib.load(tmp_path / "stacked.joblib")
    assert loaded._loaded == {}
    np.testing.assert_allclose(loaded.predict_proba(texts), p)
    with pytest.raises(ValueError):
        StackedModel({}, stack, cut)


# --------------------------------------------------------------------------- #
# scripts/stack.py end to end
# --------------------------------------------------------------------------- #
def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"skolfie_{name}_script",
                                                  REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("stackdata")
    corpus = make_corpus(n_per_cell=16, seed=3)
    corpus.to_parquet(root / "corpus.parquet", index=False)
    make_splits(corpus, seed=3).to_parquet(root / "splits.parquet", index=False)
    evaluate = _load_script("evaluate")
    for model in ("length", "tfidf"):
        assert evaluate.main([
            "--corpus", str(root / "corpus.parquet"), "--splits", str(root / "splits.parquet"),
            "--model", model, "--out-dir", str(root / model), "--save-predictions",
            "--save-model", str(root / f"{model}.joblib")]) == 0
    return {"root": root, "corpus": root / "corpus.parquet", "splits": root / "splits.parquet"}


def test_stack_script_end_to_end(reports, capsys) -> None:
    root = reports["root"]
    stack = _load_script("stack")
    out = root / "stacked"
    argv = ["--corpus", str(reports["corpus"]), "--splits", str(reports["splits"]),
            "--member", f"length={root / 'length' / 'predictions.parquet'}",
            "--member", f"tfidf={root / 'tfidf' / 'predictions.parquet'}",
            "--out-dir", str(out), "--save-predictions",
            "--model-path", f"length={root / 'length.joblib'}",
            "--model-path", f"tfidf={root / 'tfidf.joblib'}",
            "--save-model", str(root / "stacked.joblib")]
    assert stack.main(argv) == 0
    printed = capsys.readouterr().out
    assert "# Report card: stacked" in printed and "## Verdicts" in printed

    report = json.loads((out / "report.json").read_text())
    assert report["model"]["params"]["members"] == ["length", "tfidf"]
    assert "out-of-fold" in report["model"]["params"]["note"]
    assert list(report["results"])[0] == "val"
    tfidf = json.loads((root / "tfidf" / "report.json").read_text())
    assert report["results"]["test"]["metrics"]["auroc"] >= (
        tfidf["results"]["test"]["metrics"]["auroc"] - 0.05)
    cut = report["verdicts"]["cutoffs"]
    assert cut["human_threshold"] <= cut["ai_threshold"]
    assert set(report["verdicts"]["splits"]) == set(report["results"])
    preds = pd.read_parquet(out / "predictions.parquet")
    assert preds["p_ai"].between(0, 1).all()

    # The saved StackedModel scores texts itself and matches the report's test scores.
    model = joblib.load(root / "stacked.joblib")
    corpus = pd.read_parquet(reports["corpus"]).set_index("id")
    test = preds[preds["split"] == "test"]
    np.testing.assert_allclose(model.predict_proba(corpus.loc[test["id"], "text"].tolist()),
                               test["p_ai"].to_numpy(), atol=1e-9)


def test_stack_script_usage_errors(reports) -> None:
    stack = _load_script("stack")
    root = reports["root"]
    base = ["--corpus", str(reports["corpus"]), "--splits", str(reports["splits"])]
    member = ["--member", f"length={root / 'length' / 'predictions.parquet'}"]
    with pytest.raises(SystemExit):
        stack.main(base + ["--member", "nopath"])
    with pytest.raises(SystemExit):
        stack.main(base + ["--member", f"x={root / 'missing.parquet'}"])
    with pytest.raises(SystemExit):
        stack.main(base + member + member)  # duplicate names
    with pytest.raises(SystemExit):
        stack.main(base + member + ["--save-model", str(root / "m.joblib")])
    with pytest.raises(SystemExit):
        stack.main(base + member + ["--fit-split", "nope"])
