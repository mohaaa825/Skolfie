"""
tests/test_eval_mixed.py
-------------------------
Roadmap step 7: scripts/eval_mixed.py end to end on the synthetic evaluation
corpus, with a saved TF-IDF baseline as the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from skolfie.models.baselines import TfidfLRBaseline
from tests.test_evaluate import make_corpus, make_splits
from tests.test_stacker import _load_script


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("mixeddata")
    corpus = make_corpus(n_per_cell=60, seed=5)
    splits = make_splits(corpus, seed=5)
    corpus.to_parquet(root / "corpus.parquet", index=False)
    splits.to_parquet(root / "splits.parquet", index=False)
    train = corpus[corpus["id"].isin(splits.loc[splits["split"] == "train", "id"])]
    joblib.dump(TfidfLRBaseline().fit(train["text"].tolist(), train["label"].tolist()),
                root / "tfidf.joblib")
    return {"root": root, "corpus": root / "corpus.parquet", "splits": root / "splits.parquet",
            "model": root / "tfidf.joblib"}


def _argv(data: dict[str, Path], *extra: str) -> list[str]:
    return ["--corpus", str(data["corpus"]), "--splits", str(data["splits"]),
            "--model", str(data["model"]), "--n-mixed", "40", "--n-pure", "20",
            "--target-words", "20", "--min-words", "8", *extra]


def test_eval_mixed_end_to_end(data, capsys) -> None:
    out = data["root"] / "out"
    assert _load_script("eval_mixed").main(_argv(data, "--out-dir", str(out))) == 0
    assert "mixed_report.md" in capsys.readouterr().out

    report = json.loads((out / "mixed_report.json").read_text())
    assert report["sets"]["eval"] == {"n": 80, "n_human": 20, "n_ai": 20, "n_mixed": 40}
    assert set(report["rules"]) == {"chunk", "whole_text"}
    assert report["max_false_mixed"] == 0.05
    for rule in report["rules"].values():
        assert 0 <= rule["metrics"]["macro_f1"] <= 1
        assert {"human_to_mixed_rate", "ai_to_mixed_rate"} <= set(rule["metrics"])
    assert set(report["by_pattern"]) <= {"ha", "ah", "hah", "aha"}
    assert sum(r["n"] for r in report["by_pattern"].values()) == 40
    assert 0 <= report["spans"]["char_f1"] <= 1
    assert report["chunking"]["mean_chunks"] > 1

    preds = pd.read_parquet(out / "mixed_predictions.parquet")
    assert len(preds) == 80
    assert set(preds["pred_chunk"]) <= {"human", "ai", "mixed"}
    assert preds["p_whole"].between(0, 1).all()
    assert (out / "mixed_report.md").read_text().startswith("#")


def test_eval_mixed_usage_errors(data) -> None:
    script = _load_script("eval_mixed")
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--fit-split", "nope"))
    with pytest.raises(SystemExit):
        script.main(_argv(data, "--max-false-mixed", "2"))
    missing = _argv(data)
    missing[missing.index("--model") + 1] = str(data["root"] / "missing.joblib")
    with pytest.raises(SystemExit):
        script.main(missing)
