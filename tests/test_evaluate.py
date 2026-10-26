"""
tests/test_evaluate.py
-----------------------
The evaluation harness end to end: baselines' fit/predict contract, the
report writer, and scripts/evaluate.py on a small synthetic corpus + splits.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from skolfie.eval.report import EVAL_SPLITS, compare_reports, stratified_sample
from skolfie.models.baselines import (
    AIDetectorAdapter,
    JoblibScorer,
    LengthBaseline,
    TfidfLRBaseline,
    make_scorer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_HUMAN_WORDS = ("honestly i think we went there and it was kinda fun but the bus was late "
                "lol my friend said nope so we just walked home and grabbed pizza").split()
_AI_WORDS = ("furthermore it is crucial to consider the multifaceted landscape of innovative "
             "solutions additionally this comprehensive approach fosters robust outcomes "
             "ultimately leveraging key insights enhances overall efficiency").split()


def _text(rng: np.random.Generator, words: list[str], n_sentences: int, ai: bool) -> str:
    sentences = []
    for _ in range(n_sentences):
        n = int(rng.integers(12, 16)) if ai else int(rng.integers(3, 20))
        sent = " ".join(rng.choice(words, size=n))
        sentences.append(sent.capitalize() + ("." if ai else rng.choice([".", "!", "?", "..."])))
    return " ".join(sentences)


def make_corpus(n_per_cell: int = 12, seed: int = 0) -> pd.DataFrame:
    """Small corpus in the skolfie schema: 2 domains x 3 generators (+1 OOD domain)."""
    rng = np.random.default_rng(seed)
    rows = []
    cells = [(dom, gen) for dom in ("news", "casual", "academic")
             for gen in ("human", "gpt-a", "gpt-b")]
    for dom, gen in cells:
        for i in range(n_per_cell):
            ai = gen != "human"
            n_sent = 1 if i % 4 == 0 else int(rng.integers(3, 7))  # every 4th text is short
            text = _text(rng, _AI_WORDS if ai else _HUMAN_WORDS, n_sent, ai)
            rows.append({
                "id": f"{'s1' if i % 2 else 's2'}:{dom}-{gen}-{i}",
                "text": text,
                "label": "ai" if ai else "human",
                "generator": gen,
                "domain": dom,
                "subdomain": dom,
                "source": "s1" if i % 2 else "s2",
                "split_hint": "",
                "n_words": len(text.split()),
            })
    return pd.DataFrame(rows).sort_values("id", ignore_index=True)


def make_splits(corpus: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One row per (id, split); test_short overlaps test, test_ood_generator is AI-only."""
    rng = np.random.default_rng(seed)
    ood_dom = corpus["domain"] == "academic"
    ood_gen = (corpus["generator"] == "gpt-b") & ~ood_dom
    pool = corpus[~ood_dom & ~ood_gen]
    bucket = rng.choice(["train", "train", "train", "val", "test"], size=len(pool))
    rows = [(i, s) for i, s in zip(pool["id"], bucket)]
    rows += [(i, "test_ood_generator") for i in corpus.loc[ood_gen, "id"]]
    rows += [(i, "test_ood_domain") for i in corpus.loc[ood_dom, "id"]]
    test_ids = set(pool["id"][bucket == "test"]) | set(corpus.loc[ood_dom | ood_gen, "id"])
    rows += [(i, "test_short") for i, n in zip(corpus["id"], corpus["n_words"])
             if i in test_ids and n < 25]
    return pd.DataFrame(rows, columns=["id", "split"])


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return make_corpus()


@pytest.fixture(scope="module")
def data_files(tmp_path_factory: pytest.TempPathFactory, corpus: pd.DataFrame) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("evaldata")
    corpus_path = root / "corpus.parquet"
    splits_path = root / "splits.parquet"
    corpus.to_parquet(corpus_path, index=False)
    splits = make_splits(corpus)
    assert set(splits["split"]) == {"train", *EVAL_SPLITS}
    splits.to_parquet(splits_path, index=False)
    (root / "splits_manifest.json").write_text(
        json.dumps({"splits_version": "s-test123", "corpus_version": "v-test456"}))
    return {"corpus": corpus_path, "splits": splits_path, "root": root}


@pytest.fixture(scope="module")
def evaluate():
    spec = importlib.util.spec_from_file_location("skolfie_evaluate_script",
                                                  REPO_ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> dict:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _check_proba(p: np.ndarray, n: int) -> None:
    assert isinstance(p, np.ndarray)
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))


@pytest.mark.parametrize("scorer", [
    LengthBaseline(),
    TfidfLRBaseline(word_max_features=500, char_max_features=500, vocab_sample=60,
                    chunk_size=25, n_jobs=1),
])
def test_baseline_fit_predict(scorer, corpus: pd.DataFrame) -> None:
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    assert scorer.fit(texts[::2], labels[::2]) is scorer
    p = scorer.predict_proba(texts[1::2])
    _check_proba(p, len(texts[1::2]))
    y = np.array(labels[1::2]) == "ai"
    assert p[y].mean() > p[~y].mean()  # the synthetic classes are easy
    assert scorer.predict_proba([]).shape == (0,)


def test_tfidf_chunked_transform_matches_single_chunk(corpus: pd.DataFrame) -> None:
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    a = TfidfLRBaseline(chunk_size=10_000).fit(texts, labels)
    b = TfidfLRBaseline(chunk_size=7, n_jobs=1).fit(texts, labels)
    c = TfidfLRBaseline(chunk_size=40, n_jobs=2).fit(texts, labels)  # 3 chunks, 2 workers
    xa, xb = a._transform(texts), b._transform(texts)
    assert xa.dtype == np.float32 and xa.shape == xb.shape
    assert abs(xa - xb).max() == 0
    assert abs(xa - c._transform(texts)).max() == 0
    np.testing.assert_allclose(a.predict_proba(texts), b.predict_proba(texts), rtol=1e-6)


def test_ensemble_adapter_and_joblib(tmp_path: Path) -> None:
    from tests.fixtures.seed_texts import AI_TEXTS, HUMAN_TEXTS

    texts = list(HUMAN_TEXTS[:12]) + list(AI_TEXTS[:12])
    labels = [0] * 12 + [1] * 12
    adapter = AIDetectorAdapter().fit(texts, labels)
    assert not adapter.pretrained
    held_out = list(HUMAN_TEXTS[12:15]) + list(AI_TEXTS[12:15])
    p = adapter.predict_proba(held_out)
    _check_proba(p, 6)

    # A saved AIDetectorModel loads via ensemble:PATH and via joblib:PATH.
    path = tmp_path / "detector.joblib"
    adapter.model.save(str(path))
    for spec in (f"ensemble:{path}", f"joblib:{path}"):
        loaded = make_scorer(spec)
        assert loaded.pretrained
        np.testing.assert_allclose(loaded.fit([], []).predict_proba(held_out), p)


def test_joblib_scorer_sklearn_pipeline(tmp_path: Path, corpus: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, labels)
    assert list(pipe.classes_) == ["ai", "human"]  # (n, 2) output, "ai" is column 0
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    scorer = JoblibScorer(path)
    assert scorer.name == "joblib:pipe"
    p = scorer.predict_proba(texts)
    _check_proba(p, len(texts))
    np.testing.assert_allclose(p, pipe.predict_proba(texts)[:, 0])

    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    with pytest.raises(TypeError):
        JoblibScorer(tmp_path / "bad.joblib")


def test_stratified_sample_keeps_label_mix(corpus: pd.DataFrame) -> None:
    df = corpus.set_index(pd.Index(corpus["id"].to_numpy()))
    out = stratified_sample(df, 30, seed=1)
    assert len(out) == 30
    share = (df["label"] == "ai").mean()
    assert (out["label"] == "ai").mean() == pytest.approx(share, abs=1 / 30)
    assert out.index.equals(stratified_sample(df, 30, seed=1).index)
    assert not out.index.equals(stratified_sample(df, 30, seed=2).index)
    assert stratified_sample(df, None, seed=0) is df
    tiny = stratified_sample(df, 2, seed=0)
    assert sorted(tiny["label"]) == ["ai", "human"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run(evaluate, data_files: dict[str, Path], model: str, out: Path, *extra: str) -> int:
    return evaluate.main(["--corpus", str(data_files["corpus"]),
                          "--splits", str(data_files["splits"]),
                          "--model", model, "--out-dir", str(out), *extra])


@pytest.mark.parametrize("model", ["length", "tfidf"])
def test_cli_writes_report(evaluate, data_files, tmp_path: Path, capsys, model: str) -> None:
    out = tmp_path / "reports" / model
    assert _run(evaluate, data_files, model, out, "--save-predictions") == 0
    printed = capsys.readouterr().out
    assert f"# Report card: {model}" in printed

    report = _strict_json(out / "report.json")
    assert report["model"]["name"] == model
    assert report["corpus"]["corpus_version"] == "v-test456"
    assert report["splits"]["splits_version"] == "s-test123"
    assert report["train"]["n"] > 0 and report["train"]["fit_seconds"] >= 0
    assert set(report["git"]) == {"commit", "dirty"}
    assert list(report["results"]) == list(EVAL_SPLITS)
    for split, res in report["results"].items():
        m = res["metrics"]
        assert m["n"] == m["n_ai"] + m["n_human"] > 0
        assert res["predict_seconds"] >= 0
        assert set(res["breakdown"]) == {"generator", "domain", "source"}
        if m["n_human"] and m["n_ai"]:
            assert 0.0 <= m["auroc"] <= 1.0
            assert 0.0 <= m["fpr_at_95tpr"] <= 1.0
        assert 0.0 <= m["ece"] <= 1.0
    # The AI-only OOD-generator split: AUROC/FPR are null in JSON, with a note.
    ood = report["results"]["test_ood_generator"]["metrics"]
    assert ood["n_human"] == 0 and ood["auroc"] is None and ood["notes"]
    if model == "tfidf":
        assert report["results"]["test"]["metrics"]["auroc"] > 0.9

    md = (out / "report.md").read_text()
    assert "## Headline" in md and "| test_short |" in md
    assert "FPR@95%TPR" in md and "### by generator" in md

    preds = pd.read_parquet(out / "predictions.parquet")
    assert list(preds.columns) == ["id", "split", "label", "p_ai"]
    assert set(preds["split"]) == set(EVAL_SPLITS)
    assert preds["p_ai"].between(0, 1).all()
    assert len(preds) == sum(r["metrics"]["n"] for r in report["results"].values())


def test_cli_max_train_eval_and_compare(evaluate, data_files, tmp_path: Path, capsys) -> None:
    out_len = tmp_path / "length"
    out_tfidf = tmp_path / "tfidf"
    assert _run(evaluate, data_files, "length", out_len, "--max-eval", "6",
                "--max-train", "20") == 0
    assert _run(evaluate, data_files, "tfidf", out_tfidf, "--max-eval", "6") == 0
    assert not (out_len / "predictions.parquet").exists()
    small = _strict_json(out_len / "report.json")
    assert small["train"]["n"] == 20
    assert all(r["metrics"]["n"] <= 6 for r in small["results"].values())
    assert small["sampling"] == {"seed": 0, "max_train": 20, "max_eval": 6}
    capsys.readouterr()

    assert evaluate.main(["--compare", str(tmp_path / "*" / "report.json")]) == 0
    table = capsys.readouterr().out
    assert "| split | model | n | AUROC | F1 | FPR@95%TPR | ECE |" in table
    for split in EVAL_SPLITS:
        assert f"| {split} | length |" in table and f"| {split} | tfidf |" in table
    assert "**" in table
    assert table == compare_reports([out_len / "report.json", out_tfidf / "report.json"])


def test_cli_usage_errors(evaluate, data_files, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        evaluate.main(["--model", "length"])  # no corpus/splits
    with pytest.raises(SystemExit):
        _run(evaluate, data_files, "bogus", tmp_path / "x")
    with pytest.raises(SystemExit):
        evaluate.main(["--compare", str(tmp_path / "missing.json")])


def test_cli_joblib_model_is_not_retrained(evaluate, data_files, corpus, tmp_path: Path,
                                           capsys) -> None:
    model = LengthBaseline().fit(corpus["text"].tolist(), corpus["label"].tolist())
    path = tmp_path / "length_model.joblib"
    joblib.dump(model, path)
    out = tmp_path / "joblib-report"
    assert _run(evaluate, data_files, f"joblib:{path}", out) == 0
    report = _strict_json(out / "report.json")
    assert report["model"]["name"] == "length"
    assert report["model"]["spec"] == f"joblib:{path}"
    assert report["train"] == {"pretrained": True, "n": 0, "n_human": 0, "n_ai": 0,
                               "n_available": 0, "fit_seconds": 0.0}
    assert "pretrained (not trained here)" in capsys.readouterr().out
