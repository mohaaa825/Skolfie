"""
tests/test_evaluate.py
-----------------------
The evaluation harness end to end: baselines' fit/predict contract, the
report writer, and scripts/evaluate.py on a small synthetic corpus + splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from skolfie.eval.report import (
    EVAL_SPLITS,
    _filter_extra_train,
    atomic_path,
    compare_reports,
    render_report_md,
    stratified_sample,
    write_report,
)
from skolfie.models.baselines import (
    AIDetectorAdapter,
    JoblibScorer,
    LengthBaseline,
    TfidfLRBaseline,
    make_scorer,
)
from tests.conftest import load_script

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
    return load_script("evaluate")


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


def test_cli_feature_model_cache_and_saved_model(evaluate, data_files, tmp_path: Path,
                                                 capsys) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("skolfie.features.style")
    out = tmp_path / "features-nolm"
    cache = tmp_path / "cache"
    saved = tmp_path / "models" / "features.joblib"
    assert _run(evaluate, data_files, "features-nolm", out, "--cache-dir", str(cache),
                "--save-model", str(saved), "--n-jobs", "1") == 0
    printed = capsys.readouterr().out
    assert "## Feature importance" in printed
    report = _strict_json(out / "report.json")
    importance = report["model"]["feature_importance"]
    assert importance and abs(sum(r["importance"] for r in importance) - 1) < 1e-6
    assert report["model"]["params"]["use_lm"] is False
    assert (cache / "style-v1.parquet").is_file() and saved.is_file()

    # The saved model scores again without retraining and gives the same report numbers.
    again = tmp_path / "reloaded"
    assert _run(evaluate, data_files, f"joblib:{saved}", again) == 0
    reloaded = _strict_json(again / "report.json")
    assert reloaded["train"]["pretrained"] is True
    for split, res in report["results"].items():
        assert reloaded["results"][split]["metrics"]["auroc"] == res["metrics"]["auroc"]


# --------------------------------------------------------------------------- #
# --extra-train leakage filter
# --------------------------------------------------------------------------- #
@pytest.fixture()
def tiny_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = ["v1", "t1", "s1", "tr1"]
    corpus = pd.DataFrame({"id": ids, "label": ["ai", "human", "ai", "human"],
                           "text": ["Eval text one.", "A test text.", "Short one.",
                                    "A train text."]})
    corpus.index = pd.Index(ids)
    splits = pd.DataFrame({"id": ["v1", "t1", "t1", "s1", "tr1"],
                           "split": ["val", "test", "test_short", "test_custom", "train"]})
    return corpus, splits


def test_extra_train_drops_case_and_punctuation_variants_of_eval_texts(tiny_corpus) -> None:
    corpus, splits = tiny_corpus
    extra = pd.DataFrame({"text": ["EVAL text one!", "eval  text\none", "A train text.",
                                   "fresh"], "label": ["ai"] * 4})
    logs: list[str] = []
    kept, info = _filter_extra_train(extra, corpus, splits, logs.append)
    assert kept["text"].tolist() == ["A train text.", "fresh"]
    assert info["n_dropped_eval_overlap"] == 2
    assert any("no orig_id" in m for m in logs)


def test_extra_train_drops_rows_from_any_non_train_split(tiny_corpus) -> None:
    corpus, splits = tiny_corpus
    extra = pd.DataFrame({"text": ["x1", "x2", "x3", "x4"], "label": ["ai"] * 4,
                          "orig_id": ["s1", "t1", "tr1", "unknown"]})
    kept, _ = _filter_extra_train(extra, corpus, splits, lambda m: None)
    assert kept["orig_id"].tolist() == ["tr1", "unknown"]


def test_extra_train_orig_id_matches_across_dtypes() -> None:
    corpus = pd.DataFrame({"id": [1, 2], "text": ["a b", "c d"], "label": ["ai", "human"]},
                          index=[1, 2])
    splits = pd.DataFrame({"id": [1, 2], "split": ["test", "train"]})
    extra = pd.DataFrame({"text": ["z", "y"], "label": ["ai", "ai"], "orig_id": ["1", "2"]})
    kept, _ = _filter_extra_train(extra, corpus, splits, lambda m: None)
    assert kept["orig_id"].tolist() == ["2"]



# --------------------------------------------------------------------------- #
# Report writing, rendering and comparison
# --------------------------------------------------------------------------- #
def _report(group: str = "gpt") -> dict:
    return {
        "model": {"name": "m", "spec": "m", "class": "C"},
        "train": {"pretrained": True}, "corpus": {}, "splits": {},
        "results": {"test": {
            "metrics": {"n": 1, "n_ai": 1, "n_human": 0, "auroc": None, "notes": []},
            "breakdown": {"generator": [{
                "group": group, "n": 1, "n_ai": 1, "n_human": 0, "accuracy": 1.0,
                "recall_ai": 1.0, "fpr_human": None, "auroc": None}]}}},
    }


def test_atomic_path_keeps_the_old_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "model.joblib.gz"
    with atomic_path(target) as tmp:
        assert tmp.name.endswith("model.joblib.gz") and tmp.parent == target.parent
        tmp.write_text("v1")
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_text("half")
        raise RuntimeError("killed")
    assert target.read_text() == "v1"
    assert [p.name for p in target.parent.iterdir()] == ["model.joblib.gz"]


def test_write_report_files_are_complete(tmp_path: Path) -> None:
    preds = pd.DataFrame({"id": ["a"], "split": ["test"], "label": ["ai"], "p_ai": [0.9]})
    paths = write_report(_report(), tmp_path / "out", preds)
    assert json.loads(paths["json"].read_text())["model"]["name"] == "m"
    assert paths["md"].read_text().startswith("# Report card: m")
    assert len(pd.read_parquet(paths["predictions"])) == 1
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "predictions.parquet", "report.json", "report.md"]


def test_markdown_cells_cannot_break_the_table_or_inject_html() -> None:
    md = render_report_md(_report("gpt\n| x | y |\n<img src=x onerror=alert(1)>"))
    row = [line for line in md.splitlines() if line.startswith("| gpt")]
    assert len(row) == 1
    assert "<img" not in md and "&lt;img" in row[0] and "\\| x \\|" in row[0]


def test_compare_rejects_files_that_are_not_reports(tmp_path: Path) -> None:
    bad = tmp_path / "robust_report.json"
    bad.write_text(json.dumps({"rows": []}))
    with pytest.raises(ValueError, match="not a report.json"):
        compare_reports([bad])


def test_compare_handles_minimal_old_reports(tmp_path: Path) -> None:
    old = _report()
    for key in ("corpus", "splits", "train"):
        del old[key]
    path = tmp_path / "old" / "report.json"
    path.parent.mkdir()
    path.write_text(json.dumps(old))
    out = compare_reports([path, path])
    assert "| test | m (old) |" in out


def test_evaluate_compare_turns_bad_files_into_usage_errors(tmp_path: Path) -> None:
    evaluate = load_script("evaluate")
    bad = tmp_path / "report.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit):
        evaluate.main(["--compare", str(bad)])

