"""
tests/test_review_models.py
----------------------------
Regression tests for the robustness review of the models, features and
evaluation code: the feature cache, label handling in the metrics, the
--extra-train leakage filter, report writing and comparison, StackedModel
member loading, and odd text inputs to the feature models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.eval.metrics import binary_metrics, to_binary
from skolfie.eval.report import (
    _filter_extra_train,
    atomic_path,
    compare_reports,
    render_report_md,
    write_report,
)
from skolfie.features.cache import FeatureCache, text_key

NAMES = ["a", "b"]


def _fn(texts: list[str]) -> np.ndarray:
    return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def _quiet(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# FeatureCache
# --------------------------------------------------------------------------- #
def test_cache_ignores_a_corrupt_file_and_rebuilds_it(tmp_path: Path) -> None:
    (tmp_path / "x.parquet").write_bytes(b"PAR1 truncated by a dropped Colab session")
    logs: list[str] = []
    cache = FeatureCache(tmp_path, "x", NAMES, log=logs.append)
    assert len(cache) == 0 and any("unreadable" in m for m in logs)
    np.testing.assert_array_equal(cache.compute(["hey"], _fn), [[3, 1]])
    assert len(FeatureCache(tmp_path, "x", NAMES, log=_quiet)) == 1


def test_cache_file_with_duplicate_keys_keeps_rows_aligned(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"key": [text_key("aa"), text_key("aa"), text_key("c")],
                             "a": [7.0, 8.0, 9.0], "b": [1.0, 1.0, 1.0]}),
                   tmp_path / "dup.parquet")
    cache = FeatureCache(tmp_path, "dup", NAMES, log=_quiet)
    out = cache.compute(["aa", "c", "bbbbb"], _fn)
    np.testing.assert_array_equal(out, [[7, 1], [9, 1], [5, 1]])
    reloaded = FeatureCache(tmp_path, "dup", NAMES, log=_quiet)
    np.testing.assert_array_equal(reloaded.compute(["bbbbb", "c"], _fn), [[5, 1], [9, 1]])


def test_cache_in_an_unwritable_dir_falls_back_to_memory(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    logs: list[str] = []
    cache = FeatureCache(blocker / "cache", "y", NAMES, log=logs.append)
    np.testing.assert_array_equal(cache.compute(["hi", "hey"], _fn), [[2, 1], [3, 1]])
    assert cache.path is None and any("in memory only" in m for m in logs)
    np.testing.assert_array_equal(cache.compute(["hey"], _fn), [[3, 1]])


def test_cache_flush_is_atomic_and_leaves_no_temp_files(tmp_path: Path, monkeypatch) -> None:
    cache = FeatureCache(tmp_path, "z", NAMES, log=_quiet)
    cache.compute(["one"], _fn)
    before = (tmp_path / "z.parquet").read_bytes()

    import pyarrow.parquet as pq

    def crash(table, where, *args, **kwargs):
        Path(where).write_bytes(b"half")
        raise KeyboardInterrupt

    monkeypatch.setattr(pq, "write_table", crash)
    with pytest.raises(KeyboardInterrupt):
        cache.compute(["two"], _fn)
    assert (tmp_path / "z.parquet").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["z.parquet"]


def test_text_key_survives_lone_surrogates() -> None:
    assert text_key("a\ud800b") != text_key("ab")
    assert text_key("héllo") == text_key("héllo".replace("é", "é"))


# --------------------------------------------------------------------------- #
# Metrics: labels must never be truncated into a valid class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.7, float("nan"), None, 2, "maybe"])
def test_to_binary_rejects_non_binary_labels(bad) -> None:
    with pytest.raises(ValueError):
        to_binary([bad])


def test_to_binary_accepts_numeric_and_bool_labels() -> None:
    np.testing.assert_array_equal(to_binary([1.0, 0, True, np.int64(1), np.float32(0)]),
                                  [1, 0, 1, 1, 0])


@pytest.mark.parametrize("y", [[0.7, 1.0], [np.nan, 1.0], np.array([0.5, 1.0])])
def test_binary_metrics_rejects_fractional_labels(y) -> None:
    with pytest.raises(ValueError, match="only 0"):
        binary_metrics(y, [0.1, 0.9])


def test_binary_metrics_float_labels_still_work() -> None:
    assert binary_metrics([0.0, 1.0], [0.1, 0.9])["auroc"] == 1.0


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
    kept, _ = _filter_extra_train(extra, corpus, splits, _quiet)
    assert kept["orig_id"].tolist() == ["tr1", "unknown"]


def test_extra_train_orig_id_matches_across_dtypes() -> None:
    corpus = pd.DataFrame({"id": [1, 2], "text": ["a b", "c d"], "label": ["ai", "human"]},
                          index=[1, 2])
    splits = pd.DataFrame({"id": [1, 2], "split": ["test", "train"]})
    extra = pd.DataFrame({"text": ["z", "y"], "label": ["ai", "ai"], "orig_id": ["1", "2"]})
    kept, _ = _filter_extra_train(extra, corpus, splits, _quiet)
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
    from tests.test_stacker import _load_script

    evaluate = _load_script("evaluate")
    bad = tmp_path / "report.json"
    bad.write_text("{not json")
    with pytest.raises(SystemExit):
        evaluate.main(["--compare", str(bad)])


# --------------------------------------------------------------------------- #
# StackedModel and stack.py
# --------------------------------------------------------------------------- #
def test_stacked_model_names_the_missing_member() -> None:
    pytest.importorskip("sklearn")
    from skolfie.models.stacker import StackedModel, Stacker

    stacker = Stacker().fit(np.array([[0.1], [0.9], [0.2], [0.8]]), [0, 1, 0, 1])
    model = StackedModel({"features": "/content/drive/MyDrive/skolfie/models/f.joblib"},
                         stacker, {"ai_threshold": 0.5, "human_threshold": 0.5})
    with pytest.raises(FileNotFoundError, match="member 'features'.*model.members"):
        model.predict_proba(["text"])


def test_stack_cutoffs_come_from_out_of_fold_val_scores(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from tests.test_evaluate import make_corpus, make_splits
    from tests.test_stacker import _load_script

    corpus = make_corpus(n_per_cell=16, seed=5)
    corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
    make_splits(corpus, seed=5).to_parquet(tmp_path / "splits.parquet", index=False)
    evaluate = _load_script("evaluate")
    base = ["--corpus", str(tmp_path / "corpus.parquet"),
            "--splits", str(tmp_path / "splits.parquet")]
    assert evaluate.main(base + ["--model", "length", "--out-dir", str(tmp_path / "length"),
                                 "--save-predictions"]) == 0
    stack = _load_script("stack")
    assert stack.main(base + ["--member", f"length={tmp_path / 'length' / 'predictions.parquet'}",
                              "--out-dir", str(tmp_path / "stacked"), "--save-predictions",
                              "--max-fpr", "0.2"]) == 0
    report = json.loads((tmp_path / "stacked" / "report.json").read_text())
    preds = pd.read_parquet(tmp_path / "stacked" / "predictions.parquet")
    val = preds[preds["split"] == "val"]
    cut = report["verdicts"]["cutoffs"]
    human = val.loc[val["label"] == "human", "p_ai"].to_numpy()
    # The val rows are the out-of-fold scores the cut-offs were set on, so the
    # AI cut-off is one of the human val scores and honours max_fpr on them.
    assert np.isclose(human, cut["ai_threshold"]).any()
    assert (human > cut["ai_threshold"]).mean() <= 0.2


# --------------------------------------------------------------------------- #
# Feature model: odd inputs and small training sets
# --------------------------------------------------------------------------- #
_HUMAN = ["I went to the shop today and it was fine, honestly.",
          "We don't really know what happened lol."]
_AI = ["Moreover, it is important to note that the results are significant.",
       "In conclusion, the analysis demonstrates robust outcomes overall."]


def test_feature_model_small_train_set_and_missing_texts() -> None:
    pytest.importorskip("lightgbm")
    from skolfie.models.feature_model import FeatureModel

    texts = (_HUMAN + _AI) * 3 + [None]
    labels = ["human", "human", "ai", "ai"] * 3 + ["human"]
    model = FeatureModel(use_lm=False, n_jobs=1, log=_quiet).fit(texts, labels)
    p = model.predict_proba([None, float("nan"), "", "Moreover, overall it is crucial."])
    assert p.shape == (4,) and np.isfinite(p).all()
    assert p[0] == p[1] == p[2]


def test_style_features_treat_nan_as_empty() -> None:
    from skolfie.features.style import style_features

    np.testing.assert_array_equal(style_features(float("nan")), style_features(""))


def test_lm_features_handle_surrogates_and_nan() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from skolfie.features.lm import LMFeaturizer

    eos = "<|endoftext|>"
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(_HUMAN + _AI, trainers.BpeTrainer(
        vocab_size=300, special_tokens=[eos], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=eos, eos_token=eos)
    eos_id = fast.convert_tokens_to_ids(eos)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=len(fast),
                                       n_positions=64, bos_token_id=eos_id,
                                       eos_token_id=eos_id)).eval()
    lm = LMFeaturizer(model=model, tokenizer=fast, device="cpu")
    out = lm.features(["a\ud800b", "a?b", float("nan"), None, "​‏😀 Привет."])
    np.testing.assert_allclose(out[0], out[1])
    assert np.isnan(out[2]).all() and np.isnan(out[3]).all()
    assert np.isfinite(out[4]).all()


def test_transformer_unpack_rejects_tokenizer_paths_outside_its_temp_dir(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from skolfie.models.transformer_model import TransformerModel

    model = TransformerModel(use_features=False)
    model._packed = {"model_type": "distilbert",
                     "config": {"vocab_size": 10, "dim": 8, "hidden_dim": 16, "n_layers": 1,
                                "n_heads": 2, "max_position_embeddings": 16},
                     "state": {}, "tokenizer_files": {"../escape.txt": b"x"}}
    with pytest.raises(ValueError, match="bad tokenizer file name"):
        model.predict_proba(["text"])


def test_transformer_old_pickles_get_chunk_defaults() -> None:
    from skolfie.models.transformer_model import TransformerModel

    state = TransformerModel(use_features=False).__dict__.copy()
    for key in ("chunk_augment", "chunk_words", "n_chunks_"):
        del state[key]
    model = TransformerModel.__new__(TransformerModel)
    model.__setstate__(state)
    assert (model.chunk_augment, model.chunk_words, model.n_chunks_) == (0, 60, 0)
    assert "chunk_augment" not in model.params()
