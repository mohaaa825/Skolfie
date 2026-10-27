"""
tests/test_transformer_model.py
--------------------------------
Roadmap step 5: the DistilBERT + features TransformerModel. huggingface.co
may be unreachable, so a tiny DistilBERT and a WordPiece tokenizer trained
on the synthetic corpus are built locally and injected; the features path
uses the style features only (use_lm=False), so GPT-2 is never needed.
"""

from __future__ import annotations

import math

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("sklearn")

from skolfie.models.feature_model import FeatureModel  # noqa: E402
from skolfie.models.transformer_model import (  # noqa: E402
    FEATURE_CLIP,
    TransformerModel,
    fit_standardizer,
    holdout_split,
    length_batches,
    standardize,
)
from tests.test_evaluate import make_corpus  # noqa: E402

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
# Small, fast and with a high learning rate so a few epochs separate the classes.
FAST = dict(epochs=4, batch_size=8, grad_accum=2, lr=2e-3, head_lr=1e-2, max_length=64,
            warmup_ratio=0.1, eval_fraction=0.1, eval_every=3, device="cpu", n_jobs=1, seed=0)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(n_per_cell=12, seed=0), make_corpus(n_per_cell=6, seed=1)


@pytest.fixture(scope="module")
def tokenizer(corpus):
    from tokenizers import (
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    train, test = corpus
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=300, special_tokens=SPECIAL_TOKENS,
                                        show_progress=False)
    tok.train_from_iterator(train["text"].tolist() + test["text"].tolist(), trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[(t, tok.token_to_id(t)) for t in ("[CLS]", "[SEP]")])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def new_backbone(tokenizer, seed: int = 0):
    from transformers import DistilBertConfig, DistilBertModel

    torch.manual_seed(seed)
    config = DistilBertConfig(vocab_size=len(tokenizer), dim=32, hidden_dim=64, n_layers=1,
                              n_heads=2, max_position_embeddings=128,
                              pad_token_id=tokenizer.pad_token_id)
    return DistilBertModel(config)


def make(tokenizer, log=None, **kwargs) -> TransformerModel:
    params = {**FAST, "use_lm": False, **kwargs}
    return TransformerModel(backbone=new_backbone(tokenizer), tokenizer=tokenizer,
                            log=log or (lambda msg: None), **params)


@pytest.fixture(scope="module")
def fitted(tokenizer, corpus):
    train, _ = corpus
    logs: list[str] = []
    model = make(tokenizer, log=logs.append).fit(train["text"].tolist(), train["label"].tolist())
    return model, logs


@pytest.fixture(scope="module")
def fitted_text(tokenizer, corpus):
    train, _ = corpus
    # Text only learns more slowly than with the style features.
    return make(tokenizer, use_features=False, epochs=6, lr=5e-3).fit(train["text"],
                                                                        train["label"])


def separation(model: TransformerModel, df) -> tuple[float, float]:
    p = model.predict_proba(df["text"].tolist())
    ai = (df["label"] == "ai").to_numpy()
    return float(p[ai].mean()), float(p[~ai].mean())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_standardizer_handles_nan_inf_and_constant_columns() -> None:
    X = np.array([[1.0, np.nan, 5.0, 1.0],
                  [2.0, np.nan, 5.0, np.nan],
                  [3.0, np.nan, 5.0, np.inf],
                  [1e9, np.nan, 5.0, 3.0]])
    stats = fit_standardizer(X)
    assert np.isfinite(stats["median"]).all() and (stats["std"] >= 1e-6).all()
    assert stats["median"][1] == 0.0 and stats["median"][3] == 2.0  # median of 1, 3
    Z = standardize(X, stats)
    assert Z.dtype == np.float32 and np.isfinite(Z).all()
    assert np.abs(Z).max() <= FEATURE_CLIP
    assert (Z[:, 1] == 0).all() and (Z[:, 2] == 0).all()  # all-NaN and constant columns
    # NaN and inf are filled with the train median before scaling.
    assert Z[1, 3] == Z[2, 3] == pytest.approx((2.0 - stats["mean"][3]) / stats["std"][3])
    # New data with NaN everywhere maps to the medians' z-scores.
    assert np.isfinite(standardize(np.full((2, 4), np.nan), stats)).all()


def test_holdout_split_is_stratified_and_seeded() -> None:
    y = np.array([0] * 30 + [1] * 70)
    tr, ev = holdout_split(y, 0.1, seed=0)
    assert np.intersect1d(tr, ev).size == 0 and tr.size + ev.size == y.size
    assert (y[ev] == 0).sum() == 3 and (y[ev] == 1).sum() == 7
    assert np.array_equal(ev, holdout_split(y, 0.1, seed=0)[1])
    assert not np.array_equal(ev, holdout_split(y, 0.1, seed=1)[1])
    # At least one per class when possible, never the last one of a class.
    _, ev = holdout_split(np.array([0, 0, 1, 1, 1]), 0.01, seed=0)
    assert sorted(np.array([0, 0, 1, 1, 1])[ev]) == [0, 1]
    _, ev = holdout_split(np.array([0, 1, 1, 1]), 0.3, seed=0)
    assert (np.array([0, 1, 1, 1])[ev] == 1).all()
    assert holdout_split(y, 0.0, seed=0)[1].size == 0


def test_length_batches_cover_everything_and_cut_padding() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 1000, size=103)
    batches = length_batches(lengths, 8, np.random.default_rng(0), chunk_batches=4)
    assert sorted(np.concatenate(batches).tolist()) == list(range(103))
    assert max(len(b) for b in batches) == 8
    for b in batches:  # sorted within each batch
        assert np.all(np.diff(lengths[b]) >= 0)


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #
def test_fit_predict_shape_range_and_empty(fitted, corpus) -> None:
    model, _ = fitted
    _, test = corpus
    p = model.predict_proba(test["text"].tolist())
    assert model.name == "transformer" and model.pretrained is False
    assert p.shape == (len(test),) and p.dtype == np.float64
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    empty = model.predict_proba([])
    assert empty.shape == (0,)
    assert model.predict_proba(["", "   "]).shape == (2,)
    assert len(model.feature_names_) > 0
    assert all(n.startswith("sty_") for n in model.feature_names_)


def test_it_learns(fitted, fitted_text, corpus) -> None:
    _, test = corpus
    for model in (fitted[0], fitted_text):
        p_ai, p_human = separation(model, test)
        assert p_ai - p_human > 0.4, (model.name, p_ai, p_human)


def test_text_only_path(fitted_text) -> None:
    model = fitted_text
    assert model.name == "transformer-text"
    assert model.feature_names_ == [] and model._features is None
    assert "features" not in model._net
    assert model.params()["n_features"] == 0 and "use_lm" not in model.params()


def test_predictions_do_not_depend_on_order_or_batch_size(fitted, corpus, monkeypatch) -> None:
    model, _ = fitted
    _, test = corpus
    texts = test["text"].tolist()
    p = model.predict_proba(texts)
    perm = np.random.default_rng(0).permutation(len(texts))
    np.testing.assert_allclose(model.predict_proba([texts[i] for i in perm]), p[perm], atol=1e-5)
    monkeypatch.setattr(model, "batch_size", 1)
    np.testing.assert_allclose(model.predict_proba(texts), p, atol=1e-5)
    np.testing.assert_allclose(model.predict_proba(texts[3:4]), p[3:4], atol=1e-5)


def test_best_checkpoint_is_reported(fitted, corpus) -> None:
    from sklearn.metrics import log_loss

    model, logs = fitted
    train, _ = corpus
    params = model.params()
    assert 1 <= params["best_step"] <= params["total_steps"]
    assert len(model.history_) >= model.epochs
    assert params["best_eval_logloss"] == min(h["logloss"] for h in model.history_)
    assert 0 <= params["best_eval_auroc"] <= 1
    # The final weights are the best ones: re-scoring the held-out texts
    # reproduces the best held-out log-loss.
    y = (train["label"] == "ai").to_numpy().astype(int)
    _, eval_idx = holdout_split(y, FAST["eval_fraction"], FAST["seed"])
    assert params["n_eval"] == eval_idx.size and params["n_train"] == len(y) - eval_idx.size
    p = model.predict_proba(train["text"].iloc[eval_idx].tolist())
    assert log_loss(y[eval_idx], p, labels=[0, 1]) == pytest.approx(
        params["best_eval_logloss"], abs=1e-5)
    assert any("held-out log-loss" in m for m in logs)
    assert any("texts/s" in m and "ETA" in m for m in logs)


def test_best_checkpoint_is_restored(tokenizer, corpus, monkeypatch) -> None:
    import sklearn.metrics

    train, _ = corpus
    logs: list[str] = []
    model = make(tokenizer, log=logs.append, epochs=1, eval_every=2)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    evaluate = model._evaluate

    def spy(texts, y, feats, eval_idx, step, best_state):
        snapshots[step] = {k: v.detach().clone() for k, v in model._net.state_dict().items()}
        return evaluate(texts, y, feats, eval_idx, step, best_state)

    # The second evaluation is the best one by (fake) held-out log-loss.
    losses = iter([0.5, 0.1] + [0.3] * 100)
    monkeypatch.setattr(model, "_evaluate", spy)
    monkeypatch.setattr(sklearn.metrics, "log_loss", lambda *args, **kwargs: next(losses))
    model.fit(train["text"], train["label"])

    steps = sorted(snapshots)
    assert len(steps) >= 3 and steps[-1] == model.total_steps_
    assert model.best_step_ == steps[1] and model.best_eval_logloss_ == 0.1
    final = model._net.state_dict()
    for key, value in snapshots[steps[1]].items():
        torch.testing.assert_close(final[key], value, rtol=0, atol=0)
    assert any(not torch.equal(final[k], v) for k, v in snapshots[steps[-1]].items())
    assert any("restored the best checkpoint" in m for m in logs)


def test_params(fitted) -> None:
    model, _ = fitted
    params = model.params()
    for key in ("model_name", "use_features", "use_lm", "max_length", "epochs", "batch_size",
                "grad_accum", "lr", "head_lr", "best_eval_logloss", "best_eval_auroc",
                "best_step", "n_features", "train_seconds"):
        assert key in params
    assert params["n_features"] == len(model.feature_names_)
    assert params["train_seconds"] > 0
    assert params["total_steps"] == FAST["epochs"] * math.ceil(
        math.ceil(params["n_train"] / FAST["batch_size"]) / FAST["grad_accum"])


def test_joblib_round_trip_rebuilds_without_injected_objects(fitted, tokenizer, corpus,
                                                              tmp_path) -> None:
    model, _ = fitted
    _, test = corpus
    texts = test["text"].tolist()
    p = model.predict_proba(texts)
    path = tmp_path / "transformer.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.log is print
    assert loaded._net is None and loaded._tokenizer is None
    assert loaded._backbone_in is None and loaded._tokenizer_in is None
    assert loaded._features._lm is None
    np.testing.assert_allclose(loaded.predict_proba(texts), p, atol=1e-6)
    assert loaded._tokenizer is not tokenizer and loaded._packed is None
    # The original keeps working, and a rebuilt model pickles again.
    np.testing.assert_allclose(model.predict_proba(texts[:3]), p[:3], atol=1e-6)
    joblib.dump(loaded, tmp_path / "again.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "again.joblib").predict_proba(texts), p,
                               atol=1e-6)


def test_feature_columns_with_nan_are_handled(tokenizer, corpus, monkeypatch) -> None:
    original = FeatureModel.base_features

    def with_nan(self, texts):
        X, names = original(self, texts)
        extra = np.full((len(X), 2), np.nan, dtype=np.float32)
        extra[::2, 1] = 7.0  # half-missing column; the other one is all-NaN
        return np.hstack([X, extra]), names + ["nan_all", "nan_half"]

    monkeypatch.setattr(FeatureModel, "base_features", with_nan)
    train, test = corpus
    model = make(tokenizer, epochs=1).fit(train["text"], train["label"])
    assert model.feature_names_[-2:] == ["nan_all", "nan_half"]
    assert np.isfinite(model.feature_stats_["median"]).all()
    p = model.predict_proba(test["text"].tolist())
    assert np.isfinite(p).all() and p.shape == (len(test),)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_inject_both_or_neither(tokenizer) -> None:
    with pytest.raises(ValueError, match="both"):
        TransformerModel(backbone=new_backbone(tokenizer))
    with pytest.raises(ValueError, match="both"):
        TransformerModel(tokenizer=tokenizer)


def test_bad_labels_raise_before_anything_is_loaded() -> None:
    # No injected backbone and GPT-2 features on: anything but an early
    # ValueError would try to download a model.
    model = TransformerModel()
    with pytest.raises(ValueError, match="unknown label"):
        model.fit(["a", "b"], ["human", "robot"])
    with pytest.raises(ValueError, match="both human and ai"):
        model.fit(["a", "b", "c"], ["human"] * 3)
    with pytest.raises(ValueError, match="same length"):
        model.fit(["a", "b"], ["human"])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(["a"])


def test_bad_hyperparameters_raise() -> None:
    with pytest.raises(ValueError):
        TransformerModel(batch_size=0)
    with pytest.raises(ValueError):
        TransformerModel(eval_fraction=0.6)


def test_chunk_training_adds_labelled_chunks_of_training_texts_only(tokenizer, corpus,
                                                                    monkeypatch) -> None:
    train, test = corpus
    texts, labels = train["text"].tolist(), train["label"].tolist()
    seen = {}
    model = make(tokenizer, chunk_augment=2, chunk_words=8, epochs=1)
    original = model._train

    def spy(all_texts, y, feats, train_idx, eval_idx):
        seen.update(texts=all_texts, y=y, train_idx=train_idx, eval_idx=eval_idx)
        return original(all_texts, y, feats, train_idx, eval_idx)

    monkeypatch.setattr(model, "_train", spy)
    model.fit(texts, labels)
    assert model.name == "transformer-chunks"

    n = len(texts)
    added = seen["texts"][n:]
    assert len(added) == model.n_chunks_ > 0
    assert seen["train_idx"].size == n - seen["eval_idx"].size + len(added)
    assert set(range(n, n + len(added))) <= set(seen["train_idx"].tolist())
    held_out = {texts[i] for i in seen["eval_idx"]}
    y_text = dict(zip(texts, seen["y"][:n]))
    for chunk, label in zip(added, seen["y"][n:]):
        parents = [t for t in texts if chunk in t and t not in held_out]
        assert parents and any(y_text[t] == label for t in parents)
        assert len(chunk.split()) < max(len(t.split()) for t in parents)
    assert model.params()["n_chunks"] == len(added)
    assert model.predict_proba(test["text"].tolist()).shape == (len(test),)

    # Models saved before chunk training existed still load and report.
    state = model.__getstate__()
    for key in ("chunk_augment", "chunk_words", "n_chunks_"):
        state.pop(key)
    old = TransformerModel.__new__(TransformerModel)
    old.__setstate__(state)
    assert old.chunk_augment == 0 and "n_chunks" not in old.params()


def test_chunk_training_options_are_validated() -> None:
    with pytest.raises(ValueError):
        TransformerModel(chunk_augment=-1)
    with pytest.raises(ValueError):
        TransformerModel(chunk_words=0)
    assert TransformerModel().name == "transformer"
