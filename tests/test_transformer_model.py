"""
tests/test_transformer_model.py
--------------------------------
Roadmap step 5: the DistilBERT + features TransformerModel. huggingface.co
may be unreachable, so a tiny DistilBERT and a WordPiece tokenizer trained
on the synthetic corpus are built locally and injected; the features path
uses the style features only (use_lm=False), so GPT-2 is never needed.
Large-backbone support is checked with a tiny random DeBERTa-v2 model
(precision resolution needs no GPU; a SentencePiece tokenizer is trained
locally when `sentencepiece` is installed).
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


def test_unicode_tricks_are_undone_in_fit_and_predict(fitted, tokenizer, corpus,
                                                       monkeypatch) -> None:
    from skolfie.robust.attacks import homoglyph_text, text_rng, zero_width_text

    def attack(text: str) -> str:
        glyph = homoglyph_text(text, 0.2, text_rng(0, "homoglyph", text))
        return zero_width_text(glyph, 0.2, text_rng(0, "zero_width", text))

    train, test = corpus
    model, _ = fitted
    clean = test["text"].tolist()
    attacked = [attack(t) for t in clean]
    assert attacked != clean
    np.testing.assert_allclose(model.predict_proba(attacked), model.predict_proba(clean),
                               atol=1e-6)

    # fit() trains on the normalised texts, chunks included.
    seen = {}
    chunked = make(tokenizer, chunk_augment=2, chunk_words=8, epochs=1)
    original = chunked._train

    def spy(all_texts, y, feats, train_idx, eval_idx):
        seen["texts"] = all_texts
        return original(all_texts, y, feats, train_idx, eval_idx)

    monkeypatch.setattr(chunked, "_train", spy)
    texts = train["text"].tolist()
    chunked.fit([attack(t) for t in texts], train["label"].tolist())
    assert seen["texts"][:len(texts)] == texts
    assert len(seen["texts"]) > len(texts) and all(t.isascii() for t in seen["texts"])


def test_chunk_training_options_are_validated() -> None:
    with pytest.raises(ValueError):
        TransformerModel(chunk_augment=-1)
    with pytest.raises(ValueError):
        TransformerModel(chunk_words=0)
    assert TransformerModel().name == "transformer"


def test_transformer_with_a_bino_feature_model_uses_base_features_only(tokenizer,
                                                                       corpus) -> None:
    """A FeatureModel with Binoculars on can serve as the transformer's feature extractor:
    `base_features` never computes (or returns) the bino_* columns."""

    class NoBino:
        cache_key = "never"

        def features(self, texts, progress=False):
            raise AssertionError("the transformer must not compute Binoculars features")

    train, test = corpus
    model = make(tokenizer, epochs=1)
    model._features = FeatureModel(use_lm=False, n_jobs=1, log=lambda m: None,
                                   bino_observer="Qwen/Qwen2.5-1.5B")
    model._features._bino = NoBino()
    model.fit(train["text"].tolist(), train["label"].tolist())
    assert not any(n.startswith("bino_") for n in model.feature_names_)
    assert model.predict_proba(test["text"].tolist()[:4]).shape == (4,)


# --------------------------------------------------------------------------- #
# Precision, large backbones (DeBERTa-v3), gradient checkpointing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("requested, device, bf16, expected", [
    ("auto", "cuda", True, "bf16"),     # A100, L4
    ("auto", "cuda", False, "fp16"),    # T4
    ("auto", "cpu", False, "fp32"),
    ("fp16", "cuda", True, "fp16"),
    ("bf16", "cuda", True, "bf16"),
    ("bf16", "cuda", False, "fp32"),    # no bf16 on this GPU: exact fallback
    ("fp32", "cuda", True, "fp32"),
    ("fp16", "cpu", False, "fp32"),
    ("bf16", "cpu", True, "fp32"),
])
def test_resolve_precision(requested, device, bf16, expected) -> None:
    from skolfie.models.transformer_model import resolve_precision

    logs: list[str] = []
    assert resolve_precision(requested, device, bf16, "distilbert-base-cased",
                             log=logs.append) == expected
    assert not any("DeBERTa" in m for m in logs)
    if requested == "bf16" and device == "cuda" and not bf16:
        assert any("does not support bf16" in m for m in logs)


def test_deberta_in_fp16_warns() -> None:
    from skolfie.models.transformer_model import resolve_precision

    logs: list[str] = []
    assert resolve_precision("auto", "cuda", False, "microsoft/deberta-v3-large",
                             log=logs.append) == "fp16"
    assert any("WARNING" in m and "DeBERTa" in m for m in logs)
    logs.clear()  # found by the backbone's model_type too, e.g. an injected one
    assert resolve_precision("fp16", "cuda", True, "x", "deberta-v2", log=logs.append) == "fp16"
    assert logs
    logs.clear()
    assert resolve_precision("auto", "cuda", True, "microsoft/deberta-v3-large",
                             log=logs.append) == "bf16"
    assert not logs


def test_requested_precision_and_the_legacy_fp16_flag() -> None:
    from skolfie.models.transformer_model import requested_precision

    assert requested_precision(None, None) == "auto"
    assert requested_precision(None, True) == "fp16"
    assert requested_precision(None, False) == "fp32"
    assert requested_precision("BF16", True) == "bf16"  # precision wins
    with pytest.raises(ValueError, match="precision"):
        requested_precision("int8", None)
    assert TransformerModel(fp16=True).precision == "fp16"
    assert TransformerModel(fp16=False).precision == "fp32"
    assert TransformerModel().precision == "auto"
    assert TransformerModel(precision="bf16", fp16=True).precision == "bf16"
    with pytest.raises(ValueError):
        TransformerModel(precision="half")


@pytest.mark.parametrize("fp16, expected", [(None, "auto"), (True, "fp16"), (False, "fp32")])
def test_models_pickled_before_precision_load_and_predict(fitted, corpus, fp16,
                                                          expected) -> None:
    model, _ = fitted
    _, test = corpus
    texts = test["text"].tolist()[:6]
    state = model.__getstate__()
    for key in ("precision", "gradient_checkpointing", "backbone_type_", "precision_used_",
                "_precision"):
        state.pop(key)
    state["fp16"] = fp16
    old = TransformerModel.__new__(TransformerModel)
    old.__setstate__(state)
    assert old.precision == expected and old.gradient_checkpointing is False
    assert old.backbone_type_ == "distilbert"
    np.testing.assert_allclose(old.predict_proba(texts), model.predict_proba(texts), atol=1e-6)
    assert old._precision == "fp32"  # CPU
    assert old.params()["precision"] == expected


def test_params_record_the_backbone_and_precision(fitted) -> None:
    model, _ = fitted
    params = model.params()
    assert model.name == "transformer"
    assert params["backbone"] == "distilbert"
    assert params["precision"] == "auto" and params["precision_used"] == "fp32"
    assert params["gradient_checkpointing"] is False


def test_gradient_checkpointing(tokenizer, corpus) -> None:
    train, test = corpus
    logs: list[str] = []
    model = make(tokenizer, log=logs.append, gradient_checkpointing=True, epochs=1)
    model.fit(train["text"], train["label"])
    assert model._net["backbone"].is_gradient_checkpointing
    assert any("gradient checkpointing on" in m for m in logs)
    assert model.params()["gradient_checkpointing"] is True
    assert np.isfinite(model.predict_proba(test["text"].tolist()[:4])).all()


def test_non_finite_mixed_precision_batch_is_recomputed_in_fp32(fitted, corpus,
                                                                monkeypatch) -> None:
    """An fp16 overflow at predict time (DeBERTa on a T4) is redone in float32."""
    import contextlib
    import warnings

    model, _ = fitted
    _, test = corpus
    texts = test["text"].tolist()[:5]
    expected = model.predict_proba(texts)
    forward = model._forward
    calls = []

    def overflowing(ids, mask, feats):
        calls.append(1)
        out = forward(ids, mask, feats)
        return out * float("nan") if len(calls) == 1 else out

    monkeypatch.setattr(model, "_forward", overflowing)
    monkeypatch.setattr(model, "_amp", True)
    monkeypatch.setattr(model, "_autocast", contextlib.nullcontext)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # autocast(cuda) on a CPU-only machine
        p = model.predict_proba(texts)
    assert len(calls) == 2
    np.testing.assert_allclose(p, expected, atol=1e-6)


def _tiny_deberta(tokenizer):
    from transformers import DebertaV2Config, DebertaV2Model

    torch.manual_seed(0)
    config = DebertaV2Config(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=64,
                             max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id,
                             relative_attention=True, position_buckets=16,
                             pos_att_type=["p2c", "c2p"], position_biased_input=False,
                             norm_rel_ebd="layer_norm", share_att_key=True)
    return DebertaV2Model(config)


def test_deberta_backbone_trains_pickles_and_reloads(tokenizer, corpus, tmp_path) -> None:
    """A (tiny, random) DeBERTa-v2/v3 backbone: first-token pooling, gradient checkpointing,
    the joblib round trip and the report name."""
    train, test = corpus
    model = TransformerModel(backbone=_tiny_deberta(tokenizer), tokenizer=tokenizer,
                             model_name="microsoft/deberta-v3-large", use_lm=False,
                             log=lambda m: None, gradient_checkpointing=True,
                             **{**FAST, "epochs": 2})
    model.fit(train["text"].tolist(), train["label"].tolist())
    assert model.name == "transformer"
    params = model.params()
    assert params["backbone"] == "deberta-v2"
    assert params["model_name"] == "microsoft/deberta-v3-large"
    texts = test["text"].tolist()
    p = model.predict_proba(texts)
    assert np.isfinite(p).all() and p.shape == (len(texts),)
    joblib.dump(model, tmp_path / "transformer.joblib")
    loaded = joblib.load(tmp_path / "transformer.joblib")
    np.testing.assert_allclose(loaded.predict_proba(texts), p, atol=1e-6)
    assert type(loaded._net["backbone"]).__name__ == "DebertaV2Model"


def test_sentencepiece_deberta_tokenizer_round_trip(corpus, tmp_path) -> None:
    """The DeBERTa-v3 tokenizer is a SentencePiece model: loading it needs sentencepiece;
    the pickled model keeps the converted tokenizer and reloads without the .model file."""
    spm = pytest.importorskip("sentencepiece")
    pytest.importorskip("google.protobuf")
    import io

    from transformers import DebertaV2Tokenizer

    train, test = corpus
    texts = train["text"].tolist() + test["text"].tolist()
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(texts), model_writer=buf, vocab_size=200, hard_vocab_limit=False,
        model_type="unigram", pad_id=0, bos_id=1, eos_id=2, unk_id=3, pad_piece="[PAD]",
        bos_piece="[CLS]", eos_piece="[SEP]", unk_piece="[UNK]", user_defined_symbols=["[MASK]"],
        minloglevel=2)
    spm_dir = tmp_path / "spm"
    spm_dir.mkdir()
    (spm_dir / "spm.model").write_bytes(buf.getvalue())
    tok = DebertaV2Tokenizer.from_pretrained(spm_dir)
    ids = tok(texts[:1])["input_ids"][0]
    assert ids[0] == tok.cls_token_id and tok.unk_token_id not in ids[1:-1]

    model = TransformerModel(backbone=_tiny_deberta(tok), tokenizer=tok, use_lm=False,
                             log=lambda m: None, **{**FAST, "epochs": 1})
    model.fit(train["text"].tolist(), train["label"].tolist())
    p = model.predict_proba(test["text"].tolist())
    joblib.dump(model, tmp_path / "m.joblib")
    (spm_dir / "spm.model").unlink()
    loaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(loaded.predict_proba(test["text"].tolist()), p, atol=1e-6)


def test_make_scorer_passes_transformer_options_through() -> None:
    from skolfie.models.baselines import make_scorer

    opts = dict(model_name="microsoft/deberta-v3-large", precision="bf16",
                gradient_checkpointing=True, lr=1e-5, grad_accum=4, batch_size=8,
                max_length=512, epochs=1)
    model = make_scorer("transformer", seed=3, **opts)
    for key, value in opts.items():
        assert getattr(model, key) == value, key
    assert model.seed == 3 and model.name == "transformer"
    # The feature models ignore them (and None values are dropped).
    features = make_scorer("features-nolm", **opts, cache_dir=None)
    assert type(features).__name__ == "FeatureModel"
    assert make_scorer("transformer", precision=None).precision == "auto"


def test_evaluate_cli_transformer_options_and_kept_sources(tokenizer, corpus, tmp_path,
                                                           monkeypatch) -> None:
    """evaluate.py hands --backbone/--precision/--lr/--grad-accum/--gradient-checkpointing
    to the transformer, and --max-train keeps every --keep-sources row for it too."""
    import json

    import pandas as pd

    import skolfie.models.baselines as baselines
    from tests.test_modern_data import _script

    train, _ = corpus
    df = train.copy()
    pos = np.arange(len(df))
    df["source"] = np.where(pos % 5 == 0, "persuade", "hc3")
    df.loc[((df["label"] == "ai") & (pos % 3 == 0)).to_numpy(), "source"] = "fresh"
    df.to_parquet(tmp_path / "corpus.parquet", index=False)
    split = np.where(pos % 4 == 0, "val", "train")
    pd.DataFrame({"id": df["id"], "split": split}).to_parquet(tmp_path / "splits.parquet",
                                                              index=False)
    seen: dict = {}

    def fake_make_scorer(spec, seed=0, **options):
        seen.update(spec=spec, **{k: v for k, v in options.items() if v is not None})
        model = make(tokenizer, epochs=1)
        fit = model.fit

        def spy(texts, labels):
            seen["texts"] = list(texts)
            return fit(texts, labels)

        model.fit = spy
        return model

    monkeypatch.setattr(baselines, "make_scorer", fake_make_scorer)
    evaluate = _script("evaluate")
    args = ["--corpus", str(tmp_path / "corpus.parquet"), "--splits",
            str(tmp_path / "splits.parquet"), "--model", "transformer"]
    assert evaluate.main([
        *args, "--out-dir", str(tmp_path / "r"),
        "--backbone", "microsoft/deberta-v3-large", "--precision", "bf16", "--lr", "1e-5",
        "--grad-accum", "4", "--batch-size", "8", "--gradient-checkpointing",
        "--max-train", "45", "--keep-sources", "fresh,persuade"]) == 0
    assert seen["spec"] == "transformer"
    assert (seen["model_name"], seen["precision"], seen["lr"], seen["grad_accum"],
            seen["batch_size"], seen["gradient_checkpointing"]) == (
        "microsoft/deberta-v3-large", "bf16", 1e-5, 4, 8, True)
    kept = df[(split == "train") & df["source"].isin(["fresh", "persuade"]).to_numpy()]
    assert 5 < len(kept) < 45
    assert set(kept["text"]) <= set(seen["texts"]) and len(seen["texts"]) == 45
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert report["sampling"]["kept_rows"]["train"] == {
        s: int((kept["source"] == s).sum()) for s in ("fresh", "persuade")}
    assert report["model"]["name"] == "transformer"
    assert report["model"]["params"]["backbone"] == "distilbert"
    # Without the flags nothing new is passed (the model keeps its defaults).
    seen.clear()
    assert evaluate.main([*args, "--out-dir", str(tmp_path / "r2")]) == 0
    assert not {"precision", "lr", "grad_accum", "gradient_checkpointing"} & set(seen)
