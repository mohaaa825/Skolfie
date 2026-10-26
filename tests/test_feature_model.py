"""
tests/test_feature_model.py
----------------------------
Roadmap step 4: the Delta featurizer, the on-disk feature cache, and the
LightGBM / random-forest FeatureModel (with a fake LM standing in for GPT-2).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from skolfie.features.cache import FeatureCache, text_key
from skolfie.features.delta_features import DELTA_FEATURES, FEATURE_DESCRIPTIONS, DeltaFeaturizer
from tests.test_evaluate import make_corpus

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(n_per_cell=16, seed=1)


# --------------------------------------------------------------------------- #
# Delta features
# --------------------------------------------------------------------------- #
def test_delta_names_described() -> None:
    assert set(DELTA_FEATURES) == set(FEATURE_DESCRIPTIONS)


def test_delta_separates_classes(corpus) -> None:
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    delta = DeltaFeaturizer(n_words=50).fit(texts, labels)
    X = delta.transform(texts)
    assert X.shape == (len(texts), len(DELTA_FEATURES)) and X.dtype == np.float32
    ai = np.array([lab == "ai" for lab in labels])
    diff = X[:, DELTA_FEATURES.index("delta_cos_diff")]
    assert diff[ai].mean() > 0 > diff[~ai].mean()
    bur = X[:, DELTA_FEATURES.index("delta_bur_diff")]
    assert bur[ai].mean() > bur[~ai].mean()
    assert 0 < len(delta.vocabulary) <= 50


def test_delta_matches_reference_model(corpus) -> None:
    """Same Cosine Delta as the per-text BurrowsDeltaModel on the same vocabulary."""
    from skolfie.features.burrows_delta import BurrowsDeltaModel
    from skolfie.features.delta_features import _WORD_RE

    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    delta = DeltaFeaturizer(n_words=40).fit(texts, labels)
    ref = BurrowsDeltaModel(method="cosine").fit(
        [_WORD_RE.findall(t.lower()) for t in texts], labels, n_words=10_000,
        candidate_vocab=delta.vocabulary)
    assert sorted(ref.vocabulary) == sorted(delta.vocabulary)
    X = delta.transform(texts[:5])
    for i, text in enumerate(texts[:5]):
        out = ref.score(_WORD_RE.findall(text.lower()))
        assert X[i, 0] == pytest.approx(out["delta_human"], abs=1e-4)
        assert X[i, 1] == pytest.approx(out["delta_ai"], abs=1e-4)


def test_delta_chunking_and_empty_texts(corpus) -> None:
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    a = DeltaFeaturizer(n_words=30).fit(texts, labels)
    b = DeltaFeaturizer(n_words=30, chunk_size=7).fit(texts, labels)
    probe = texts[:20] + ["", "123 456 !!!"]
    np.testing.assert_allclose(a.transform(probe), b.transform(probe), rtol=1e-5, atol=1e-6)
    out = a.transform(probe)
    assert np.isnan(out[-2:]).all() and np.isfinite(out[:-2]).all()


def test_delta_errors() -> None:
    with pytest.raises(RuntimeError):
        DeltaFeaturizer().transform(["text"])
    with pytest.raises(ValueError):
        DeltaFeaturizer().fit(["a b c", "d e f"], ["human", "human"])
    with pytest.raises(ValueError):
        DeltaFeaturizer().fit(["1 2", "3 4"], ["human", "ai"])


# --------------------------------------------------------------------------- #
# Feature cache
# --------------------------------------------------------------------------- #
def _counting_fn(calls: list[int]):
    def fn(texts):
        calls.append(len(texts))
        return np.array([[len(t), t.count("a")] for t in texts], dtype=np.float32)
    return fn


def test_cache_computes_only_new_texts(tmp_path: Path) -> None:
    calls: list[int] = []
    cache = FeatureCache(tmp_path, "toy-v1", ["len", "n_a"], flush_every=2, log=lambda m: None)
    out = cache.compute(["aa", "b", "aa", "cab"], _counting_fn(calls))
    np.testing.assert_array_equal(out, [[2, 2], [1, 0], [2, 2], [3, 1]])
    assert calls == [2, 1]  # 3 unique texts, flushed every 2
    assert (tmp_path / "toy-v1.parquet").is_file()

    again = FeatureCache(tmp_path, "toy-v1", ["len", "n_a"], log=lambda m: None)
    assert len(again) == 3
    out = again.compute(["cab", "new", "b"], _counting_fn(calls))
    np.testing.assert_array_equal(out, [[3, 1], [3, 0], [1, 0]])
    assert calls == [2, 1, 1]
    assert again.compute([], _counting_fn(calls)).shape == (0, 2)


def test_cache_ignores_changed_columns_and_works_in_memory(tmp_path: Path) -> None:
    FeatureCache(tmp_path, "toy", ["len", "n_a"], log=lambda m: None).compute(
        ["x"], _counting_fn([]))
    changed = FeatureCache(tmp_path, "toy", ["len", "other"], log=lambda m: None)
    assert len(changed) == 0
    calls: list[int] = []
    memory = FeatureCache(None, "toy", ["len", "n_a"], flush_every=1, log=lambda m: None)
    memory.compute(["a", "b", "c"], _counting_fn(calls))
    assert calls == [3]  # no disk -> one batch
    assert text_key("a") != text_key("a ")


def test_cache_rejects_wrong_shape(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path, "bad", ["x", "y"], log=lambda m: None)
    with pytest.raises(ValueError):
        cache.compute(["a"], lambda texts: np.zeros((len(texts), 3)))


# --------------------------------------------------------------------------- #
# FeatureModel
# --------------------------------------------------------------------------- #
class FakeLM:
    """Stands in for LMFeaturizer: a cheap, label-correlated 'perplexity'."""

    cache_key = "fake-t8"

    def __init__(self) -> None:
        self.calls = 0

    def features(self, texts, progress=False):
        from skolfie.features.lm import LM_FEATURES

        self.calls += len(texts)
        out = np.zeros((len(texts), len(LM_FEATURES)), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.split()
            out[i, 0] = len(set(words)) / max(len(words), 1)
        return out


@pytest.fixture
def model_cls():
    pytest.importorskip("skolfie.features.style")
    from skolfie.models.feature_model import FeatureModel

    return FeatureModel


@pytest.mark.parametrize("classifier", ["lgbm", "rf"])
def test_feature_model_nolm_fit_predict(model_cls, corpus, classifier) -> None:
    if classifier == "lgbm":
        pytest.importorskip("lightgbm")
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    model = model_cls(classifier=classifier, use_lm=False, delta_words=50, delta_folds=3,
                      n_jobs=1, log=lambda m: None, lgbm_params={"n_estimators": 50})
    model.fit(texts, labels)
    p = model.predict_proba(texts)
    assert p.shape == (len(texts),) and ((p >= 0) & (p <= 1)).all()
    y = np.array([lab == "ai" for lab in labels])
    assert p[y].mean() > p[~y].mean() + 0.3
    assert model.predict_proba([]).shape == (0,)

    names = model.feature_names_
    assert names[-len(DELTA_FEATURES):] == list(DELTA_FEATURES)
    assert all(n.startswith(("sty_", "delta_")) for n in names)
    imp = model.feature_importance()
    assert [r["feature"] for r in imp] and sum(r["importance"] for r in imp) == pytest.approx(1)
    assert {r["feature"] for r in imp} == set(names)
    assert model.name == ("features-nolm" if classifier == "lgbm" else "features-nolm-rf")
    assert model.params()["n_features"] == len(names)


def test_feature_model_oof_delta_differs_from_in_sample(model_cls, corpus) -> None:
    """Training rows get out-of-fold Delta features, not in-sample ones."""
    pytest.importorskip("lightgbm")
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    model = model_cls(use_lm=False, delta_words=50, delta_folds=4, n_jobs=1,
                      log=lambda m: None)
    from skolfie.eval.metrics import to_binary

    oof = model._oof_delta(texts, to_binary(labels))
    full = DeltaFeaturizer(n_words=50).fit(texts, labels).transform(texts)
    assert np.isfinite(oof).all()
    assert not np.allclose(oof, full)


def test_feature_model_with_lm_and_cache(model_cls, corpus, tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("skolfie.features.lm")
    from skolfie.features.lm import LM_FEATURES

    fake = FakeLM()
    texts, labels = corpus["text"].tolist(), corpus["label"].tolist()
    model = model_cls(use_lm=True, delta_words=50, delta_folds=3, n_jobs=1, cache_dir=tmp_path,
                      log=lambda m: None, lgbm_params={"n_estimators": 50})
    model._lm = fake
    model.fit(texts, labels)
    assert fake.calls == len(set(texts))
    assert any(n.startswith("lm_") for n in model.feature_names_)
    assert len(model.feature_names_) == model.featurize(texts[:2]).shape[1]
    cached = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert cached == ["lm-v1-fake-t8.parquet", "style-v1.parquet"]
    model.predict_proba(texts)
    assert fake.calls == len(set(texts))  # all cached now
    assert model.params()["lm_model"] == "gpt2"
    assert len(LM_FEATURES) > 0

    # Pickling drops the LM and caches; the reloaded model scores from the disk cache.
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded._lm is None and loaded._caches == {}
    loaded._lm = fake
    np.testing.assert_allclose(loaded.predict_proba(texts), model.predict_proba(texts))
    assert fake.calls == len(set(texts))


def test_feature_model_errors(model_cls) -> None:
    with pytest.raises(ValueError):
        model_cls(classifier="svm")
    with pytest.raises(ValueError):
        model_cls(delta_folds=1)
    model = model_cls(use_lm=False, log=lambda m: None)
    with pytest.raises(RuntimeError):
        model.predict_proba(["text"])
    with pytest.raises(ValueError):
        model.fit(["a b", "c d"], ["ai", "ai"])
    assert model.feature_importance() == []


def test_make_scorer_feature_specs(model_cls, tmp_path: Path) -> None:
    from skolfie.models.baselines import make_scorer

    m = make_scorer("features-nolm-rf", seed=3, cache_dir=tmp_path, lm_model=None)
    assert isinstance(m, model_cls)
    assert (m.classifier, m.use_lm, m.seed, m.cache_dir) == ("rf", False, 3, str(tmp_path))
    assert m.lm_model == "gpt2"  # None options fall back to the defaults
    assert make_scorer("features").use_lm is True
