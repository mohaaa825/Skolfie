"""
tests/test_model_pickles.py
----------------------------
Models pickled by older versions must keep loading (see AGENTS.md, "Contracts
you must not break"). These checks need no torch, so they also run in CI,
where tests/test_transformer_model.py is skipped.
"""

from __future__ import annotations

from skolfie.models.transformer_model import TransformerModel


def test_transformer_model_state_without_chunk_fields() -> None:
    state = TransformerModel(use_features=False).__dict__.copy()
    for key in ("chunk_augment", "chunk_words", "n_chunks_"):
        del state[key]
    model = TransformerModel.__new__(TransformerModel)
    model.__setstate__(state)
    assert (model.chunk_augment, model.chunk_words, model.n_chunks_) == (0, 60, 0)
    assert "chunk_augment" not in model.params()



def test_old_pickles_keep_normaliser_version_1() -> None:
    from skolfie.models.feature_model import FeatureModel
    from skolfie.robust.normalize import CURRENT_NORMALIZER

    for cls, kwargs in ((TransformerModel, {"use_features": False}),
                        (FeatureModel, {"use_lm": False})):
        fresh = cls(**kwargs)
        assert fresh.normalizer == CURRENT_NORMALIZER == 2
        state = fresh.__dict__.copy()
        del state["normalizer"]
        model = cls.__new__(cls)
        model.__setstate__(state)
        assert model.normalizer == 1
