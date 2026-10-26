"""
test_lm.py
-----------
Unit tests for skolfie.features.lm. huggingface.co may be unreachable in
CI, so a tiny random GPT-2 and a byte-level BPE tokenizer are built locally.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from skolfie.features.lm import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    LM_FEATURES,
    LMFeaturizer,
    _token_stats,
)

EOS = "<|endoftext|>"
TRAIN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries across the world!",
    "Honestly? I have no idea what happened, but it was fun.",
    "We walked to the market, bought apples, and went home again.",
    "Large language models predict the next token given a prefix.",
    "She said it would rain tomorrow, so we stayed inside.\nThen it snowed.",
] * 3

TEXTS = [
    "The cat sat on the mat. It was happy! Was it?",
    "Large language models predict tokens.\nThey are trained on text.",
    "one",
    "We walked to the market and bought apples. Then we went home. It rained.",
    "Honestly I have no idea what happened but it was fun and we laughed a lot.",
    "A. B. C.",
    "   ",
]


@pytest.fixture(scope="module")
def tiny():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(TRAIN_SENTENCES, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=EOS, eos_token=EOS)
    eos_id = fast.convert_tokens_to_ids(EOS)

    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=len(fast),
        n_positions=128,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(config).eval()
    return model, fast


def make(tiny, **kwargs) -> LMFeaturizer:
    model, tok = tiny
    return LMFeaturizer(model=model, tokenizer=tok, device="cpu", **kwargs)


def col(name: str) -> int:
    return LM_FEATURES.index(name)


def test_feature_names_prefixed_unique_and_described():
    assert len(LM_FEATURES) == len(set(LM_FEATURES))
    assert all(name.startswith("lm_") for name in LM_FEATURES)
    assert set(FEATURE_DESCRIPTIONS) == set(LM_FEATURES)
    assert all(isinstance(d, str) and d.strip() for d in FEATURE_DESCRIPTIONS.values())


def test_cache_key():
    assert LMFeaturizer().cache_key == "gpt2-t512"
    assert LMFeaturizer("openai-community/gpt2", max_tokens=256).cache_key.endswith("-t256")


def test_import_does_not_import_torch():
    code = (
        "import sys, skolfie.features.lm; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'transformers' not in sys.modules, 'transformers imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_shape_dtype_and_empty_input(tiny):
    feat = make(tiny)
    X = feat.features(TEXTS)
    assert X.shape == (len(TEXTS), len(LM_FEATURES))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()  # every text here has >= 1 token
    empty = feat.features([])
    assert empty.shape == (0, len(LM_FEATURES)) and empty.dtype == np.float32


def test_order_is_preserved_under_shuffling(tiny):
    feat = make(tiny, batch_size=3)
    X = feat.features(TEXTS)
    perm = np.random.default_rng(0).permutation(len(TEXTS))
    Xp = feat.features([TEXTS[i] for i in perm])
    np.testing.assert_allclose(Xp, X[perm], atol=1e-5)


def test_batch_size_does_not_change_values(tiny):
    X1 = make(tiny, batch_size=1).features(TEXTS)
    X4 = make(tiny, batch_size=4).features(TEXTS)
    np.testing.assert_allclose(X1, X4, atol=1e-4)


def test_empty_text_gives_nan_row(tiny):
    X = make(tiny).features(["", TEXTS[0], ""])
    assert np.isnan(X[0]).all() and np.isnan(X[2]).all()
    assert np.isfinite(X[1]).all()
    assert np.isnan(make(tiny).features([""])).all()


def test_mean_nll_and_top1_match_hand_computation(tiny):
    model, tok = tiny
    text = TEXTS[3]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    full = torch.tensor([[tok.eos_token_id] + ids])
    with torch.no_grad():
        logits = model(full).logits[0, :-1].float()
    target = torch.tensor(ids)
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp[torch.arange(len(ids)), target]
    entropy = -(logp.exp() * logp).sum(-1)
    top1 = (logits.argmax(-1) == target).float().mean()

    row = make(tiny).features([text])[0]
    assert row[col("lm_mean_nll")] == pytest.approx(nll.mean().item(), abs=1e-4)
    assert row[col("lm_std_nll")] == pytest.approx(nll.std(unbiased=False).item(), abs=1e-4)
    assert row[col("lm_mean_entropy")] == pytest.approx(entropy.mean().item(), abs=1e-4)
    assert row[col("lm_top1_frac")] == pytest.approx(top1.item(), abs=1e-6)


def test_fractions_are_valid(tiny):
    X = make(tiny).features(TEXTS)
    top1, top10 = X[:, col("lm_top1_frac")], X[:, col("lm_top10_frac")]
    assert ((top1 >= 0) & (top1 <= 1)).all()
    assert ((top10 >= 0) & (top10 <= 1)).all()
    assert (top10 >= top1).all()
    assert (X[:, col("lm_mean_log_rank")] >= 0).all()
    assert (X[:, col("lm_sent_mean_nll_std")] >= 0).all()


def test_sentence_features_zero_with_fewer_than_two_sentences(tiny):
    X = make(tiny).features(["one", "The quick brown fox jumps over the lazy dog"])
    assert X[:, col("lm_sent_mean_nll_std")].tolist() == [0.0, 0.0]
    assert X[:, col("lm_sent_nll_cv")].tolist() == [0.0, 0.0]


def test_token_stats_rank_and_entropy():
    logits = torch.tensor([[0.0, 5.0, 1.0], [0.0, 5.0, 1.0], [0.0, 5.0, 1.0], [2.0, 2.0, 2.0]])
    targets = torch.tensor([1, 2, 0, 1])
    nll, rank, ent = _token_stats(logits.half(), targets)
    assert rank.tolist() == [1, 2, 3, 1]  # argmax -> 1, ties count as rank 1
    logp = torch.log_softmax(logits, -1)
    expected = -logp[torch.arange(4), targets]
    torch.testing.assert_close(nll, expected, atol=1e-3, rtol=0)
    assert nll.dtype == torch.float32 and ent.dtype == torch.float32
    assert ent[3].item() == pytest.approx(np.log(3), abs=1e-4)
    # float32 input must not be modified in place
    before = logits.clone()
    _token_stats(logits, targets)
    assert torch.equal(logits, before)


def test_truncation(tiny):
    _, tok = tiny
    long_text = " ".join(TEXTS[:5]) * 200
    # max_tokens above the context window is capped instead of crashing.
    X = make(tiny, max_tokens=10_000).features([long_text, ""])
    assert np.isfinite(X[0]).all() and np.isnan(X[1]).all()

    # Scoring with max_tokens=k equals scoring exactly the first k tokens.
    enc = tok(long_text, add_special_tokens=False, return_offsets_mapping=True)
    k = next(
        k
        for k in range(32, 0, -1)
        if tok(long_text[: enc["offset_mapping"][k][0]], add_special_tokens=False)["input_ids"]
        == enc["input_ids"][:k]
    )
    prefix = long_text[: enc["offset_mapping"][k][0]]
    truncated = make(tiny, max_tokens=k).features([long_text])
    full_prefix = make(tiny, max_tokens=100).features([prefix])
    np.testing.assert_allclose(truncated, full_prefix, atol=1e-5)
    assert not np.allclose(truncated, make(tiny, max_tokens=64).features([long_text]))


def test_progress_prints(tiny, capsys):
    make(tiny, batch_size=2).features(TEXTS, progress=True)
    out = capsys.readouterr().out
    assert f"{len(TEXTS)}/{len(TEXTS)}" in out
