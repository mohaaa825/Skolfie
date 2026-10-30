"""
test_binoculars.py
-------------------
Unit tests for skolfie.features.binoculars. No real language model and no
network: tiny "bigram" models (the logits depend only on the current token,
from a fixed random table) and a whitespace tokenizer stand in for the Qwen
pair, so every feature can be recomputed by hand.
"""

from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from skolfie.features.binoculars import (
    BINO_DESCRIPTIONS,
    BINO_FEATURES,
    BinocularsFeaturizer,
    _aggregate,
)

WORDS = "the cat sat on mat it was happy a dog ran far away and then slept".split()
TEXTS = [
    "the cat sat on the mat",
    "a dog ran far away and then slept",
    "it",
    "the dog was happy and the cat was happy",
    "",
    "   ",
    "cat",
]
EOS, PAD = 0, 1


class FakeTokenizer:
    """Whitespace tokenizer with the slice of the HF fast-tokenizer API the featurizer uses."""

    bos_token_id = None  # like Qwen: no BOS, so EOS is prepended
    eos_token_id = EOS
    pad_token_id = PAD

    def __init__(self, words=WORDS) -> None:
        self.vocab = {"<eos>": EOS, "<pad>": PAD, **{w: i + 2 for i, w in enumerate(words)}}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def __call__(self, texts, add_special_tokens=False, truncation=True, max_length=None,
                 return_attention_mask=False):
        ids = [[self.vocab[w] for w in t.split() if w in self.vocab] for t in texts]
        if truncation and max_length is not None:
            ids = [x[:max_length] for x in ids]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def bigram_model(torch, seed: int, vocab: int, n_pos: int = 64):
    """A causal LM whose next-token logits depend only on the current token."""

    class Bigram(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.table = torch.nn.Parameter(torch.randn(vocab, vocab, generator=gen) * 2)
            self.config = types.SimpleNamespace(max_position_embeddings=n_pos)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            logits = self.table[input_ids]
            if attention_mask is not None:
                # Garbage at padded positions: it must never reach a feature.
                logits = logits + (1 - attention_mask.unsqueeze(-1).float()) * 1e4
            return types.SimpleNamespace(logits=logits)

    return Bigram().eval()


@pytest.fixture(scope="module")
def pair(torch):
    tok = FakeTokenizer()
    v = len(tok.vocab)
    return bigram_model(torch, 0, v), bigram_model(torch, 1, v), tok


def make(pair, performer: bool = True, **kwargs) -> BinocularsFeaturizer:
    obs, perf, tok = pair
    return BinocularsFeaturizer(
        "fake/observer", "fake/performer" if performer else None, device="cpu",
        observer_model=obs, performer_model=perf if performer else None, tokenizer=tok,
        performer_tokenizer=FakeTokenizer() if performer else None, **kwargs)


def col(name: str) -> int:
    return BINO_FEATURES.index(name)


def by_hand(torch, pair, text: str) -> dict[str, float]:
    """The five features of one text, straight from the definitions."""
    obs, perf, tok = pair
    ids = tok([text])["input_ids"][0]
    full = torch.tensor([[EOS] + ids])
    with torch.no_grad():
        lo = torch.log_softmax(obs(full).logits[0, :-1].double(), -1)
        lp = torch.log_softmax(perf(full).logits[0, :-1].double(), -1)
    tgt = torch.tensor(ids)
    obs_nll = -lo[torch.arange(len(ids)), tgt]
    perf_nll = -lp[torch.arange(len(ids)), tgt]
    x = -(lo.exp() * lp).sum(-1)
    return {"bino_obs_log_ppl": obs_nll.mean().item(), "bino_x_ppl": x.mean().item(),
            "bino_score": (perf_nll.mean() / x.mean()).item(),
            "bino_perf_log_ppl": perf_nll.mean().item(),
            "bino_obs_nll_std": obs_nll.std(unbiased=False).item()}


# --------------------------------------------------------------------------- #
# Names, keys, imports
# --------------------------------------------------------------------------- #
def test_feature_names_prefixed_unique_and_described() -> None:
    assert len(BINO_FEATURES) == len(set(BINO_FEATURES)) == 5
    assert all(name.startswith("bino_") for name in BINO_FEATURES)
    assert set(BINO_DESCRIPTIONS) == set(BINO_FEATURES)
    assert all(isinstance(d, str) and d.strip() for d in BINO_DESCRIPTIONS.values())


def test_cache_key_encodes_models_truncation_and_quantization() -> None:
    pair_key = BinocularsFeaturizer("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct").cache_key
    assert pair_key == "Qwen_Qwen2.5-1.5B+Qwen_Qwen2.5-1.5B-Instruct-t256"
    solo = BinocularsFeaturizer("Qwen/Qwen3.8-27B", "", max_tokens=128, quantization="4bit")
    assert solo.performer is None and solo.cache_key == "Qwen_Qwen3.8-27B-t128-q4bit"
    assert len({pair_key, solo.cache_key,
                BinocularsFeaturizer("Qwen/Qwen3.8-27B", max_tokens=128).cache_key}) == 3


def test_constructor_errors() -> None:
    with pytest.raises(ValueError):
        BinocularsFeaturizer("")
    with pytest.raises(ValueError):
        BinocularsFeaturizer("x", max_tokens=0)
    with pytest.raises(ValueError):
        BinocularsFeaturizer("x", batch_size=0)
    with pytest.raises(ValueError, match="quantization"):
        BinocularsFeaturizer("x", quantization="3bit")
    with pytest.raises(ValueError):
        BinocularsFeaturizer("x", observer_model=object())  # no tokenizer


def test_import_does_not_import_torch() -> None:
    code = ("import sys, skolfie.features.binoculars; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'transformers' not in sys.modules, 'transformers imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- #
# The math
# --------------------------------------------------------------------------- #
def test_token_stats_on_tiny_logits(torch) -> None:
    from skolfie.features.binoculars import _bino_token_stats

    obs = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    perf = torch.tensor([[1.0, 1.0, 1.0], [5.0, 0.0, 0.0]])
    tgt = torch.tensor([0, 2])
    obs_nll, cross, perf_nll = _bino_token_stats(obs.half(), perf.half(), tgt)
    lo, lp = torch.log_softmax(obs, -1), torch.log_softmax(perf, -1)
    torch.testing.assert_close(obs_nll, -lo[[0, 1], tgt], atol=2e-3, rtol=0)
    torch.testing.assert_close(perf_nll, -lp[[0, 1], tgt], atol=2e-3, rtol=0)
    torch.testing.assert_close(cross, -(lo.exp() * lp).sum(-1), atol=2e-3, rtol=0)
    assert obs_nll.dtype == cross.dtype == perf_nll.dtype == torch.float32
    # Uniform performer: the cross-entropy is log(vocab) whatever the observer says.
    flat = _bino_token_stats(obs, torch.zeros(2, 3), tgt)[1]
    torch.testing.assert_close(flat, torch.full((2,), float(np.log(3))), atol=1e-4, rtol=0)
    # Observer-only.
    solo = _bino_token_stats(obs, None, tgt)
    assert solo[1] is None and solo[2] is None
    torch.testing.assert_close(solo[0], -lo[[0, 1], tgt])
    # Different logit widths (padded embedding rows) are cut to the common width.
    wide = torch.cat([perf, torch.full((2, 2), -50.0)], dim=-1)
    torch.testing.assert_close(_bino_token_stats(obs, wide, tgt)[1], cross.float(), atol=2e-3,
                               rtol=0)


def test_aggregate_score_is_perf_log_ppl_over_x_ppl() -> None:
    row = _aggregate(np.array([1.0, 3.0]), np.array([2.0, 2.0]), np.array([0.5, 0.5]))
    assert row[col("bino_obs_log_ppl")] == 2.0 and row[col("bino_x_ppl")] == 2.0
    assert row[col("bino_score")] == 0.25 and row[col("bino_perf_log_ppl")] == 0.5
    assert row[col("bino_obs_nll_std")] == 1.0
    assert np.isnan(_aggregate(np.zeros(0), np.zeros(0), np.zeros(0))).all()
    solo = _aggregate(np.array([1.0, 3.0]), None, None)
    assert np.isnan(solo[[col("bino_x_ppl"), col("bino_score"), col("bino_perf_log_ppl")]]).all()
    assert solo[col("bino_obs_log_ppl")] == 2.0


def test_features_match_hand_computation(torch, pair) -> None:
    X = make(pair).features(TEXTS)
    assert X.shape == (len(TEXTS), len(BINO_FEATURES)) and X.dtype == np.float32
    for i, text in enumerate(TEXTS):
        if not text.strip():
            continue
        expected = by_hand(torch, pair, text)
        for name, value in expected.items():
            assert X[i, col(name)] == pytest.approx(value, rel=1e-4, abs=1e-4), (text, name)
    np.testing.assert_allclose(X[:, col("bino_score")],
                               X[:, col("bino_perf_log_ppl")] / X[:, col("bino_x_ppl")],
                               rtol=1e-5)


def test_padding_batching_and_order(pair) -> None:
    X1 = make(pair, batch_size=1).features(TEXTS)
    X3 = make(pair, batch_size=3).features(TEXTS)
    # The fake model adds 1e4 to padded logits: any scored pad would show up here.
    np.testing.assert_allclose(X1, X3, rtol=1e-5, equal_nan=True)
    perm = np.random.default_rng(0).permutation(len(TEXTS))
    Xp = make(pair, batch_size=3).features([TEXTS[i] for i in perm])
    np.testing.assert_allclose(Xp, X3[perm], rtol=1e-5, equal_nan=True)


def test_empty_and_odd_input_gives_nan_rows(pair) -> None:
    feat = make(pair)
    X = feat.features(["", None, float("nan"), "   ", "unknown words only", "the cat"])
    assert np.isnan(X[:5]).all()
    assert np.isfinite(X[5]).all()
    assert feat.features([]).shape == (0, len(BINO_FEATURES))


def test_observer_only_mode(torch, pair) -> None:
    X = make(pair, performer=False).features(TEXTS)
    full = make(pair).features(TEXTS)
    nan_cols = [col("bino_x_ppl"), col("bino_score"), col("bino_perf_log_ppl")]
    assert np.isnan(X[:, nan_cols]).all()
    keep = [col("bino_obs_log_ppl"), col("bino_obs_nll_std")]
    np.testing.assert_allclose(X[:, keep], full[:, keep], rtol=1e-5, equal_nan=True)


def test_truncation(pair) -> None:
    long_text = " ".join(WORDS * 20)
    short = " ".join((WORDS * 20)[:5])
    X = make(pair, max_tokens=5).features([long_text])
    np.testing.assert_allclose(X, make(pair).features([short]), rtol=1e-5)
    # max_tokens above the context window (64 positions) is capped, not a crash.
    assert np.isfinite(make(pair, max_tokens=10_000).features([long_text])).all()


def test_tokenizer_mismatch_is_a_clear_error(pair) -> None:
    obs, perf, tok = pair
    other = FakeTokenizer(words=list(reversed(WORDS)))
    feat = BinocularsFeaturizer("fake/a", "fake/b", device="cpu", observer_model=obs,
                                performer_model=perf, tokenizer=tok, performer_tokenizer=other)
    with pytest.raises(ValueError, match="same tokenizer"):
        feat.features(["the cat"])


def test_progress_prints(pair, capsys) -> None:
    make(pair, batch_size=2).features(TEXTS, progress=True)
    assert f"[bino] {len(TEXTS)}/{len(TEXTS)}" in capsys.readouterr().out


def test_unknown_architecture_says_to_upgrade_transformers(torch, monkeypatch) -> None:
    """A model family this transformers does not know -> a hint to upgrade it (fake module)."""
    def from_pretrained(name, **kwargs):
        assert kwargs["trust_remote_code"] is False
        raise ValueError("The checkpoint you are trying to load has model type `qwen3_8`")

    fake = types.ModuleType("transformers")
    fake.__version__ = "4.0.0"
    fake.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    feat = BinocularsFeaturizer("Qwen/Qwen3.8-27B")
    with pytest.raises(ValueError, match="pip install -U transformers"):
        feat._load_model("Qwen/Qwen3.8-27B", "cpu")


def test_quantization_needs_cuda(torch, monkeypatch) -> None:
    fake = types.ModuleType("transformers")
    fake.__version__ = "4.0.0"
    fake.AutoModelForCausalLM = None
    monkeypatch.setitem(sys.modules, "transformers", fake)
    feat = BinocularsFeaturizer("Qwen/Qwen3.8-27B", quantization="4bit")
    with pytest.raises(ValueError, match="CUDA"):
        feat._load_model("Qwen/Qwen3.8-27B", "cpu")
