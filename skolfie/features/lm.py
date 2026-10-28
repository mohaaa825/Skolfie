"""
skolfie/features/lm.py
-----------------------
Language-model features (perplexity, burstiness, token-rank statistics)
computed with a causal LM -- GPT-2 small by default -- for the LightGBM
AI-vs-human classifier.

Every text gets exactly one forward pass. For each of its tokens t we take
the model's predicted distribution given the prefix and record:

    * NLL      = -log p(t | prefix)        (surprise)
    * rank     = 1 + #vocab entries with a strictly larger logit than t
    * entropy  = H(p(. | prefix))           (expected surprise)

and aggregate them into the per-text features listed in `LM_FEATURES`:
log-perplexity and its spread (token-level and GPTZero-style sentence-level
"burstiness"), GLTR-style top-k shares, and a DetectLLM-style
"surprise minus expected surprise" score. Machine text tends to have low,
uniform NLL and many top-1 / top-10 tokens; human text is burstier.

Details that matter:
    * The BOS token (for GPT-2 the same <|endoftext|> token as EOS) is
      prepended, so the first real token is scored too.
    * Texts are truncated to `max_tokens` tokens (further capped by the
      model's context window minus the BOS slot). The features therefore
      depend on the truncation length -- `LMFeaturizer.cache_key` encodes
      the model and truncation so cached features are never mixed up.
    * Texts are sorted by token length and batched (right padding plus an
      attention mask) to minimise padding; padded positions are never
      scored, so results do not depend on batch composition. Rows are
      returned in input order.
    * On CUDA the model runs in fp16, but all statistics are computed in
      float32 (on the GPU, in chunks, so no tensor much bigger than the
      logits is ever materialised). On CPU everything is float32.
    * Undefined values -- e.g. an empty text, which has zero tokens to
      score -- are NaN (LightGBM handles NaN natively). Nothing raises on
      "", whitespace-only or very long input.

torch / transformers are imported lazily, so importing this module is cheap
and works without them; they are only needed once features are computed.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

LM_FEATURES: tuple[str, ...] = (
    "lm_mean_nll",
    "lm_std_nll",
    "lm_p90_nll",
    "lm_sent_mean_nll_std",
    "lm_sent_nll_cv",
    "lm_mean_log_rank",
    "lm_top1_frac",
    "lm_top10_frac",
    "lm_mean_entropy",
    "lm_nll_minus_entropy",
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lm_mean_nll": "Average surprise of the language model per token (log-perplexity); "
    "AI text is usually less surprising.",
    "lm_std_nll": "How much the per-token surprise varies across the text "
    "(token-level burstiness).",
    "lm_p90_nll": "Surprise of the most surprising tokens (90th percentile of per-token surprise).",
    "lm_sent_mean_nll_std": "How much the average surprise varies from sentence to sentence "
    "(sentence-level burstiness, as in GPTZero).",
    "lm_sent_nll_cv": "Sentence-to-sentence variation in surprise relative to the average "
    "surprise (coefficient of variation).",
    "lm_mean_log_rank": "Average log rank of each actual token among the model's predictions "
    "(lower means the text follows the model's top guesses).",
    "lm_top1_frac": "Share of tokens that were the model's single most likely prediction.",
    "lm_top10_frac": "Share of tokens that were among the model's 10 most likely predictions.",
    "lm_mean_entropy": "Average uncertainty of the model's predictions (entropy of the "
    "predicted next-token distribution).",
    "lm_nll_minus_entropy": "Average gap between actual and expected surprise per token; "
    "near or below zero suggests machine-like predictability.",
}

_IDX = {name: i for i, name in enumerate(LM_FEATURES)}
_MIN_SENTENCE_TOKENS = 3
# A sentence ends after a run of . ! ? followed by whitespace, or at a newline.
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?=\s)|\n")
# Texts are clipped to this many characters per allowed token before
# tokenizing, purely to bound tokenizer cost on pathological input. No real
# token is this long, so this never changes the scored tokens of normal text.
_MAX_CHARS_PER_TOKEN = 256
# Target number of float32 elements per statistics chunk (~256 MB).
_CHUNK_ELEMENTS = 1 << 26


def _token_stats(
    logits: "torch.Tensor", targets: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Per-position NLL, rank and entropy, computed in float32.

    `logits` is (n, vocab) in any float dtype, `targets` is (n,) token ids.
    Returns three (n,) tensors on the same device: NLL (float32), rank
    (int64, 1 = most likely) and entropy in nats (float32).
    """
    import torch

    lf = logits.float()
    logz = torch.logsumexp(lf, dim=-1)
    true_logit = lf.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    nll = logz - true_logit
    rank = (lf > true_logit.unsqueeze(-1)).sum(dim=-1) + 1
    # lf becomes log-probabilities in place (it is already a fresh copy unless
    # the input was float32, in which case `.float()` aliased it -> clone).
    if lf is logits:
        lf = lf.clone()
    lf.sub_(logz.unsqueeze(-1))
    lf.clamp_(min=torch.finfo(lf.dtype).min)  # -inf * 0 would give NaN
    entropy = -(lf.exp() * lf).sum(dim=-1)
    return nll, rank, entropy


def _clean_text(t: Any, clip: int) -> str:
    """`t` as a str clipped to `clip` characters; None/NaN -> "".

    Lone surrogates (not encodable as UTF-8) make the Rust tokenizer raise,
    so they become "?" (one character each, so offsets are unchanged).
    """
    if not isinstance(t, str):
        t = "" if t is None or (isinstance(t, float) and math.isnan(t)) else str(t)
    t = t[:clip]
    if not t.isascii():
        try:
            t.encode("utf-8")
        except UnicodeEncodeError:
            t = t.encode("utf-8", "replace").decode("utf-8")
    return t


def _sentence_ids(text: str, starts: np.ndarray) -> np.ndarray:
    """Map token start offsets (characters) to sentence indices."""
    bounds = [m.end() for m in _SENTENCE_BOUNDARY.finditer(text)]
    if not bounds:
        return np.zeros(len(starts), dtype=np.int64)
    return np.searchsorted(np.asarray(bounds, dtype=np.int64), starts, side="right")


def _aggregate(nll: np.ndarray, rank: np.ndarray, ent: np.ndarray, sent: np.ndarray) -> np.ndarray:
    """Turn per-token statistics of one text into its feature row."""
    row = np.full(len(LM_FEATURES), np.nan, dtype=np.float64)
    if nll.size == 0:
        return row
    nll = nll.astype(np.float64)
    ent = ent.astype(np.float64)
    row[_IDX["lm_mean_nll"]] = nll.mean()
    row[_IDX["lm_std_nll"]] = nll.std()
    row[_IDX["lm_p90_nll"]] = np.percentile(nll, 90)

    counts = np.bincount(sent)
    sums = np.bincount(sent, weights=nll)
    keep = counts >= _MIN_SENTENCE_TOKENS
    sent_std = sent_cv = 0.0
    if keep.sum() >= 2:
        means = sums[keep] / counts[keep]
        sent_std = float(means.std())
        mu = float(means.mean())
        sent_cv = sent_std / mu if mu > 0 else 0.0
    row[_IDX["lm_sent_mean_nll_std"]] = sent_std
    row[_IDX["lm_sent_nll_cv"]] = sent_cv

    row[_IDX["lm_mean_log_rank"]] = np.log(rank.astype(np.float64)).mean()
    row[_IDX["lm_top1_frac"]] = (rank <= 1).mean()
    row[_IDX["lm_top10_frac"]] = (rank <= 10).mean()
    row[_IDX["lm_mean_entropy"]] = ent.mean()
    row[_IDX["lm_nll_minus_entropy"]] = (nll - ent).mean()
    return row


class LMFeaturizer:
    """Compute `LM_FEATURES` for texts with a causal language model.

    Args:
        model_name: Hugging Face model id, loaded lazily on first use with
            AutoModelForCausalLM / AutoTokenizer(use_fast=True).
        device: "cuda", "cpu", ... ; None picks "cuda" when available (or
            the injected model's device). fp16 is used only on CUDA.
        max_tokens: number of text tokens scored per text (the rest is
            truncated); also capped at the model's context size minus one.
        batch_size: texts per forward pass.
        model, tokenizer: inject an already-built model and fast tokenizer
            (used as-is apart from `.eval()` and, if `device` is given,
            moving the model there). Useful for tests and custom models.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        max_tokens: int = 512,
        batch_size: int = 16,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if (model is None) != (tokenizer is None):
            raise ValueError("inject both `model` and `tokenizer`, or neither")
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.batch_size = int(batch_size)
        self._device_arg = device
        self.device: str | None = None
        self.model = model
        self.tokenizer = tokenizer
        self._ready = False

    @property
    def cache_key(self) -> str:
        """Identifies model + truncation length, e.g. "gpt2-t512"."""
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_name).strip("_") or "lm"
        return f"{name}-t{self.max_tokens}"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return LM_FEATURES

    # ------------------------------------------------------------------
    # Lazy setup
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._ready:
            return
        import torch

        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if torch.device(device).type == "cuda":
                model = model.half()
            self.model = model.to(device)
        else:
            if self._device_arg is not None:
                device = self._device_arg
                self.model = self.model.to(device)
            else:
                device = str(next(self.model.parameters()).device)
        self.device = device
        self.model.eval()

        tok = self.tokenizer
        if not getattr(tok, "is_fast", False):
            raise ValueError("LMFeaturizer needs a fast tokenizer (offset mappings)")
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if bos is None:
            raise ValueError("tokenizer has neither a BOS nor an EOS token to prepend")
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        self._bos_id = int(bos)
        self._pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else self._bos_id

        cfg = self.model.config
        n_pos = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        self._n_text = min(self.max_tokens, int(n_pos) - 1) if n_pos else self.max_tokens
        self._ready = True

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Token ids and per-token sentence ids for each text (compact arrays)."""
        clip = self._n_text * _MAX_CHARS_PER_TOKEN
        ids_out: list[np.ndarray] = []
        sent_out: list[np.ndarray] = []
        chunk = 1024
        for lo in range(0, len(texts), chunk):
            part = [_clean_text(t, clip) for t in texts[lo : lo + chunk]]
            enc = self.tokenizer(
                part,
                add_special_tokens=False,
                truncation=True,
                max_length=self._n_text,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            for text, ids, offs in zip(part, enc["input_ids"], enc["offset_mapping"]):
                ids = ids[: self._n_text]
                starts = np.fromiter((o[0] for o in offs[: len(ids)]), np.int64, len(ids))
                ids_out.append(np.asarray(ids, dtype=np.int64))
                sent_out.append(_sentence_ids(text, starts))
        return ids_out, sent_out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_batch(self, batch: list[np.ndarray]) -> list[tuple[np.ndarray, ...]]:
        """Forward one batch; return (nll, rank, entropy) arrays per text."""
        import torch

        lengths = [len(x) for x in batch]
        seq_len = max(lengths) + 1
        ids = np.full((len(batch), seq_len), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), seq_len), dtype=np.int64)
        for i, x in enumerate(batch):
            ids[i, 0] = self._bos_id
            ids[i, 1 : len(x) + 1] = x
            mask[i, : len(x) + 1] = 1
        input_ids = torch.from_numpy(ids).to(self.device)
        attn = torch.from_numpy(mask).to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        # Scored positions: predictions at 0..len-1 for targets at 1..len.
        rows, cols = attn[:, 1:].nonzero(as_tuple=True)  # row-major = per-text order
        targets = input_ids[rows, cols + 1]
        step = max(1, _CHUNK_ELEMENTS // logits.shape[-1])
        nll_parts, rank_parts, ent_parts = [], [], []
        for lo in range(0, rows.numel(), step):
            r, c = rows[lo : lo + step], cols[lo : lo + step]
            nll, rank, ent = _token_stats(logits[r, c], targets[lo : lo + step])
            nll_parts.append(nll)
            rank_parts.append(rank)
            ent_parts.append(ent)
        del logits
        nll = torch.cat(nll_parts).cpu().numpy()
        rank = torch.cat(rank_parts).cpu().numpy()
        ent = torch.cat(ent_parts).cpu().numpy()
        splits = np.cumsum(lengths)[:-1]
        return list(zip(np.split(nll, splits), np.split(rank, splits), np.split(ent, splits)))

    def features(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Feature matrix of shape (len(texts), len(LM_FEATURES)), float32, input order."""
        texts = list(texts)
        n = len(texts)
        out = np.full((n, len(LM_FEATURES)), np.nan, dtype=np.float32)
        if n == 0:
            return out
        self._load()
        import torch

        ids, sents = self._encode(texts)
        lengths = np.fromiter((len(x) for x in ids), np.int64, n)
        # Longest first: padding is minimal and any OOM shows up immediately.
        order = [int(i) for i in np.argsort(-lengths, kind="stable") if lengths[i] > 0]

        done = n - len(order)  # empty texts are already NaN rows
        next_report = 0.0
        start = time.perf_counter()
        with torch.inference_mode():
            for lo in range(0, len(order), self.batch_size):
                idx = order[lo : lo + self.batch_size]
                stats = self._score_batch([ids[i] for i in idx])
                for i, (nll, rank, ent) in zip(idx, stats):
                    out[i] = _aggregate(nll, rank, ent, sents[i])
                done += len(idx)
                if progress and (done / n >= next_report or done == n):
                    rate = done / max(time.perf_counter() - start, 1e-9)
                    print(
                        f"[lm] {done}/{n} texts ({100 * done / n:.0f}%), {rate:.1f} texts/s",
                        flush=True,
                    )
                    next_report = math.floor(done / n * 20 + 1) / 20
        return out
