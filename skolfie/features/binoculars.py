"""
skolfie/features/binoculars.py
-------------------------------
Binoculars features (Hans et al. 2024, "Spotting LLMs With Binoculars:
Zero-Shot Detection of Machine-Generated Text", ICML) from a pair of modern
causal LMs, as extra columns for the LightGBM feature model.

Two models with the same tokenizer read the same text:

    observer   e.g. Qwen/Qwen2.5-1.5B (base) or Qwen/Qwen3.8-27B
    performer  e.g. Qwen/Qwen2.5-1.5B-Instruct; optional

For each token t (given its prefix) we record

    * obs NLL   = -log p_obs(t | prefix)
    * perf NLL  = -log p_perf(t | prefix)
    * cross     = -sum_v p_obs(v | prefix) * log p_perf(v | prefix)
                  (cross-entropy of the performer against the observer's
                  next-token distribution: softmax(observer logits) against
                  log_softmax(performer logits), as in the authors' reference
                  code, github.com/ahans30/Binoculars)

and aggregate them into `BINO_FEATURES`. The Binoculars score is the
performer's log-perplexity divided by the cross-perplexity (again as in the
reference code): perplexity alone flags any
"predictable" text, while dividing by how surprising one LM finds the other
LM's guesses calibrates for prompts that make any continuation predictable.
Low scores mean AI text. Without a performer ("observer-only mode") the
cross, score and performer columns are NaN; the observer's perplexity and
burstiness from a modern LM are still a much stronger signal than GPT-2's.

Details that matter (they mirror `skolfie.features.lm`):
    * The two tokenizers must have the same vocabulary (checked, ValueError
      otherwise), so both models see the same ids. The observer tokenizer's
      BOS token (or EOS when there is none, as for Qwen) is prepended to
      both, so the first real token is scored too.
    * Texts are truncated to `max_tokens` tokens (further capped by the
      observer's context window minus the BOS slot); `cache_key` encodes
      both model ids, the truncation and the quantization.
    * Texts are sorted by token length and batched with right padding and
      an attention mask; padded positions are never scored, so results do
      not depend on batch composition. Rows are returned in input order.
    * On CUDA the models run in fp16 (or 4-/8-bit via bitsandbytes with
      `quantization`), but all statistics are computed in float32, in
      chunks of scored positions, so no tensor much bigger than the logits
      is ever materialised.
    * An empty text (zero tokens) gives a NaN row; nothing raises on "",
      whitespace-only or very long input.

Cost: two 1.5B models in fp16 on a free Colab T4 take roughly 0.05-0.1 s
per text at 256 tokens (about 10x GPT-2 small); a 4-bit 27B observer alone
roughly 0.3-1 s per text on an L4. torch / transformers (and bitsandbytes
for quantization) are imported lazily, so importing this module is cheap.
"""

from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from skolfie.features.lm import _MAX_CHARS_PER_TOKEN, _clean_text

if TYPE_CHECKING:  # pragma: no cover
    import torch

BINO_FEATURES: tuple[str, ...] = (
    "bino_obs_log_ppl",
    "bino_x_ppl",
    "bino_score",
    "bino_perf_log_ppl",
    "bino_obs_nll_std",
)

BINO_DESCRIPTIONS: dict[str, str] = {
    "bino_obs_log_ppl": "Average surprise per token of a modern language model (the observer's "
    "log-perplexity); AI text is usually less surprising.",
    "bino_x_ppl": "How surprised a chat-tuned model is by a base model's guesses for each "
    "next word (cross-perplexity); it measures how predictable the text's context makes any "
    "continuation.",
    "bino_score": "Binoculars score: the chat model's surprise divided by the cross-perplexity. "
    "Low values are typical of AI text, even for prompts that make any answer predictable.",
    "bino_perf_log_ppl": "Average surprise per token of the chat-tuned (performer) model; "
    "AI text is usually very predictable to a chat model.",
    "bino_obs_nll_std": "How much the observer model's per-token surprise varies across the "
    "text (burstiness); human writing is usually burstier.",
}

QUANTIZATIONS = ("4bit", "8bit")

_IDX = {name: i for i, name in enumerate(BINO_FEATURES)}
# Target number of float32 elements per statistics chunk (~256 MB per tensor).
_CHUNK_ELEMENTS = 1 << 26


def _bino_token_stats(
    obs_logits: "torch.Tensor",
    perf_logits: "torch.Tensor | None",
    targets: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor | None", "torch.Tensor | None"]:
    """Per-position observer NLL, cross-entropy and performer NLL, in float32.

    `obs_logits` / `perf_logits` are (n, vocab) in any float dtype (the
    performer may be None: observer-only mode), `targets` is (n,) token ids.
    When the two logit widths differ (embedding matrices padded to different
    multiples), both are cut to the common width: the tokenizers are
    identical, so the extra rows are never real tokens.
    Returns (obs_nll, cross, perf_nll); the last two are None without a performer.
    """
    import torch

    if perf_logits is not None and perf_logits.shape[-1] != obs_logits.shape[-1]:
        width = min(perf_logits.shape[-1], obs_logits.shape[-1])
        obs_logits, perf_logits = obs_logits[..., :width], perf_logits[..., :width]
    tgt = targets.long().unsqueeze(-1)
    obs_logp = torch.log_softmax(obs_logits.float(), dim=-1)
    obs_nll = -obs_logp.gather(-1, tgt).squeeze(-1)
    if perf_logits is None:
        return obs_nll, None, None
    perf_logp = torch.log_softmax(perf_logits.float(), dim=-1)
    perf_nll = -perf_logp.gather(-1, tgt).squeeze(-1)
    # -inf log-probabilities times 0 would give NaN.
    perf_logp.clamp_(min=torch.finfo(perf_logp.dtype).min)
    cross = -(obs_logp.exp() * perf_logp).sum(dim=-1)
    return obs_nll, cross, perf_nll


def _aggregate(obs_nll: np.ndarray, cross: np.ndarray | None,
               perf_nll: np.ndarray | None) -> np.ndarray:
    """Turn per-token statistics of one text into its feature row."""
    row = np.full(len(BINO_FEATURES), np.nan, dtype=np.float64)
    if obs_nll.size == 0:
        return row
    obs = obs_nll.astype(np.float64)
    row[_IDX["bino_obs_log_ppl"]] = obs.mean()
    row[_IDX["bino_obs_nll_std"]] = obs.std()
    if cross is not None and perf_nll is not None:
        x_ppl = float(cross.astype(np.float64).mean())
        perf_log_ppl = float(perf_nll.astype(np.float64).mean())
        row[_IDX["bino_x_ppl"]] = x_ppl
        row[_IDX["bino_score"]] = perf_log_ppl / x_ppl if x_ppl > 0 else np.nan
        row[_IDX["bino_perf_log_ppl"]] = perf_log_ppl
    return row


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "lm"


class BinocularsFeaturizer:
    """Compute `BINO_FEATURES` for texts with an observer (+ performer) causal LM pair.

    Args:
        observer: Hugging Face id of the observer model (a base model).
        performer: id of the performer (usually the instruct/chat version of
            the observer, with the same tokenizer); None or "" for
            observer-only features (the cross / score / performer columns
            are then NaN).
        max_tokens: number of text tokens scored per text (the rest is
            truncated); also capped at the observer's context size minus one.
        batch_size: texts per forward pass.
        device: "cuda", "cpu", ... ; None picks "cuda" when available (or
            the injected observer's device). fp16 is used only on CUDA.
        quantization: None, "4bit" (nf4, fp16 compute) or "8bit": load the
            models with bitsandbytes (CUDA only), e.g. for a 27B observer on
            a 24 GB GPU.
        observer_model, performer_model, tokenizer, performer_tokenizer:
            inject already-built models and tokenizers (used as-is apart
            from `.eval()` and, if `device` is given, moving the models
            there). Used by the tests.
    """

    def __init__(
        self,
        observer: str,
        performer: str | None = None,
        max_tokens: int = 256,
        batch_size: int = 8,
        device: str | None = None,
        quantization: str | None = None,
        observer_model: Any = None,
        performer_model: Any = None,
        tokenizer: Any = None,
        performer_tokenizer: Any = None,
    ) -> None:
        if not observer:
            raise ValueError("BinocularsFeaturizer needs an observer model id")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        quantization = quantization or None
        if quantization is not None and quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS} or None, "
                             f"got {quantization!r}")
        if (observer_model is None) != (tokenizer is None):
            raise ValueError("inject both `observer_model` and `tokenizer`, or neither")
        self.observer = observer
        self.performer = performer or None
        self.max_tokens = int(max_tokens)
        self.batch_size = int(batch_size)
        self.quantization = quantization
        self._device_arg = device
        self.device: str | None = None
        self.observer_model = observer_model
        self.performer_model = performer_model
        self.tokenizer = tokenizer
        self.performer_tokenizer = performer_tokenizer
        self._ready = False

    @property
    def cache_key(self) -> str:
        """Both model ids, truncation and quantization, e.g.
        "Qwen_Qwen2.5-1.5B+Qwen_Qwen2.5-1.5B-Instruct-t256"."""
        key = _slug(self.observer)
        if self.performer:
            key += "+" + _slug(self.performer)
        key += f"-t{self.max_tokens}"
        if self.quantization:
            key += f"-q{self.quantization}"
        return key

    @property
    def feature_names(self) -> tuple[str, ...]:
        return BINO_FEATURES

    # ------------------------------------------------------------------
    # Lazy setup
    # ------------------------------------------------------------------
    def _load_model(self, name: str, device: str) -> Any:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {"trust_remote_code": False}
        on_cuda = torch.device(device).type == "cuda"
        if self.quantization is not None:
            if not on_cuda:
                raise ValueError(f"{self.quantization} quantization needs a CUDA GPU "
                                 f"(bitsandbytes); got device {device!r}")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(f"{self.quantization} quantization needs bitsandbytes: "
                                  "pip install -U bitsandbytes") from exc
            from transformers import BitsAndBytesConfig

            if self.quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": device}
            kwargs["torch_dtype"] = torch.float16
        elif on_cuda:
            kwargs["torch_dtype"] = torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except (KeyError, ValueError) as exc:
            # An architecture this transformers version does not know shows up
            # as "model type ... not recognized" (ValueError) or a KeyError.
            raise ValueError(
                f"transformers {transformers.__version__} cannot load {name!r} ({exc}). "
                "A new model family usually needs a newer transformers: "
                "pip install -U transformers (trust_remote_code stays off)") from exc
        if self.quantization is None:
            model = model.to(device)
        return model

    def _load(self) -> None:
        if self._ready:
            return
        import torch

        if self.observer_model is None:
            from transformers import AutoTokenizer

            device = self._device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.observer)
            if self.performer:
                self.performer_tokenizer = AutoTokenizer.from_pretrained(self.performer)
            self._check_tokenizers()  # before downloading the second model
            self.observer_model = self._load_model(self.observer, device)
            if self.performer:
                self.performer_model = self._load_model(self.performer, device)
        else:
            if self._device_arg is not None:
                device = self._device_arg
                self.observer_model = self.observer_model.to(device)
                if self.performer_model is not None:
                    self.performer_model = self.performer_model.to(device)
            else:
                device = str(next(self.observer_model.parameters()).device)
            if self.performer and self.performer_model is None:
                raise ValueError("a performer id is set but no performer_model was injected")
            if self.performer_model is not None and self.performer_tokenizer is None:
                self.performer_tokenizer = self.tokenizer
            self._check_tokenizers()
        self.device = device
        self.observer_model.eval()
        if self.performer_model is not None:
            self.performer_model.eval()

        tok = self.tokenizer
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if bos is None:
            raise ValueError("observer tokenizer has neither a BOS nor an EOS token to prepend")
        self._bos_id = int(bos)
        pad = getattr(tok, "pad_token_id", None)
        self._pad_id = int(pad) if pad is not None else self._bos_id

        cfg = self.observer_model.config
        n_pos = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        self._n_text = min(self.max_tokens, int(n_pos) - 1) if n_pos else self.max_tokens
        self._ready = True

    def _check_tokenizers(self) -> None:
        """Both models must map text to the same ids, or the cross-entropy is meaningless."""
        other = self.performer_tokenizer
        if other is None or other is self.tokenizer:
            return
        if self.tokenizer.get_vocab() != other.get_vocab():
            raise ValueError(
                f"observer {self.observer!r} and performer {self.performer!r} have different "
                "tokenizer vocabularies; Binoculars needs two models with the same tokenizer "
                "(e.g. a base model and its instruct version)")

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _encode(self, texts: Sequence[Any]) -> list[np.ndarray]:
        """Token ids for each text (compact arrays, at most `_n_text` long)."""
        clip = self._n_text * _MAX_CHARS_PER_TOKEN
        out: list[np.ndarray] = []
        chunk = 1024
        for lo in range(0, len(texts), chunk):
            part = [_clean_text(t, clip) for t in texts[lo : lo + chunk]]
            enc = self.tokenizer(
                part,
                add_special_tokens=False,
                truncation=True,
                max_length=self._n_text,
                return_attention_mask=False,
            )
            out += [np.asarray(ids[: self._n_text], dtype=np.int64) for ids in enc["input_ids"]]
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_batch(self, batch: list[np.ndarray]) -> list[tuple[Any, Any, Any]]:
        """Forward one batch; return (obs_nll, cross, perf_nll) arrays per text."""
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

        obs = self.observer_model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
        perf = None
        if self.performer_model is not None:
            perf = self.performer_model(
                input_ids=input_ids, attention_mask=attn, use_cache=False
            ).logits
        # Scored positions: predictions at 0..len-1 for targets at 1..len.
        rows, cols = attn[:, 1:].nonzero(as_tuple=True)  # row-major = per-text order
        targets = input_ids[rows, cols + 1]
        width = obs.shape[-1] if perf is None else max(obs.shape[-1], perf.shape[-1])
        step = max(1, _CHUNK_ELEMENTS // width)
        parts: list[list[Any]] = [[], [], []]
        for lo in range(0, rows.numel(), step):
            r, c = rows[lo : lo + step], cols[lo : lo + step]
            stats = _bino_token_stats(obs[r, c], None if perf is None else perf[r, c],
                                      targets[lo : lo + step])
            for acc, s in zip(parts, stats):
                if s is not None:
                    acc.append(s)
        del obs, perf
        splits = np.cumsum(lengths)[:-1]
        arrays = [np.split(torch.cat(p).cpu().numpy(), splits) if p else [None] * len(batch)
                  for p in parts]
        return list(zip(*arrays))

    def features(self, texts: Sequence[Any], progress: bool = False) -> np.ndarray:
        """Feature matrix of shape (len(texts), len(BINO_FEATURES)), float32, input order."""
        texts = list(texts)
        n = len(texts)
        out = np.full((n, len(BINO_FEATURES)), np.nan, dtype=np.float32)
        if n == 0:
            return out
        self._load()
        import torch

        ids = self._encode(texts)
        lengths = np.fromiter((len(x) for x in ids), np.int64, n)
        # Longest first: padding is minimal and any OOM shows up immediately.
        order = [int(i) for i in np.argsort(-lengths, kind="stable") if lengths[i] > 0]

        done = n - len(order)  # empty texts are already NaN rows
        next_report = 0.0
        start = time.perf_counter()
        with torch.no_grad():
            for lo in range(0, len(order), self.batch_size):
                idx = order[lo : lo + self.batch_size]
                stats = self._score_batch([ids[i] for i in idx])
                for i, (obs_nll, cross, perf_nll) in zip(idx, stats):
                    out[i] = _aggregate(obs_nll, cross, perf_nll)
                done += len(idx)
                if progress and (done / n >= next_report or done == n):
                    rate = done / max(time.perf_counter() - start, 1e-9)
                    print(
                        f"[bino] {done}/{n} texts ({100 * done / n:.0f}%), "
                        f"{rate:.2f} texts/s",
                        flush=True,
                    )
                    next_report = math.floor(done / n * 20 + 1) / 20
        return out
