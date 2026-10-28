"""
skolfie/models/transformer_model.py
------------------------------------
Model A of the roadmap: DistilBERT fine-tuned together with the hand-made
features (Annepaka-style fusion).

Network, trained end to end with binary cross-entropy on P(ai):

    text  -> tokenizer -> DistilBERT -> first-token ([CLS]) hidden state (768)
    style [+ GPT-2] features -> standardise -> Linear(n, 64) -> GELU -> dropout
    concat(768 + 64) -> dropout -> Linear(., 1) -> logit of P(ai)

`use_features=False` drops the feature branch (name "transformer-text").

Chunk training (`chunk_augment=k`, name "transformer-chunks"): besides each
whole training text, the model also trains on up to k of its chunks
(`skolfie.mixed.chunking.chunk_spans`, about `chunk_words` words of whole
sentences), labelled like the text. The Mixed detector (roadmap step 7)
scores such chunks one by one, and a model that only saw whole texts scores
them noisily. Only texts with two or more chunks add any, and the held-out
texts are split off first, so none of their chunks are trained on.
The default backbone is `distilbert-base-cased`: case, punctuation and stop
words are kept, since they carry much of the style signal.
fit() and predict_proba() first map every text (and every training chunk)
through `skolfie.robust.normalize.normalize_for_scoring`, which undoes
invisible characters and look-alike letters, also for models pickled
before it existed.

Features: the label-free style (+ optional GPT-2) features of
`FeatureModel.base_features()`, disk-cached by text hash when `cache_dir` is
given. No Delta features (they need labels, hence out-of-fold fitting).
Standardisation is fitted on the training texts: non-finite -> column
median, then (x - mean) / max(std, 1e-6), clipped to [-10, 10]. GPT-2 is
dropped from GPU memory once the training features exist and is reloaded
lazily by `FeatureModel` at predict time.

Training (GPU; the loop also runs on CPU, without fp16):

    * AdamW, two learning rates: `lr` for the backbone, `head_lr` for the
      feature MLP and the output layer; no weight decay on biases and
      LayerNorm weights (all 1-D parameters). Linear warmup over
      `warmup_ratio` of the optimisation steps, then linear decay to 0.
      Gradient accumulation (`grad_accum`), gradient clipping at 1.0.
    * fp16 on CUDA by default: the weights stay float32, the forward pass
      runs under torch.autocast with a GradScaler (no dtype= / torch_dtype=
      loading kwargs, which differ between transformers 4.x and 5.x).
    * Texts are tokenised per batch (head of the text, truncated to
      `max_length` tokens, padded to the longest text in the batch). Batches
      are length-bucketed to cut padding: shuffle, sort by character length
      within chunks of 100 batches, cut into batches, shuffle the batch order.
    * `eval_fraction` of the training texts (stratified, seeded, at least one
      per class when possible) is held out. Every `eval_every` optimisation
      steps and at the end of each epoch its log-loss and AUROC are logged;
      the weights with the lowest held-out log-loss are kept in CPU memory and
      restored at the end. The harness's own validation split stays untouched.

Inference sorts texts by length, batches them, runs under
torch.inference_mode (+ autocast on CUDA) and restores the input order, so
results do not depend on the order or batching of the input.

Pickling (joblib): the network is stored as a CPU state dict, the backbone
as its config dict and the tokenizer as its saved files ({name: bytes}), so a
saved model reloads without internet access; the network is rebuilt on the
first predict after loading. The inner FeatureModel drops its GPT-2 weights
and caches itself (GPT-2 is reloaded from `lm_model` when needed).

The scorer interface matches `skolfie.models.baselines` (fit /
predict_proba / params). torch and transformers are imported lazily.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from skolfie.eval.metrics import to_binary
from skolfie.models.feature_model import FeatureModel, scoring_texts
from skolfie.robust.normalize import normalize_for_scoring

FEATURE_HIDDEN = 64
DROPOUT = 0.1
CLIP_GRAD_NORM = 1.0
FEATURE_CLIP = 10.0
STD_FLOOR = 1e-6
SORT_CHUNK_BATCHES = 100
PREDICT_BATCH_FACTOR = 4
LOSS_EMA = 0.98
# Texts are clipped to this many characters per allowed token before
# tokenising, only to bound tokenizer cost on pathological input; normal text
# never has tokens this long, so the tokens actually used are unchanged.
_MAX_CHARS_PER_TOKEN = 64


# --------------------------------------------------------------------------- #
# Small helpers (module level so tests can check them directly)
# --------------------------------------------------------------------------- #
def fit_standardizer(X: np.ndarray) -> dict[str, np.ndarray]:
    """Column median (NaN fill), mean and floored std of a feature matrix."""
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        median = np.nanmedian(X, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isnan(X), median, X)
    mean = filled.mean(axis=0)
    std = np.maximum(filled.std(axis=0), STD_FLOOR)
    return {"median": median, "mean": mean, "std": std}


def standardize(X: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply `fit_standardizer` stats: fill non-finite, z-score, clip; float32."""
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, stats["median"])
    Z = (X - stats["mean"]) / stats["std"]
    return np.clip(Z, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)


def holdout_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, eval_idx): stratified, seeded; >= 1 per class when the class has >= 2."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    parts = []
    if fraction > 0:
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            k = min(max(int(round(fraction * idx.size)), 1), idx.size - 1)
            if k > 0:
                parts.append(rng.choice(idx, size=k, replace=False))
    eval_idx = np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
    keep = np.ones(y.size, dtype=bool)
    keep[eval_idx] = False
    return np.flatnonzero(keep), eval_idx.astype(np.int64)


def length_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator,
                   chunk_batches: int = SORT_CHUNK_BATCHES) -> list[np.ndarray]:
    """Positions into `lengths`, cut into length-bucketed batches in random order."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * chunk_batches
    batches: list[np.ndarray] = []
    for lo in range(0, len(order), chunk):
        part = order[lo:lo + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def _lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1 over `warmup` steps, then linear decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / (warmup + 1)
    return max(0.0, (total - step) / max(1, total - warmup))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    from scipy.special import expit

    return np.clip(expit(np.asarray(z, dtype=np.float64)), 0.0, 1.0)


def _fmt_seconds(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _grad_scaler(enabled: bool) -> Any:
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #
class TransformerModel:
    pretrained = False

    def __init__(self, model_name: str = "distilbert-base-cased", use_features: bool = True,
                 use_lm: bool = True, max_length: int = 512, epochs: int = 1,
                 batch_size: int = 16, grad_accum: int = 2, lr: float = 3e-5,
                 head_lr: float = 1e-3, weight_decay: float = 0.01, warmup_ratio: float = 0.06,
                 eval_fraction: float = 0.02, eval_every: int | None = 2000,
                 device: str | None = None, fp16: bool | None = None,
                 cache_dir: str | os.PathLike[str] | None = None, lm_model: str = "gpt2",
                 lm_max_tokens: int = 512, lm_batch_size: int = 32, n_jobs: int = -1,
                 seed: int = 0, backbone: Any = None, tokenizer: Any = None,
                 log: Callable[[str], None] = print, lm_device: str | None = None,
                 chunk_augment: int = 0, chunk_words: int = 60) -> None:
        if (backbone is None) != (tokenizer is None):
            raise ValueError("inject both `backbone` and `tokenizer`, or neither")
        for name, value in (("max_length", max_length), ("epochs", epochs),
                            ("batch_size", batch_size), ("grad_accum", grad_accum)):
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if max_length < 2:
            raise ValueError("max_length must be >= 2 (room for [CLS] and [SEP])")
        if int(chunk_augment) < 0:
            raise ValueError(f"chunk_augment must be >= 0, got {chunk_augment}")
        if int(chunk_words) < 1:
            raise ValueError(f"chunk_words must be >= 1, got {chunk_words}")
        if not 0.0 <= eval_fraction < 0.5:
            raise ValueError("eval_fraction must be in [0, 0.5)")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        self.model_name = model_name
        self.use_features = use_features
        self.use_lm = use_lm
        self.max_length = int(max_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.lr = lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.eval_fraction = eval_fraction
        self.eval_every = eval_every
        self.device = device
        self.fp16 = fp16
        self.cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        self.lm_model = lm_model
        self.lm_max_tokens = lm_max_tokens
        self.lm_batch_size = lm_batch_size
        self.lm_device = lm_device
        self.n_jobs = n_jobs
        self.seed = seed
        self.log = log
        self.chunk_augment = int(chunk_augment)
        self.chunk_words = int(chunk_words)
        self.name = "transformer" if use_features else "transformer-text"
        if self.chunk_augment:
            self.name += "-chunks"

        self._features: FeatureModel | None = None
        if use_features:
            self._features = FeatureModel(use_lm=use_lm, cache_dir=self.cache_dir,
                                          lm_model=lm_model, lm_max_tokens=lm_max_tokens,
                                          lm_batch_size=lm_batch_size, lm_device=lm_device,
                                          n_jobs=n_jobs, seed=seed, log=log)
        # Fitted state.
        self.feature_names_: list[str] = []
        self.feature_stats_: dict[str, np.ndarray] | None = None
        self.max_length_: int | None = None
        self.history_: list[dict[str, Any]] = []
        self.best_step_: int | None = None
        self.best_eval_logloss_: float | None = None
        self.best_eval_auroc_: float | None = None
        self.total_steps_ = 0
        self.n_train_ = 0
        self.n_eval_ = 0
        self.n_chunks_ = 0
        self.feature_seconds_: float | None = None
        self.train_seconds_: float | None = None
        # Live objects (never pickled) and their pickled form.
        self._backbone_in = backbone
        self._tokenizer_in = tokenizer
        self._net: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._amp = False
        self._packed: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Pickling: CPU state dict + config dict + tokenizer files.
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["log"] = print
        if self._net is not None:
            state["_packed"] = self._pack()
        for key in ("_backbone_in", "_tokenizer_in", "_net", "_tokenizer", "_device"):
            state[key] = None
        state["_amp"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Models saved before chunk training existed.
        state.setdefault("chunk_augment", 0)
        state.setdefault("chunk_words", 60)
        state.setdefault("n_chunks_", 0)
        self.__dict__.update(state)

    def _pack(self) -> dict[str, Any]:
        backbone = self._net["backbone"]
        config = backbone.config.to_dict()
        model_type = config.pop("model_type", None) or backbone.config.model_type
        net_state = {k: v.detach().cpu().numpy() for k, v in self._net.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            self._tokenizer.save_pretrained(tmp)
            files = {p.relative_to(tmp).as_posix(): p.read_bytes()
                     for p in sorted(Path(tmp).rglob("*")) if p.is_file()}
        return {"model_type": model_type, "config": config, "state": net_state,
                "tokenizer_files": files}

    def _unpack(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        packed = self._packed
        assert packed is not None
        config = AutoConfig.for_model(packed["model_type"], **packed["config"])
        backbone = AutoModel.from_config(config)
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in packed["tokenizer_files"].items():
                path = Path(tmp) / name
                if ".." in Path(name).parts or Path(name).is_absolute():
                    raise ValueError(f"bad tokenizer file name in the saved model: {name!r}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            tokenizer = AutoTokenizer.from_pretrained(tmp, use_fast=True)
        self._setup(backbone, tokenizer)
        self._net.load_state_dict({k: torch.tensor(v) for k, v in packed["state"].items()})
        self._net.eval()
        self._packed = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> None:
        import torch

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        amp = device.type == "cuda" if self.fp16 is None else bool(self.fp16)
        if amp and device.type != "cuda":
            self.log(f"[transformer] fp16 needs CUDA; running in float32 on {device}")
            amp = False
        self._device, self._amp = device, amp

    def _setup(self, backbone: Any, tokenizer: Any) -> None:
        """Wrap backbone + new head into one module on the device; prepare the tokenizer."""
        import torch
        from torch import nn

        self._resolve_device()
        if tokenizer.pad_token is None:
            fallback = tokenizer.sep_token or tokenizer.eos_token or tokenizer.unk_token
            if fallback is None:
                raise ValueError("tokenizer has no pad token and nothing to fall back on")
            tokenizer.pad_token = fallback
        # Right padding keeps the [CLS] token at position 0; right truncation
        # keeps the head of the text.
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        self._tokenizer = tokenizer

        config = backbone.config
        hidden = getattr(config, "hidden_size", None) or getattr(config, "dim")
        n_pos = getattr(config, "max_position_embeddings", None)
        self.max_length_ = min(self.max_length, int(n_pos)) if n_pos else self.max_length

        torch.manual_seed(self.seed)  # reproducible head initialisation
        modules: dict[str, Any] = {"backbone": backbone.float(), "dropout": nn.Dropout(DROPOUT)}
        width = int(hidden)
        if self.use_features:
            modules["features"] = nn.Sequential(
                nn.Linear(len(self.feature_names_), FEATURE_HIDDEN), nn.GELU(),
                nn.Dropout(DROPOUT))
            width += FEATURE_HIDDEN
        modules["head"] = nn.Linear(width, 1)
        self._net = nn.ModuleDict(modules).to(self._device)

    def _ensure_net(self) -> None:
        if self._net is None:
            if self._packed is None:
                raise RuntimeError("TransformerModel is not fitted")
            self._unpack()

    def _seed_everything(self) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #
    def _raw_features(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        assert self._features is not None
        return self._features.base_features(texts)

    def _release_lm(self) -> None:
        """Free GPT-2 (FeatureModel reloads it lazily at predict time)."""
        if self._features is None or self._features._lm is None:
            return
        import gc

        import torch

        self._features._lm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _autocast(self) -> Any:
        import torch

        if self._amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _encode(self, texts: list[str]) -> tuple[Any, Any]:
        clip = self.max_length_ * _MAX_CHARS_PER_TOKEN
        enc = self._tokenizer([t[:clip] for t in texts], truncation=True,
                              max_length=self.max_length_, padding=True,
                              pad_to_multiple_of=8 if self._amp else None,
                              return_tensors="pt")
        dev = self._device
        return (enc["input_ids"].to(dev, non_blocking=True),
                enc["attention_mask"].to(dev, non_blocking=True))

    def _forward(self, input_ids: Any, attention_mask: Any, feats: Any) -> Any:
        import torch

        out = self._net["backbone"](input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        parts = [hidden[:, 0]]
        if feats is not None:
            parts.append(self._net["features"](feats))
        h = torch.cat(parts, dim=-1)
        return self._net["head"](self._net["dropout"](h)).squeeze(-1)

    def _logits(self, texts: list[str], feats: np.ndarray | None) -> np.ndarray:
        """Logits in input order (length-sorted batches, eval mode, no grad)."""
        import torch

        n = len(texts)
        out = np.empty(n, dtype=np.float64)
        bs = self.batch_size * PREDICT_BATCH_FACTOR
        lengths = np.fromiter((len(t) for t in texts), np.int64, n)
        order = np.argsort(-lengths, kind="stable")
        self._net.eval()
        with torch.inference_mode(), self._autocast():
            for lo in range(0, n, bs):
                idx = order[lo:lo + bs]
                ids, mask = self._encode([texts[i] for i in idx])
                f = None
                if feats is not None:
                    f = torch.from_numpy(feats[idx]).to(self._device, non_blocking=True)
                out[idx] = self._forward(ids, mask, f).float().cpu().numpy()
        return out

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def _param_groups(self) -> list[dict[str, Any]]:
        groups = []
        for in_backbone, lr in ((True, self.lr), (False, self.head_lr)):
            decay, no_decay = [], []
            for name, p in self._net.named_parameters():
                if not p.requires_grad or name.startswith("backbone.") != in_backbone:
                    continue
                # Biases and LayerNorm weights are the 1-D parameters.
                (no_decay if p.ndim < 2 else decay).append(p)
            groups.append({"params": decay, "lr": lr, "weight_decay": self.weight_decay})
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return [g for g in groups if g["params"]]

    def fit(self, texts: Sequence[str], labels: Sequence[Any]) -> "TransformerModel":
        texts = scoring_texts(texts)
        y = to_binary(labels)
        if len(texts) != len(y):
            raise ValueError("texts and labels must have the same length")
        if np.unique(y).size < 2:
            raise ValueError("need both human and ai training texts")
        self._seed_everything()

        train_idx, eval_idx = holdout_split(y, self.eval_fraction, self.seed)
        if self.chunk_augment:
            texts, y, train_idx = self._add_chunks(texts, y, train_idx)

        start = time.perf_counter()
        feats = None
        if self.use_features:
            X, names = self._raw_features(texts)
            self.feature_names_ = list(names)
            self.feature_stats_ = fit_standardizer(X)
            feats = standardize(X, self.feature_stats_)
            del X
            self._release_lm()
        self.feature_seconds_ = time.perf_counter() - start

        if self._backbone_in is not None:
            backbone, tokenizer = self._backbone_in, self._tokenizer_in
        else:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            backbone = AutoModel.from_pretrained(self.model_name)
        self._packed = None
        self._setup(backbone, tokenizer)

        start = time.perf_counter()
        self._train(texts, y, feats, train_idx, eval_idx)
        self.train_seconds_ = time.perf_counter() - start
        return self

    def _add_chunks(self, texts: list[str], y: np.ndarray,
                    train_idx: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Append up to `chunk_augment` chunks of each training text (same label)."""
        from skolfie.mixed.chunking import chunk_spans

        rng = np.random.default_rng(self.seed)
        min_words = max(1, int(round(self.chunk_words * 25 / 60)))
        extra, extra_y = [], []
        for i in train_idx:
            spans = chunk_spans(texts[i], target_words=self.chunk_words, min_words=min_words)
            if len(spans) < 2:
                continue
            k = min(self.chunk_augment, len(spans))
            for j in sorted(rng.choice(len(spans), size=k, replace=False)):
                a, b = spans[j]
                # Cut from the normalised text and normalised again, like a chunk
                # the Mixed detector cuts from the original text and scores.
                extra.append(normalize_for_scoring(texts[i][a:b]))
                extra_y.append(y[i])
        self.n_chunks_ = len(extra)
        self.log(f"[transformer] chunk training: {len(extra):,} chunks of about "
                 f"{self.chunk_words} words added to {train_idx.size:,} training texts")
        new_idx = np.arange(len(texts), len(texts) + len(extra), dtype=train_idx.dtype)
        return (texts + extra, np.concatenate([y, np.asarray(extra_y, dtype=y.dtype)]),
                np.concatenate([train_idx, new_idx]))

    def _train(self, texts: list[str], y: np.ndarray, feats: np.ndarray | None,
               train_idx: np.ndarray, eval_idx: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F

        net, dev = self._net, self._device
        rng = np.random.default_rng(self.seed)
        lengths = np.fromiter((len(texts[i]) for i in train_idx), np.int64, train_idx.size)
        n_batches = math.ceil(train_idx.size / self.batch_size)
        steps_per_epoch = math.ceil(n_batches / self.grad_accum)
        total = steps_per_epoch * self.epochs
        warmup = int(round(self.warmup_ratio * total))
        self.total_steps_, self.n_train_, self.n_eval_ = total, int(train_idx.size), int(
            eval_idx.size)
        self.history_ = []
        self.best_step_ = self.best_eval_logloss_ = self.best_eval_auroc_ = None

        optimizer = torch.optim.AdamW(self._param_groups(), lr=self.lr, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _lr_factor(s, warmup, total))
        scaler = _grad_scaler(self._amp)
        y_t = torch.from_numpy(y.astype(np.float32))
        feats_t = None if feats is None else torch.from_numpy(feats)
        best_state: dict[str, Any] | None = None

        self.log(f"[transformer] fine-tuning {self.model_name} on {dev}"
                 f"{' (fp16)' if self._amp else ''}: {train_idx.size:,} train / "
                 f"{eval_idx.size:,} held-out texts, {self.epochs} epoch(s), {total:,} steps of "
                 f"{self.batch_size}x{self.grad_accum} texts, max_length {self.max_length_}, "
                 f"{len(self.feature_names_)} features")
        log_every = max(1, total // 100)
        step, seen, ema, last_eval = 0, 0, None, -1
        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            net.train()
            batches = length_batches(lengths, self.batch_size, rng)
            optimizer.zero_grad(set_to_none=True)
            for i, pos in enumerate(batches):
                idx = train_idx[pos]
                group_n = min(self.grad_accum, len(batches) - (i // self.grad_accum)
                              * self.grad_accum)
                ids, mask = self._encode([texts[j] for j in idx])
                idx_t = torch.from_numpy(idx)
                f = None if feats_t is None else feats_t[idx_t].to(dev, non_blocking=True)
                target = y_t[idx_t].to(dev, non_blocking=True)
                with self._autocast():
                    logits = self._forward(ids, mask, f)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
                scaler.scale(loss / group_n).backward()
                seen += idx.size
                value = float(loss.detach())
                ema = value if ema is None else LOSS_EMA * ema + (1 - LOSS_EMA) * value
                if (i + 1) % self.grad_accum and i + 1 != len(batches):
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                if step % log_every == 0 or step == total:
                    elapsed = time.perf_counter() - t0
                    rate = seen / max(elapsed, 1e-9)
                    eta = elapsed / step * (total - step)
                    self.log(f"[transformer] epoch {epoch + 1}/{self.epochs} step {step:,}/"
                             f"{total:,} ({100 * step / total:.0f}%) loss {ema:.4f} | "
                             f"{rate:.1f} texts/s | ETA {_fmt_seconds(eta)}")
                if eval_idx.size and self.eval_every and step % self.eval_every == 0:
                    best_state = self._evaluate(texts, y, feats, eval_idx, step, best_state)
                    last_eval = step
                    net.train()
            if eval_idx.size and last_eval != step:
                best_state = self._evaluate(texts, y, feats, eval_idx, step, best_state)
                last_eval = step

        if best_state is not None and self.best_step_ != step:
            net.load_state_dict(best_state)
            self.log(f"[transformer] restored the best checkpoint (step {self.best_step_:,}, "
                     f"held-out log-loss {self.best_eval_logloss_:.4f})")
        net.eval()

    def _evaluate(self, texts: list[str], y: np.ndarray, feats: np.ndarray | None,
                  eval_idx: np.ndarray, step: int,
                  best_state: dict[str, Any] | None) -> dict[str, Any] | None:
        from sklearn.metrics import log_loss, roc_auc_score

        y_eval = y[eval_idx]
        p = _sigmoid(self._logits([texts[i] for i in eval_idx],
                                  None if feats is None else feats[eval_idx]))
        ll = float(log_loss(y_eval, p, labels=[0, 1]))
        auc = float(roc_auc_score(y_eval, p)) if np.unique(y_eval).size == 2 else math.nan
        self.history_.append({"step": step, "logloss": ll, "auroc": auc})
        better = self.best_eval_logloss_ is None or ll < self.best_eval_logloss_
        if better:
            self.best_step_, self.best_eval_logloss_, self.best_eval_auroc_ = step, ll, auc
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in self._net.state_dict().items()}
        self.log(f"[transformer] step {step:,}: held-out log-loss {ll:.4f}, AUROC {auc:.4f}"
                 f"{' (best)' if better else ''}")
        return best_state

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self._net is None and self._packed is None:
            raise RuntimeError("TransformerModel is not fitted")
        texts = scoring_texts(texts)
        if not texts:
            return np.zeros(0)
        self._ensure_net()
        feats = None
        if self.use_features:
            X, names = self._raw_features(texts)
            if list(names) != self.feature_names_:
                raise RuntimeError("feature columns differ from the ones seen in fit()")
            feats = standardize(X, self.feature_stats_)
        return _sigmoid(self._logits(texts, feats))

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: getattr(self, k) for k in (
            "model_name", "use_features", "max_length", "epochs", "batch_size", "grad_accum",
            "lr", "head_lr", "weight_decay", "warmup_ratio", "eval_fraction", "seed")}
        if self.chunk_augment:
            out.update(chunk_augment=self.chunk_augment, chunk_words=self.chunk_words,
                       n_chunks=self.n_chunks_)
        if self.use_features:
            out["use_lm"] = self.use_lm
            if self.use_lm:
                out.update(lm_model=self.lm_model, lm_max_tokens=self.lm_max_tokens)
        auc = self.best_eval_auroc_
        out.update(
            best_eval_logloss=self.best_eval_logloss_,
            best_eval_auroc=None if auc is None or math.isnan(auc) else auc,
            best_step=self.best_step_, total_steps=self.total_steps_,
            n_train=self.n_train_, n_eval=self.n_eval_, n_features=len(self.feature_names_),
            feature_seconds=self.feature_seconds_, train_seconds=self.train_seconds_,
        )
        return out
