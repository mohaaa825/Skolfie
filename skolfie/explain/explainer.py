"""
skolfie/explain/explainer.py
-----------------------------
Roadmap step 8: one explanation per verdict.

`Explainer(verdict_model, feature_model, chunk_scorer, chunk_rule)` puts
together, for each text:

    label         "human" | "uncertain" | "ai" | "mixed": the verdict, set to
                  "mixed" when a chunk rule is given and says Mixed
    verdict       the verdict model's own label (never "mixed")
    p_ai          the verdict model's P(ai)
    chunks        [{"start", "end", "p", "label"}] per chunk of the text
                  (step 7's `score_texts`), for the heatmap; label is "ai"
                  when p >= `highlight_threshold`
    highlights    the same chunks merged into AI / human spans (step 7's
                  `highlight`)
    top_features  the feature model's top-k SHAP features (see
                  `skolfie.explain.contributions.explain_features`)
    feature_model {"p_ai", "raw_score", "base_value", "agrees"} of the feature
                  model; agrees is False when it alone would say the other
                  side (P(ai) on the other side of 0.5 than an AI / human
                  verdict), so its top features do not explain the verdict,
                  and None for an uncertain verdict
    obfuscation   {"invisible", "lookalikes"}: the `obfuscation_stats` of the
                  text as given, i.e. how many invisible characters and
                  look-alike letters the models' normalisation undid
                  (`skolfie.robust.normalize`); non-zero counts are a hint
                  that the text was disguised to evade detection

Offsets are character offsets into the text, so `text[start:end]` is the
span: chunks are cut from the text as given, and each chunk is normalised
inside the scorers' predict_proba, so the offsets never shift. Everything
is plain Python (floats, ints, strings, None), so the dict is JSON as it
is; step 11's API returns it.

The verdict model is any scorer with `predict_proba(texts)`. Its verdict
comes from its `cutoffs` (a `StackedModel`), else from `predict()` when that
returns "human" / "uncertain" / "ai", else from P(ai) >= 0.5. The SHAP
explanations always come from the LightGBM feature model (Model B): pass it,
or let the Explainer find it as the verdict model itself or as a member of a
`StackedModel`. Chunks are scored with `chunk_scorer` (default: the verdict
model). Models can be passed as objects or as paths to joblib files.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np

VERDICTS = ("human", "uncertain", "ai")
LABELS = ("human", "uncertain", "ai", "mixed")
AI_THRESHOLD = 0.5

Model = Any


def load_model(model: Model) -> Any:
    """A model object as it is, or the object saved in a joblib file (str / PathLike)."""
    if isinstance(model, (str, os.PathLike)):
        from skolfie.models.baselines import JoblibScorer

        return JoblibScorer(model).obj
    return model


def predict_p_ai(model: Any, texts: Sequence[str]) -> np.ndarray:
    """P(ai) per text from `model.predict_proba`, which may return (n,) or (n, 2)."""
    texts = list(texts)
    if not texts:
        return np.zeros(0)
    out = np.asarray(model.predict_proba(texts), dtype=np.float64)
    if out.ndim == 2 and out.shape[1] == 2:
        classes = [str(c).lower() for c in getattr(model, "classes_", [])]
        out = out[:, classes.index("ai") if "ai" in classes else 1]
    if out.shape != (len(texts),):
        raise ValueError(f"predict_proba returned shape {out.shape} for {len(texts)} texts")
    return out


def find_feature_model(model: Any) -> Any:
    """The LightGBM FeatureModel in `model` (itself, or a StackedModel member), else None."""
    from skolfie.models.feature_model import FeatureModel

    if isinstance(model, FeatureModel):
        return model
    members = getattr(model, "members", None)
    if isinstance(members, dict) and callable(getattr(model, "_member", None)):
        for name in members:
            obj = getattr(model._member(name), "obj", None)
            if isinstance(obj, FeatureModel):
                return obj
    return None


def feature_model_agrees(verdict: str, feature_p_ai: float) -> Optional[bool]:
    """Whether the feature model alone lands on the verdict's side (None: no side)."""
    if verdict not in ("human", "ai"):
        return None
    return (feature_p_ai >= AI_THRESHOLD) == (verdict == "ai")


class _ChunkScorer:
    """Adapter for `score_texts`: P(ai) as (n,), reusing scores already known by text."""

    def __init__(self, model: Any, known: Optional[dict[str, float]] = None) -> None:
        self.model = model
        self.known = dict(known or {})

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self.known))
        if missing:
            self.known.update(zip(missing, predict_p_ai(self.model, missing).tolist()))
        return np.array([self.known[t] for t in texts], dtype=np.float64)


class Explainer:
    """Verdict + chunk heatmap + top SHAP features per text (see the module docstring)."""

    def __init__(self, verdict_model: Model, feature_model: Model = None,
                 chunk_scorer: Model = None, chunk_rule: Any = None, top_k: int = 5,
                 reference: Optional[dict[str, dict[str, Optional[float]]]] = None,
                 chunker: Optional[Callable[[str], Sequence[tuple[int, int]]]] = None,
                 highlight_threshold: float = AI_THRESHOLD, batch_size: int = 64) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.verdict_model = load_model(verdict_model)
        if feature_model is None:
            self.feature_model = find_feature_model(self.verdict_model)
            if self.feature_model is None:
                raise ValueError("no feature model found in the verdict model; pass "
                                 "feature_model (e.g. the path to features.joblib)")
        else:
            self.feature_model = load_model(feature_model)
        self.chunk_scorer = (self.verdict_model if chunk_scorer is None
                             else load_model(chunk_scorer))
        self.chunk_rule = chunk_rule
        self.top_k = int(top_k)
        self.reference = reference
        self.chunker = chunker
        self.highlight_threshold = float(highlight_threshold)
        self.batch_size = int(batch_size)

    def info(self) -> dict[str, Any]:
        """What produced the explanations (for the header of a saved report)."""
        def name(m: Any) -> str:
            return str(getattr(m, "name", None) or type(m).__name__)

        cutoffs = getattr(self.verdict_model, "cutoffs", None)
        rule = self.chunk_rule
        return {
            "verdict_model": name(self.verdict_model),
            "cutoffs": {k: float(v) for k, v in cutoffs.items()} if isinstance(cutoffs, dict)
            else None,
            "feature_model": name(self.feature_model),
            "chunk_model": name(self.chunk_scorer),
            "chunk_rule": None if rule is None else {
                "type": type(rule).__name__, **{k: v for k, v in vars(rule).items()}},
            "top_k": self.top_k,
            "highlight_threshold": self.highlight_threshold,
            "has_reference": self.reference is not None,
        }

    # ------------------------------------------------------------------ #
    def _verdicts(self, texts: list[str], p: np.ndarray) -> list[str]:
        model = self.verdict_model
        cutoffs = getattr(model, "cutoffs", None)
        if isinstance(cutoffs, dict) and {"ai_threshold", "human_threshold"} <= set(cutoffs):
            from skolfie.models.stacker import verdicts

            return [str(v) for v in verdicts(p, cutoffs)]
        if callable(getattr(model, "predict", None)):
            out = [str(v).lower() for v in model.predict(texts)]
            if len(out) == len(texts) and set(out) <= set(VERDICTS):
                return out
        return ["ai" if x >= AI_THRESHOLD else "human" for x in p]

    def _explain_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        from skolfie.eval.report import _json_safe
        from skolfie.explain.contributions import explain_features
        from skolfie.mixed.detector import highlight, score_texts
        from skolfie.robust.normalize import obfuscation_stats

        p = predict_p_ai(self.verdict_model, texts)
        verdicts = self._verdicts(texts, p)
        known = (dict(zip(texts, p.tolist())) if self.chunk_scorer is self.verdict_model
                 else None)
        scored = score_texts(_ChunkScorer(self.chunk_scorer, known), texts,
                             chunker=self.chunker, batch_size=max(self.batch_size, 256),
                             log=None)
        rule = (self.chunk_rule.predict(scored) if self.chunk_rule is not None
                else [None] * len(texts))
        feats = explain_features(self.feature_model, texts, top_k=self.top_k,
                                 reference=self.reference)

        out = []
        for i, text in enumerate(texts):
            chunks = scored[i]["chunks"]
            out.append(_json_safe({
                "label": "mixed" if rule[i] == "mixed" else verdicts[i],
                "verdict": verdicts[i],
                "p_ai": float(p[i]),
                "chunks": [{"start": int(s), "end": int(e), "p": float(cp),
                            "label": "ai" if cp >= self.highlight_threshold else "human"}
                           for s, e, cp in chunks],
                "highlights": highlight(text, chunks, threshold=self.highlight_threshold),
                "top_features": feats[i]["top_features"],
                "feature_model": {
                    **{k: feats[i][k] for k in ("p_ai", "raw_score", "base_value")},
                    "agrees": feature_model_agrees(verdicts[i], feats[i]["p_ai"])},
                "obfuscation": obfuscation_stats(text),
            }))
        return out

    def explain_many(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """One explanation dict per text, in input order (scored in batches)."""
        texts = ["" if t is None else str(t) for t in texts]
        out: list[dict[str, Any]] = []
        for lo in range(0, len(texts), self.batch_size):
            out += self._explain_batch(texts[lo:lo + self.batch_size])
        return out

    def explain(self, text: str) -> dict[str, Any]:
        return self.explain_many([text])[0]
