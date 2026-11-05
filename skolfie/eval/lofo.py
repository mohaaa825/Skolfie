"""
skolfie/eval/lofo.py
---------------------
Leave-one-family-out (LOFO) evaluation: how well does a detector catch a
model *family* it has never seen? The held-out Haiku split answers that for
one family; LOFO repeats it for each family in turn (`scripts/eval_lofo.py`).

For each family F (default `DEFAULT_FAMILIES`: the modern claude,
claude-haiku and gemini):

1. Splits: the base config (`configs/splits_modern.json`) with every
   generator of F (via `generator_families`) added to `held_out_generators`,
   built by `skolfie.eval.splits.make_splits`, so `check_guarantees` runs on
   every one. The base held-out generators stay held out (holding out
   `claude-haiku`, already held out there, reproduces the base splits).
2. Train a fresh scorer (by default the feature model, which reuses the
   feature cache) on `train` with `run_evaluation`, so oversampling,
   `max_train` / `max_eval` and `keep_sources` work exactly as in
   `scripts/evaluate.py`. Only what the numbers need is scored: all of
   `val`, F's AI essays in `test_ood_generator`, and the PERSUADE human
   essays in `test`.
3. Cut-off: chosen on that run's `val` so that at most `max_fpr` (1%) of its
   human texts score above it, the rule of `skolfie.models.stacker`'s
   `operating_points` (ai_threshold). Note: this is the single member's
   P(ai), not the stacked model's.
4. Reported for F: recall at that cut-off, AUROC of F's essays against the
   PERSUADE test humans, mean P(ai); and the PERSUADE test humans' false
   positive rate at the cut-off.

Shared topics and PERSUADE essays. PERSUADE human essays are never grouped
with anything (their group key is their own id), so holding out a family
can never move one of them out of `train`: their primary split is the same
in every LOFO run (the result records the count per run). AI essays of F on
a PERSUADE assignment (`shared_topic_groups` "fresh:persuade-") are
detached from their topic group instead of pulling the other models'
essays on that topic into `test`. AI essays of F on an AI-only topic (the
Claude college "bank" topics) still taint their group: with the shipped
essays those groups hold only Claude / Claude-Haiku essays, and Gemini's 27
school topics are its own, so no other family's training essays are lost
(`fresh_train_rows` in the result shows what is left per run).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from skolfie.eval.metrics import auroc
from skolfie.eval.splits import SplitConfig, assign_splits, make_splits

DEFAULT_FAMILIES = ("claude", "claude-haiku", "gemini")
LOFO_SPLITS = ("val", "test", "test_ood_generator")
LOFO_SCHEMA = 1
PERSUADE_SOURCE = "persuade"


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def family_generators(config: SplitConfig, family: str,
                      corpus_generators: Sequence[str] = ()) -> list[str]:
    """Every generator of `family`: mapped ones from the config, plus unmapped
    corpus generators that are their own family (sorted)."""
    gens = {g for g, f in config.generator_families.items() if f == family}
    gens |= {str(g) for g in corpus_generators if config.family(str(g)) == family}
    return sorted(gens)


def lofo_config(base: SplitConfig, family: str,
                corpus_generators: Sequence[str] = ()) -> SplitConfig:
    """`base` with every generator of `family` also held out."""
    gens = family_generators(base, family, corpus_generators) or [family]
    held = list(base.held_out_generators) + [g for g in gens
                                             if g not in base.held_out_generators]
    return dataclasses.replace(base, held_out_generators=held)


def ai_families(df: pd.DataFrame, config: SplitConfig) -> list[str]:
    """Families with AI rows in the corpus."""
    gens = df.loc[df["label"] == "ai", "generator"].astype(str).unique()
    return sorted({config.family(g) for g in gens})


def lofo_splits(df: pd.DataFrame, base: SplitConfig,
                family: str) -> tuple[SplitConfig, pd.DataFrame]:
    """(config, checked (id, split) memberships) with `family` held out."""
    config = lofo_config(base, family, df["generator"].astype(str).unique())
    return config, make_splits(df, config)  # make_splits runs check_guarantees


def family_mask(df: pd.DataFrame, config: SplitConfig, family: str) -> np.ndarray:
    gens = df["generator"].astype(str)
    return ((df["label"] == "ai") & gens.map(config.family).eq(family)).to_numpy()


def persuade_human_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["label"] == "human")
            & (df["source"].astype(str) == PERSUADE_SOURCE)).to_numpy()


def scoring_memberships(df: pd.DataFrame, memberships: pd.DataFrame, config: SplitConfig,
                        family: str) -> pd.DataFrame:
    """The memberships a LOFO run fits and scores: all of train and val,
    `family`'s AI rows in test_ood_generator and the PERSUADE humans in test."""
    fam_ids = set(df.loc[family_mask(df, config, family), "id"].astype(str))
    human_ids = set(df.loc[persuade_human_mask(df), "id"].astype(str))
    split, ids = memberships["split"], memberships["id"].astype(str)
    keep = (split.isin(["train", "val"])
            | ((split == "test_ood_generator") & ids.isin(fam_ids))
            | ((split == "test") & ids.isin(human_ids)))
    return memberships[keep.to_numpy()].reset_index(drop=True)


def split_counts(df: pd.DataFrame, memberships: pd.DataFrame,
                 config: SplitConfig) -> dict[str, Any]:
    """Train rows per source and per AI family, and the PERSUADE humans per split."""
    train_ids = memberships.loc[memberships["split"] == "train", "id"]
    train = df[df["id"].isin(train_ids)]
    fresh = train[train["source"].astype(str) == "fresh"]
    persuade = df[persuade_human_mask(df)]
    by_split = memberships[memberships["id"].isin(persuade["id"])]["split"].value_counts()
    return {
        "train_rows_by_source": {str(k): int(v) for k, v in
                                 train["source"].astype(str).value_counts().sort_index().items()},
        "fresh_train_rows": {str(k): int(v) for k, v in
                             fresh["generator"].astype(str).map(config.family)
                             .value_counts().sort_index().items()},
        "persuade_humans_by_split": {str(k): int(v) for k, v in by_split.sort_index().items()},
    }


def persuade_primary_unchanged(df: pd.DataFrame, base: SplitConfig, config: SplitConfig) -> bool:
    """True when every PERSUADE human essay has the same primary split as under `base`."""
    mask = persuade_human_mask(df)
    if not mask.any():
        return True
    return bool((assign_splits(df, base)[mask] == assign_splits(df, config)[mask]).all())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ai_cutoff(y_val: Sequence[Any], p_val: Sequence[float], max_fpr: float = 0.01) -> float:
    """The stacker's ai_threshold rule on a validation split."""
    from skolfie.models.stacker import operating_points

    return operating_points(y_val, p_val, max_fpr=max_fpr)["ai_threshold"]


def family_metrics(y_val: Sequence[Any], p_val: Sequence[float], p_family: Sequence[float],
                   p_human: Sequence[float], max_fpr: float = 0.01) -> dict[str, Any]:
    """Cut-off from val; recall / AUROC / mean P(ai) of the family's essays
    and the human texts' FPR at the cut-off."""
    cut = ai_cutoff(y_val, p_val, max_fpr)
    pf = np.asarray(p_family, dtype=np.float64)
    ph = np.asarray(p_human, dtype=np.float64)
    nan = math.nan
    y = np.concatenate([np.ones(pf.size, dtype=int), np.zeros(ph.size, dtype=int)])
    return {
        "ai_threshold": float(cut),
        "max_fpr": max_fpr,
        "n_val": int(len(p_val)),
        "n_family": int(pf.size),
        "n_persuade_human": int(ph.size),
        "recall": float((pf > cut).mean()) if pf.size else nan,
        "auroc_vs_persuade": auroc(y, np.concatenate([pf, ph])) if pf.size and ph.size else nan,
        "mean_p_ai": float(pf.mean()) if pf.size else nan,
        "persuade_fpr": float((ph > cut).mean()) if ph.size else nan,
    }


# --------------------------------------------------------------------------- #
# One run per family
# --------------------------------------------------------------------------- #
def _oversample_present(oversample: dict[str, int] | None, df: pd.DataFrame,
                        memberships: pd.DataFrame, log: Callable[[str], None]) -> dict[str, int]:
    """Drop oversample sources with no train rows in this run (logged)."""
    if not oversample:
        return {}
    train_ids = memberships.loc[memberships["split"] == "train", "id"]
    present = set(df.loc[df["id"].isin(train_ids), "source"].astype(str))
    out = {k: v for k, v in oversample.items() if k in present}
    for k in sorted(set(oversample) - set(out)):
        log(f"[lofo] no train rows of source {k!r} in this run: not oversampled")
    return out


def run_family(make_scorer: Callable[[], Any], corpus: pd.DataFrame, base: SplitConfig,
               family: str, *, max_fpr: float = 0.01, oversample: dict[str, int] | None = None,
               max_train: int | None = None, max_eval: int | None = None,
               keep_sources: Sequence[str] | None = None, seed: int = 0,
               log: Callable[[str], None] = print) -> tuple[dict[str, Any], dict[str, Any],
                                                              pd.DataFrame]:
    """Train a fresh scorer with `family` held out. Returns (row, report, predictions).

    `corpus` is a `skolfie.eval.report.load_corpus` frame plus `n_words`.
    """
    from skolfie.eval.report import run_evaluation

    t0 = time.time()
    config, memberships = lofo_splits(corpus, base, family)
    counts = split_counts(corpus, memberships, config)
    unchanged = persuade_primary_unchanged(corpus, base, config)
    held = family_generators(base, family, corpus["generator"].astype(str).unique())
    log(f"[lofo] {family}: holding out {', '.join(held) or family}; train has "
        + ", ".join(f"{k} {v:,}" for k, v in counts["train_rows_by_source"].items()))
    scored = scoring_memberships(corpus, memberships, config, family)
    report, preds = run_evaluation(
        make_scorer(), corpus, scored, model_spec=None, max_train=max_train,
        max_eval=max_eval, seed=seed, eval_splits=LOFO_SPLITS,
        oversample=_oversample_present(oversample, corpus, memberships, log) or None,
        keep_sources=keep_sources, log=log)
    fam = set(corpus.loc[family_mask(corpus, config, family), "id"])
    human = set(corpus.loc[persuade_human_mask(corpus), "id"])
    val = preds[preds["split"] == "val"]
    is_fam = (preds["split"] == "test_ood_generator") & preds["id"].isin(fam)
    p_hum = preds.loc[(preds["split"] == "test") & preds["id"].isin(human), "p_ai"]
    row = {"family": family, "held_out_generators": held,
           **family_metrics(val["label"].tolist(), val["p_ai"], preds.loc[is_fam, "p_ai"],
                            p_hum, max_fpr)}
    scored_gens = corpus.loc[corpus["id"].isin(preds.loc[is_fam, "id"]), "generator"]
    row["family_generators_scored"] = sorted(scored_gens.astype(str).unique().tolist())
    row.update(n_train=report["train"]["n"], persuade_split_unchanged=unchanged,
               all_held_out_generators=list(config.held_out_generators),
               seconds=round(time.time() - t0, 1), **counts)
    log(f"[lofo] {family}: recall {_pct(row['recall'])} at cut-off "
        f"{row['ai_threshold']:.4f}, AUROC {_num(row['auroc_vs_persuade'])}, mean P(ai) "
        f"{_num(row['mean_p_ai'])}, PERSUADE FPR {_pct(row['persuade_fpr'])}")
    return row, report, preds


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _pct(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _num(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def render_lofo_md(result: dict[str, Any]) -> str:
    """Markdown table, one row per held-out family."""
    from skolfie.eval.report import _table

    rows = [[r["family"], ", ".join(r["held_out_generators"]), f"{r['n_family']:,}",
             _pct(r["recall"]), _num(r["auroc_vs_persuade"]), _num(r["mean_p_ai"]),
             f"{r['n_persuade_human']:,}", _pct(r["persuade_fpr"]), f"{r['ai_threshold']:.4f}"]
            for r in result["families"]]
    lines = [
        "# Leave-one-family-out evaluation", "",
        f"- model: `{result['model']}`, splits config: `{result['config']}`",
        f"- base held-out generators: {', '.join(result['base_held_out_generators']) or 'none'}",
        f"- cut-off per run: at most {result['max_fpr']:.0%} of that run's `val` human texts "
        "above it (the stacker's AI rule, here on the single model's P(ai))",
        "- recall / mean P(ai): the held-out family's essays (`test_ood_generator`); AUROC: "
        "those essays vs the PERSUADE human essays in `test`; FPR: those humans above the "
        "cut-off", "",
    ]
    lines += _table(["held-out family", "generators", "essays", "recall", "AUROC",
                     "mean P(ai)", "PERSUADE humans", "human FPR", "cut-off"], rows)
    changed = [r["family"] for r in result["families"] if not r["persuade_split_unchanged"]]
    lines += ["", "PERSUADE human essays kept their splits in every run."
              if not changed else f"**Warning:** PERSUADE human essays moved for: "
              f"{', '.join(changed)}."]
    return "\n".join(lines) + "\n"
