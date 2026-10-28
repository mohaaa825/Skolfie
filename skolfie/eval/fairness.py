"""
skolfie/eval/fairness.py
-------------------------
Roadmap step 10: how often is human text wrongly called AI, per subgroup?

A detector that is fine on average can still be unfair to a group: GPT
detectors flag essays by non-native English writers far more often than
essays by native writers, because their plainer vocabulary and syntax make
them more predictable to a language model (low perplexity; Liang et al.
2023). Short and technical texts are other known trouble spots. This module
measures it and turns what it finds into user-facing caveats.

Rates
    A text is flagged when P(ai) > threshold (the stacker's own rule for an
    AI verdict, `skolfie.models.stacker.verdicts`). `group_rates()` gives per
    group the count n, the flagged count k, the rate k / n and its 95% Wilson
    interval (`wilson_interval`). On human texts that rate is the false-
    positive rate (FPR); on AI texts it is the recall; on human texts that an
    AI later polished or simplified ("AI-edited") it is just the share
    flagged as AI, reported apart and never counted in a human FPR.

Flag rule (`FlagRule`, `flag_groups`)
    The reference is the FPR on all corpus human texts of the evaluation
    splits. A human group is

        too_small        n < min_n (default 30): too few texts to judge,
                         whatever its rate
        far_above        rate >= min_ratio x reference (default 2x) AND
                         rate >= reference + min_excess (default 2 pp) AND
                         its Wilson lower bound > reference: clearly worse
        above_uncertain  the first two hold but the interval still reaches
                         the reference: maybe worse, not shown yet
        ok               otherwise

    The ratio catches "twice as often" when the base rate is small; the
    absolute margin keeps 0.4% vs 0.2% from counting; the interval keeps a
    handful of unlucky texts from counting. Only `far_above` groups become
    known limitations.

Groupings
    Corpus human texts of the evaluation splits (`corpus_groupings`): by
    split, domain, source, length bucket (`LENGTH_BUCKETS`) and, for the
    technical domain, by length. External fairness sets
    (`external_groupings`; built by `scripts/build_fairness_sets.py`, columns
    `set`, `group`, `writer`, `edited_by_ai`, `proficiency`, `grade`): human
    texts by set, by writer (native / non-native / unknown), by length, and
    ELLIPSE essays by proficiency band (`proficiency_band`) and by grade;
    AI-edited texts by set; AI texts by set (recall).

Why (`group_feature_medians`)
    Optionally, per group the median of a few label-free features
    (`KEY_FEATURES`: GPT-2 log-perplexity, vocabulary variety, word and
    sentence length) next to the corpus human and AI medians, to see whether
    a flagged group writes "like AI" on the features the detector uses.

`known_limitations()` turns the `far_above` groups of user-facing groupings
into short sentences for the app. Not user-facing: dataset sources and split
names (meaningless to a user) and the external sets by length (confounded
with the set; the corpus length buckets cover length).
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

Z_95 = 1.959963984540054

# (label, lowest n_words, highest n_words or None); half-open word-count buckets.
LENGTH_BUCKETS = (("<50", 0, 49), ("50-99", 50, 99), ("100-199", 100, 199),
                  ("200-399", 200, 399), ("400+", 400, None))
LENGTH_LABELS = tuple(label for label, _, _ in LENGTH_BUCKETS)

# ELLIPSE "Overall" scores run from 1 to 5 in steps of 0.5. Fixed bands, not
# quantiles, so a band means the same in every run and on every sample:
# low 1-2, middle 2.5-3, high 3.5-5 (cut at 2.25 and 3.25 so averaged
# scores fall into the nearest band). Most essays score near 3, so the low
# band is the smallest.
PROFICIENCY_BANDS = (("1-2 (low)", -math.inf, 2.25), ("2.5-3 (middle)", 2.25, 3.25),
                     ("3.5-5 (high)", 3.25, math.inf))
PROFICIENCY_LABELS = tuple(label for label, _, _ in PROFICIENCY_BANDS)

KEY_FEATURES = ("lm_mean_nll", "sty_mattr50", "sty_word_len_mean", "sty_sent_len_mean")
FEATURE_TITLES = {
    "lm_mean_nll": "GPT-2 log-perplexity",
    "sty_mattr50": "vocabulary variety (MATTR-50)",
    "sty_word_len_mean": "mean word length",
    "sty_sent_len_mean": "mean sentence length",
}

TECHNICAL_DOMAIN = "technical"
EXTERNAL_COLUMNS = ("set", "group", "writer", "edited_by_ai")
WRITERS = ("non-native", "native", "unknown", "ai")
ELLIPSE_SET = "ellipse"
STATUSES = ("far_above", "above_uncertain", "ok", "too_small")

# Built-in descriptions of the external sets' sources, used when the fairness
# sets have no manifest that names them.
DEFAULT_SOURCES = (
    {"name": "GPT detector bias data (Liang et al. 2023)",
     "sets": "toefl_real, toefl_gpt4polished, college_real, cs224n_real, hewlett_real, "
             "hewlett_gptsimplify, college_gpt3, college_gpt3prompteng, cs224n_gpt3, "
             "cs224n_gpt3prompteng",
     "citation": "Liang, Yuksekgonul, Mao, Wu, Zou (2023). GPT detectors are biased against "
                 "non-native English writers. Patterns 4(7), 100779.",
     "licence": "MIT",
     "url": "https://github.com/Weixin-Liang/ChatGPT-Detector-Bias"},
    {"name": "ELLIPSE corpus",
     "sets": "ellipse",
     "citation": "Crossley, Tian, Baffour, Franklin, Benner, Boser (2023). A large-scale corpus "
                 "for assessing written argumentation: ELLIPSE (English Language Learner "
                 "Insight, Proficiency and Skills Evaluation).",
     "licence": "CC BY 4.0",
     "url": "https://github.com/scrosseye/ELLIPSE-Corpus"},
)


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% (by default) Wilson score interval for k successes in n trials; (nan, nan) if n = 0."""
    if n < 0 or not 0 <= k <= max(n, 0):
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return math.nan, math.nan
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # The exact bounds at k = 0 and k = n are 0 and 1; rounding must not move them.
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def flagged(p: Any, threshold: float) -> np.ndarray:
    """True where P(ai) > threshold (an AI verdict)."""
    return np.asarray(p, dtype=np.float64) > threshold


def threshold_for_fpr(p_human: Any, max_fpr: float) -> float:
    """The cut-off t that flags (P(ai) > t) at most `max_fpr` of these human texts.

    The same choice as the stacker's ai_threshold (`operating_points`).
    """
    p = np.asarray(p_human, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        raise ValueError("no human texts to set the threshold on")
    if not 0 < max_fpr < 1:
        raise ValueError("max_fpr must be in (0, 1)")
    return float(np.quantile(p, 1 - max_fpr, method="higher"))


def _labels(df: pd.DataFrame, by: Any) -> pd.Series:
    if isinstance(by, str):
        return df[by]
    return pd.Series(by, index=df.index) if not isinstance(by, pd.Series) else by


def group_rates(df: pd.DataFrame, p: Any, threshold: float, by: Any,
                z: float = Z_95) -> list[dict[str, Any]]:
    """
    [{group, n, k, rate, ci_low, ci_high}] per group: k texts of n with P(ai) > threshold.

    `p` is aligned with the rows of `df`; `by` is a column of `df` or one label
    per row. Rows without a group (None / NaN) are left out. Groups come in
    category order for a categorical `by`, else sorted.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(df),):
        raise ValueError(f"p must have shape ({len(df)},), got {p.shape}")
    labels = _labels(df, by)
    if len(labels) != len(df):
        raise ValueError("by must have one label per row")
    hit = flagged(p, threshold)
    keep = labels.notna().to_numpy()
    values = labels.to_numpy()[keep]
    hit = hit[keep]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        order = [c for c in labels.cat.categories if (values == c).any()]
    else:
        order = sorted(pd.unique(values), key=str)
    rows = []
    for group in order:
        mask = values == group
        n, k = int(mask.sum()), int(hit[mask].sum())
        low, high = wilson_interval(k, n, z)
        rows.append({"group": str(group), "n": n, "k": k, "rate": k / n if n else math.nan,
                     "ci_low": low, "ci_high": high})
    return rows


def overall_rate(p: Any, threshold: float, z: float = Z_95) -> dict[str, Any]:
    """{n, k, rate, ci_low, ci_high} over all of `p`."""
    hit = flagged(p, threshold)
    n, k = int(hit.size), int(hit.sum())
    low, high = wilson_interval(k, n, z)
    return {"n": n, "k": k, "rate": k / n if n else math.nan, "ci_low": low, "ci_high": high}


# --------------------------------------------------------------------------- #
# Flag rule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlagRule:
    """When a human group's FPR is "far above average" (see the module docstring)."""

    min_ratio: float = 2.0    # rate at least this many times the reference
    min_excess: float = 0.02  # ... and at least this much above it (absolute)
    min_n: int = 30           # fewer texts: "too small to judge"

    def __post_init__(self) -> None:
        if self.min_ratio < 1 or self.min_excess < 0 or self.min_n < 1:
            raise ValueError(f"invalid flag rule {self}")

    def describe(self) -> str:
        return (f"FPR >= {self.min_ratio:g}x the reference and >= reference + "
                f"{100 * self.min_excess:g} pp, with the 95% Wilson lower bound above the "
                f"reference; groups under {self.min_n} texts are too small to judge")


def status_of(row: dict[str, Any], reference_rate: float, rule: FlagRule) -> str:
    """One of STATUSES for a row of `group_rates` (see the module docstring)."""
    if row["n"] < rule.min_n:
        return "too_small"
    rate = row["rate"]
    if rate >= rule.min_ratio * reference_rate and rate >= reference_rate + rule.min_excess:
        return "far_above" if row["ci_low"] > reference_rate else "above_uncertain"
    return "ok"


def flag_groups(rows: Sequence[dict[str, Any]], reference_rate: float,
                rule: Optional[FlagRule] = None) -> list[dict[str, Any]]:
    """The rows with `ratio` (rate / reference), `status` and `far_above` (bool) added."""
    rule = rule or FlagRule()
    if not math.isfinite(reference_rate) or not 0 <= reference_rate <= 1:
        raise ValueError(f"reference_rate must be in [0, 1], got {reference_rate}")
    out = []
    for row in rows:
        status = status_of(row, reference_rate, rule)
        ratio = (row["rate"] / reference_rate if reference_rate > 0
                 else (math.inf if row["rate"] > 0 else 1.0))
        out.append({**row, "ratio": ratio, "status": status, "far_above": status == "far_above"})
    return out


# --------------------------------------------------------------------------- #
# Groupings
# --------------------------------------------------------------------------- #
@dataclass
class Grouping:
    """Rows of a frame (by position) and their group labels, plus how to report them.

    kind: "human" (an FPR, flag rule applies), "ai_edited" (share flagged as
    AI, no flag rule) or "ai" (recall). `user_facing`: its flagged groups may
    become known limitations (not for dataset sources or split names).
    """

    key: str
    title: str
    section: str
    kind: str
    user_facing: bool
    positions: np.ndarray
    labels: pd.Series

    def meta(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "section": self.section,
                "kind": self.kind, "user_facing": self.user_facing}


def length_bucket(n_words: Any) -> pd.Categorical:
    """Word-count bucket per text (`LENGTH_LABELS`, in order); NaN stays NaN."""
    n = pd.to_numeric(pd.Series(n_words), errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(n.shape, None, dtype=object)
    for label, lo, hi in LENGTH_BUCKETS:
        mask = (n >= lo) & (n <= hi if hi is not None else True)
        out[mask] = label
    return pd.Categorical(out, categories=LENGTH_LABELS, ordered=True)


def proficiency_band(score: Any) -> pd.Categorical:
    """ELLIPSE overall score -> band (`PROFICIENCY_LABELS`); NaN stays NaN."""
    s = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(s.shape, None, dtype=object)
    for label, lo, hi in PROFICIENCY_BANDS:
        out[(s >= lo) & (s < hi)] = label
    return pd.Categorical(out, categories=PROFICIENCY_LABELS, ordered=True)


def grade_label(grade: Any) -> pd.Series:
    """"grade 8" etc. per text, ordered by grade; NaN stays NaN."""
    g = pd.to_numeric(pd.Series(grade), errors="coerce")
    names = g.map(lambda x: None if pd.isna(x) else f"grade {x:g}")
    cats = [f"grade {x:g}" for x in sorted(g.dropna().unique())]
    return pd.Series(pd.Categorical(names, categories=cats, ordered=True))


def word_counts(df: pd.DataFrame) -> np.ndarray:
    """`n_words` where present, else whitespace word count of `text`."""
    counted = df["text"].map(lambda t: len(str(t).split()) if isinstance(t, str) else 0)
    if "n_words" not in df:
        return counted.to_numpy(dtype=np.float64)
    n = pd.to_numeric(df["n_words"], errors="coerce")
    return n.fillna(counted).to_numpy(dtype=np.float64)


def _grouping(key: str, title: str, section: str, kind: str, user_facing: bool,
              mask: Any, labels: Any) -> Grouping:
    mask = np.asarray(mask, dtype=bool)
    labels = pd.Series(labels).reset_index(drop=True)
    return Grouping(key, title, section, kind, user_facing, np.flatnonzero(mask),
                    labels[mask].reset_index(drop=True))


def corpus_groupings(df: pd.DataFrame) -> list[Grouping]:
    """Groupings of corpus human texts (one row per text); by split is `split_rows`."""
    every = np.ones(len(df), dtype=bool)
    length = length_bucket(word_counts(df))
    out = []
    if "domain" in df:
        out.append(_grouping("corpus_domain", "Corpus human texts by domain", "corpus",
                             "human", True, every, df["domain"].astype("string")))
    out.append(_grouping("corpus_length", "Corpus human texts by length (words)", "corpus",
                         "human", True, every, length))
    if "domain" in df:
        tech = (df["domain"].astype(str) == TECHNICAL_DOMAIN).to_numpy()
        if tech.any():
            out.append(_grouping("corpus_technical_length",
                                 "Corpus technical human texts by length (words)", "corpus",
                                 "human", True, tech, length))
    if "source" in df:
        out.append(_grouping("corpus_source", "Corpus human texts by source dataset", "corpus",
                             "human", False, every, df["source"].astype("string")))
    return out


def split_rows(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """(row positions, split label) with one entry per (text, split) membership, from the
    `splits` column (a list per text): a text in test and test_short counts in both."""
    pos, labels = [], []
    for i, splits in enumerate(df["splits"]):
        for s in splits:
            pos.append(i)
            labels.append(s)
    return np.asarray(pos, dtype=np.int64), pd.Series(labels, dtype="string")


def _ordered(values: Iterable[Any], known: Sequence[str]) -> list[str]:
    """The known values first, in their order, then any others, sorted."""
    seen = set(values)
    return [v for v in known if v in seen] + sorted(seen - set(known))


def external_groupings(ext: pd.DataFrame) -> list[Grouping]:
    """Groupings of the external fairness sets (columns as in EXTERNAL_COLUMNS, see above)."""
    missing = set(EXTERNAL_COLUMNS) - set(ext.columns)
    if missing:
        raise ValueError(f"fairness sets are missing column(s): {', '.join(sorted(missing))}")
    writer = ext["writer"].astype(str).str.strip().str.lower()
    edited = ext["edited_by_ai"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    ai = (writer == "ai").to_numpy()
    human = ~ai & ~edited
    sets = ext["set"].astype(str)
    group = ext["group"].fillna(sets).astype(str)
    out = [
        _grouping("ext_set", "External human texts by set", "external", "human", True,
                  human, group),
        _grouping("ext_writer", "External human texts by writer", "external", "human", True,
                  human, pd.Categorical(writer.where(human),
                                        categories=_ordered(writer[human], WRITERS))),
        # Not user-facing: length here is confounded with the set (TOEFL essays are short);
        # the corpus length buckets speak to length.
        _grouping("ext_length", "External human texts by length (words)", "external", "human",
                  False, human, length_bucket(word_counts(ext))),
    ]
    ellipse = human & (sets == ELLIPSE_SET).to_numpy()
    if ellipse.any() and "proficiency" in ext:
        band = proficiency_band(ext["proficiency"])
        if pd.notna(band[ellipse]).any():
            out.append(_grouping("ellipse_proficiency",
                                 "ELLIPSE essays (English learners) by proficiency score",
                                 "external", "human", True, ellipse, band))
    if ellipse.any() and "grade" in ext:
        grades = grade_label(ext["grade"])
        if grades[ellipse].notna().any():
            out.append(_grouping("ellipse_grade", "ELLIPSE essays (English learners) by grade",
                                 "external", "human", True, ellipse, grades))
    if edited.any():
        out.append(_grouping("ext_ai_edited", "Human texts polished or simplified by AI, by set",
                             "ai_edited", "ai_edited", False, edited, group))
    if ai.any():
        out.append(_grouping("ext_ai", "AI texts by set (recall)", "ai", "ai", False, ai, group))
    return out


def grouping_rates(g: Grouping, p: Any, threshold: float, reference_rate: float,
                   rule: FlagRule) -> dict[str, Any]:
    """A report table: the grouping's meta plus its rows (flagged against the reference
    for human groupings)."""
    p = np.asarray(p, dtype=np.float64)[g.positions]
    frame = pd.DataFrame(index=range(len(g.positions)))
    rows = group_rates(frame, p, threshold, g.labels)
    if g.kind == "human":
        rows = flag_groups(rows, reference_rate, rule)
    return {**g.meta(), "rows": rows}


# --------------------------------------------------------------------------- #
# Why: feature medians
# --------------------------------------------------------------------------- #
def group_feature_medians(g: Grouping, features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """{group: {feature: median}} over the grouping's rows; `features` is aligned by
    position with the frame the grouping was built on (NaN rows = not featurised)."""
    sub = features.iloc[g.positions].reset_index(drop=True)
    out: dict[str, dict[str, Any]] = {}
    labels = g.labels
    groups = (labels.cat.categories if isinstance(labels.dtype, pd.CategoricalDtype)
              else sorted(labels.dropna().unique(), key=str))
    for group in groups:
        mask = (labels == group).to_numpy()
        if not mask.any():
            continue
        part = sub[mask]
        med = {}
        for col in features.columns:
            vals = part[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            med[col] = float(np.median(vals)) if vals.size else None
        med["n"] = int(np.isfinite(part.to_numpy(dtype=np.float64)).all(axis=1).sum())
        out[str(group)] = med
    return out


def column_medians(features: pd.DataFrame) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for col in features.columns:
        vals = features[col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[col] = float(np.median(vals)) if vals.size else None
    return out


# --------------------------------------------------------------------------- #
# Known limitations
# --------------------------------------------------------------------------- #
DOMAIN_NOUNS = {
    "qa": "Answers to questions (Q&A posts)",
    "technical": "Technical texts",
    "academic": "Academic texts",
    "review": "Reviews",
    "news": "News articles",
    "other": "Texts of other kinds",
}
WRITER_NOUNS = {
    "non-native": "Essays by non-native English writers",
    "native": "Essays by native English writers",
    "unknown": "Essays by writers of unknown background",
}


def _length_phrase(label: str) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if name == label:
            if lo == 0:
                return f"under {hi + 1} words"
            if hi is None:
                return f"of {lo} words or more"
            return f"of {lo} to {hi} words"
    return f"of {label} words"


def describe_group(key: str, group: str) -> str:
    """The subject of a known-limitation sentence for a group of a grouping."""
    if key == "corpus_domain":
        return DOMAIN_NOUNS.get(group, f"Texts in the {group} domain")
    if key in ("corpus_length", "ext_length"):
        return f"Texts {_length_phrase(group)}"
    if key == "corpus_technical_length":
        return f"Technical texts {_length_phrase(group)}"
    if key == "ext_writer":
        return WRITER_NOUNS.get(group, f"Texts by {group} writers")
    if key == "ext_set":
        # Set names read "<writer>: <texts>", e.g. "non-native: TOEFL essays".
        who, sep, what = group.partition(": ")
        if sep and who in WRITER_NOUNS:
            return f"{what[:1].upper()}{what[1:]} by {who} English writers"
        return f"Texts like the {group}"
    if key == "ellipse_proficiency":
        return (f"Essays by English learners with proficiency {group.split(' ')[0]} "
                "on a 1-5 scale")
    if key == "ellipse_grade":
        return f"Essays by English learners in {group}"
    return f"Texts in group {group}"


def short_pct(x: float) -> str:
    """12% / 1.5% / 0.3%: whole percent from 10% up, one decimal below."""
    if x is None or not math.isfinite(x):
        return "n/a"
    pct = 100 * x
    text = f"{pct:.0f}" if pct >= 10 else f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _slug(text: str) -> str:
    text = text.lower().replace("<", " under ").replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def known_limitations(report: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One user-facing caveat per `far_above` group of a user-facing grouping, worst first.

    `report` is one model's result (keys `reference` and `tables`, as
    `evaluate_fairness` returns it). Each item: id, grouping, group, n, rate,
    ci_low, ci_high, reference_rate, message.
    """
    ref = report["reference"]["rate"]
    items = []
    for table in report.get("tables", []):
        if table.get("kind") != "human" or not table.get("user_facing"):
            continue
        for row in table["rows"]:
            if row.get("status") != "far_above":
                continue
            subject = describe_group(table["key"], row["group"])
            message = (f"{subject} are flagged as AI {short_pct(row['rate'])} of the time "
                       f"(vs {short_pct(ref)} for human text in our test data). Treat an AI "
                       "verdict on such text with caution.")
            items.append({"id": f"{table['key']}:{_slug(row['group'])}",
                          "grouping": table["key"], "group": row["group"], "n": row["n"],
                          "rate": row["rate"], "ci_low": row["ci_low"],
                          "ci_high": row["ci_high"], "reference_rate": ref,
                          "message": message})
    items.sort(key=lambda it: -it["rate"])
    return items


# --------------------------------------------------------------------------- #
# One model
# --------------------------------------------------------------------------- #
def evaluate_fairness(corpus_human: pd.DataFrame, p_corpus: Any, threshold: float, *,
                      external: Optional[pd.DataFrame] = None, p_external: Any = None,
                      rule: Optional[FlagRule] = None,
                      features_corpus: Optional[pd.DataFrame] = None,
                      features_external: Optional[pd.DataFrame] = None,
                      features_ai: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """
    One model's fairness result: reference FPR, a table per grouping, feature
    medians (when feature frames are given) and the known limitations.

    `corpus_human`: one row per corpus human text of the evaluation splits
    (columns text, domain, source, n_words; optional `splits`, a list per
    text); `p_corpus` its P(ai). `external` / `p_external`: the fairness sets
    and their P(ai), or None. `features_*`: KEY_FEATURES per row of the
    matching frame (NaN = not computed) and of a sample of corpus AI texts.
    """
    rule = rule or FlagRule()
    p_corpus = np.asarray(p_corpus, dtype=np.float64)
    if p_corpus.shape != (len(corpus_human),):
        raise ValueError("p_corpus must have one value per corpus human text")
    if len(corpus_human) == 0:
        raise ValueError("no corpus human texts to set the reference FPR on")
    reference = overall_rate(p_corpus, threshold)
    ref_rate = reference["rate"]

    tables: list[dict[str, Any]] = []
    medians: dict[str, Any] = {}
    if "splits" in corpus_human:
        pos, labels = split_rows(corpus_human)
        g = Grouping("corpus_split", "Corpus human texts by evaluation split", "corpus",
                     "human", False, pos, labels)
        tables.append(grouping_rates(g, p_corpus, threshold, ref_rate, rule))
    for g in corpus_groupings(corpus_human):
        tables.append(grouping_rates(g, p_corpus, threshold, ref_rate, rule))
        if features_corpus is not None:
            medians[g.key] = group_feature_medians(g, features_corpus)
    if external is not None and len(external):
        p_external = np.asarray(p_external, dtype=np.float64)
        if p_external.shape != (len(external),):
            raise ValueError("p_external must have one value per external text")
        for g in external_groupings(external):
            tables.append(grouping_rates(g, p_external, threshold, ref_rate, rule))
            if features_external is not None:
                medians[g.key] = group_feature_medians(g, features_external)

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "flag_rule": {**asdict(rule), "description": rule.describe()},
        "reference": {**reference, "description": "all corpus human texts of the evaluation "
                                                  "splits"},
        "tables": tables,
        "far_above": [{"grouping": t["key"], **r} for t in tables if t["kind"] == "human"
                      for r in t["rows"] if r["status"] == "far_above"],
    }
    if features_corpus is not None:
        result["features"] = {
            "names": list(features_corpus.columns),
            "corpus_human": column_medians(features_corpus),
            "corpus_ai": column_medians(features_ai) if features_ai is not None else None,
            "n_corpus_ai": 0 if features_ai is None else int(len(features_ai)),
            "groups": medians,
        }
    result["known_limitations"] = known_limitations(result)
    return result


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def cap_per_stratum(df: pd.DataFrame, strata: Sequence[Any], max_n: Optional[int],
                    seed: int) -> pd.DataFrame:
    """
    At most `max_n` rows per stratum (the combination of the `strata` labels,
    one array per key, aligned with the rows), drawn with a fixed seed; rows
    keep their original order. Small strata are kept whole, so the small
    groups the fairness check is about are not thinned out.
    """
    if max_n is None or len(df) == 0:
        return df
    keys = pd.DataFrame({f"k{i}": pd.Series(np.asarray(s, dtype=object)).fillna("")
                         .astype(str).to_numpy() for i, s in enumerate(strata)})
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(df), dtype=bool)
    for _, idx in keys.groupby(list(keys.columns), sort=True).indices.items():
        if len(idx) > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        keep[idx] = True
    return df[keep]
