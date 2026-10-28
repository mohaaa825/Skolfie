"""
scripts/eval_fairness.py
-------------------------
Roadmap step 10: the false-positive rate on human text, per subgroup, for
one or more saved models (`skolfie.eval.fairness` has the rates, the flag
rule and the wording).

Human texts measured:

    corpus     every human text of --eval-splits (default test and
               test_ood_domain), by split, domain, source dataset, length
               bucket and, for technical text, by length
    external   the fairness sets of scripts/build_fairness_sets.py
               (--fairness-sets): TOEFL essays by non-native writers and
               native-writer essays (Liang et al. 2023), ELLIPSE essays by
               English learners by proficiency and grade; human texts that an
               AI polished or simplified are reported apart, and the AI texts
               of these sets as recall. Without the file this part is skipped
               and the report says so.

Threshold: a text is flagged when P(ai) > t, the AI verdict. t is, in this
order, --threshold; the model's own `cutoffs["ai_threshold"]` (a
StackedModel: set by stack.py to flag 1% of the human texts it was fitted
on); or the cut-off that flags --max-fpr (default 1%) of the human texts of
--threshold-split (default val), computed here. The report says which.

A human group is "far above average" when its FPR is at least --min-ratio
(2x) the corpus-wide human FPR and at least --min-excess (2 pp) above it,
and its 95% Wilson lower bound is above it; groups under --min-n (30)
texts are "too small to judge". With a feature model available (the model
itself, a StackedModel member or the transformer's feature extractor),
each group also gets the median GPT-2 log-perplexity, vocabulary variety and
word / sentence length next to the corpus human and AI medians.

Writes into --out-dir: fairness_report.json (everything, per model),
fairness_report.md and known_limitations.json (the caveats for the app, for
the first --model). Each distinct text is scored once per model.

Example:

    python scripts/eval_fairness.py --corpus data/processed/corpus.parquet \\
        --splits data/processed/splits.parquet \\
        --fairness-sets data/processed/fairness/fairness_sets.parquet \\
        --model stacked=models/stacked.joblib --out-dir reports/fairness/stacked
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SPLITS = "test,test_ood_domain"
# build_fairness_sets.py writes fairness_manifest.json; the others are fallbacks.
MANIFEST_NAMES = ("fairness_manifest.json", "{stem}_manifest.json", "manifest.json")


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return n


def _open_fraction(value: str) -> float:
    x = float(value)
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (exclusive)")
    return x


def _probability(value: str) -> float:
    x = float(value)
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return x


def _at_least_one(value: str) -> float:
    x = float(value)
    if x < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return x


def _model(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep:
        return Path(value).stem, Path(value)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"expected [NAME=]PATH, got {value!r}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_fairness.py",
        description="False-positive rate on human text per subgroup (non-native writers, short "
                    "and technical text) for one or more saved models.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fairness-sets", type=Path,
                        help="fairness_sets.parquet from build_fairness_sets.py (optional; "
                             "without it the external sets are skipped with a note).")
    parser.add_argument("--model", type=_model, action="append", required=True,
                        metavar="[NAME=]PATH",
                        help="A saved model (joblib) with predict_proba(texts); repeat to "
                             "compare models (known_limitations.json is for the first).")
    parser.add_argument("--threshold", type=_probability,
                        help="Flag texts with P(ai) above this, for every model (default: the "
                             "model's cutoffs['ai_threshold'], else the --max-fpr cut-off on "
                             "--threshold-split).")
    parser.add_argument("--max-fpr", type=_open_fraction, default=0.01,
                        help="Human FPR on --threshold-split that sets the threshold of a "
                             "model without cut-offs (default: 0.01).")
    parser.add_argument("--threshold-split", default="val",
                        help="Split whose human texts set that threshold (default: val).")
    parser.add_argument("--max-val", type=_positive_int,
                        help="At most this many human texts of --threshold-split (default: all).")
    parser.add_argument("--eval-splits", default=DEFAULT_EVAL_SPLITS,
                        help=f"Comma-separated corpus splits whose human texts are measured "
                             f"(default: {DEFAULT_EVAL_SPLITS}).")
    parser.add_argument("--max-per-group", type=_positive_int,
                        help="Cap per stratum, for speed: corpus human texts per domain x "
                             "length bucket, fairness-set texts per set (x proficiency band); "
                             "default: no cap.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-ratio", type=_at_least_one, default=2.0,
                        help="Flag rule: FPR at least this many times the reference (default: 2).")
    parser.add_argument("--min-excess", type=_probability, default=0.02,
                        help="Flag rule: FPR at least this far above the reference "
                             "(default: 0.02, i.e. 2 pp).")
    parser.add_argument("--min-n", type=_positive_int, default=30,
                        help="Groups with fewer texts are too small to judge (default: 30).")
    parser.add_argument("--features-model", type=Path,
                        help="Feature model (features.joblib) for the feature medians; default: "
                             "the one inside each --model, if any.")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip the feature medians.")
    parser.add_argument("--feature-ai-n", type=_non_negative_int, default=500,
                        help="Corpus AI texts for the AI feature medians (default: 500).")
    parser.add_argument("--batch-size", type=_positive_int, default=4096,
                        help="Texts per predict_proba call (default: 4096).")
    parser.add_argument("--out-dir", type=Path,
                        help="Default: reports/fairness/<model> (or compare-<models>).")
    return parser


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_corpus(path: Path) -> Any:
    """The corpus columns this check needs, indexed by id."""
    import pandas as pd
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    missing = {"id", "text", "label"} - available
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    cols = [c for c in ("id", "text", "label", "domain", "source", "n_words") if c in available]
    df = pd.read_parquet(path, columns=cols)
    if df["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    df["label"] = df["label"].astype(str).str.lower()
    df.index = pd.Index(df["id"].to_numpy())
    return df


def corpus_human(corpus: Any, splits: Any, eval_splits: Sequence[str]) -> Any:
    """Human texts of the evaluation splits, one row per text, with a `splits` list column."""
    mem = splits[splits["split"].isin(eval_splits) & splits["id"].isin(corpus.index)]
    mem = mem[mem["id"].isin(corpus.index[corpus["label"] == "human"])]
    order = {s: i for i, s in enumerate(eval_splits)}
    by_id = mem.groupby("id", sort=True)["split"].agg(
        lambda s: sorted(set(s), key=order.__getitem__))
    df = corpus.loc[by_id.index].copy()
    df["splits"] = list(by_id)
    return df


def split_human(corpus: Any, splits: Any, split: str, max_n: Optional[int], seed: int) -> Any:
    """Human texts of one split (at most max_n, seeded), e.g. val for the threshold."""
    ids = splits.loc[splits["split"] == split, "id"]
    pool = corpus.loc[ids[ids.isin(corpus.index)].unique()]
    pool = pool[pool["label"] == "human"]
    if max_n is not None and len(pool) > max_n:
        pool = pool.sample(n=max_n, random_state=seed)
    return pool


def read_fairness_sets(path: Path) -> Any:
    import pandas as pd

    from skolfie.eval.fairness import EXTERNAL_COLUMNS

    df = pd.read_parquet(path)
    missing = ({"text", *EXTERNAL_COLUMNS}) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}; "
                         "rebuild it with scripts/build_fairness_sets.py")
    for col in ("proficiency", "grade"):
        if col not in df:
            df[col] = float("nan")
    df["text"] = df["text"].fillna("").astype(str)
    return df.reset_index(drop=True)


def fairness_manifest(path: Path) -> dict[str, Any]:
    """The fairness sets' manifest next to the parquet, or {}."""
    for pattern in MANIFEST_NAMES:
        candidate = path.with_name(pattern.format(stem=path.stem))
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return {**data, "_path": str(candidate)}
    return {}


def _as_dict(item: Any) -> Optional[dict[str, Any]]:
    import dataclasses

    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return None


def _source_list(raw: Any) -> list[dict[str, Any]]:
    """Sources as a list of dicts, from a {name: dict | dataclass | str} mapping or a list."""
    if isinstance(raw, dict):
        out = []
        for key, value in raw.items():
            item = _as_dict(value) or {"citation": str(value)}
            out.append({"name": str(key), **item})
        return out
    if not isinstance(raw, (list, tuple)):
        return []
    return [d for d in map(_as_dict, raw) if d is not None]


def data_sources(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """(sources, where they came from): the manifest's, else the fairness-sets module's
    constants, else this module's built-in list."""
    sources = _source_list(manifest.get("sources"))
    if sources:
        return sources, manifest["_path"]
    try:
        import skolfie.data.fairness_sets as module
    except Exception:  # not there yet, or fails to import: fall back to the built-in list
        module = None
    for attr in ("SOURCES", "SOURCE_INFO", "DATA_SOURCES"):
        sources = _source_list(getattr(module, attr, None))
        if sources:
            return sources, f"skolfie.data.fairness_sets.{attr}"
    from skolfie.eval.fairness import DEFAULT_SOURCES

    return [dict(s) for s in DEFAULT_SOURCES], "skolfie.eval.fairness.DEFAULT_SOURCES"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def find_featurizer(model: Any) -> Any:
    """A FeatureModel whose label-free `base_features` can be computed: the model's own
    feature model (or a StackedModel member's), else a transformer's feature extractor."""
    from skolfie.explain.explainer import find_feature_model
    from skolfie.models.feature_model import FeatureModel

    found = find_feature_model(model)
    if found is not None:
        return found
    candidates = [model]
    members = getattr(model, "members", None)
    if isinstance(members, dict) and callable(getattr(model, "_member", None)):
        candidates += [getattr(model._member(n), "obj", None) for n in members]
    for obj in candidates:
        inner = getattr(obj, "_features", None)
        if isinstance(inner, FeatureModel):
            return inner
    return None


def key_features(featurizer: Any, texts: Sequence[str], batch_size: int) -> Any:
    """KEY_FEATURES (those the featurizer has) per text, as a DataFrame."""
    import numpy as np
    import pandas as pd

    from skolfie.eval.fairness import KEY_FEATURES

    blocks, names = [], None
    for lo in range(0, len(texts), batch_size):
        X, names = featurizer.base_features(list(texts[lo:lo + batch_size]))
        blocks.append(np.asarray(X, dtype=np.float64))
    if names is None:
        return pd.DataFrame(columns=[])
    cols = [names.index(f) for f in KEY_FEATURES if f in names]
    return pd.DataFrame(np.vstack(blocks)[:, cols], columns=[names[c] for c in cols])


def choose_threshold(args: argparse.Namespace, model: Any,
                     p_val_human: Optional[Any]) -> dict[str, Any]:
    """{value, source, description, ...}: how this model's threshold was chosen."""
    from skolfie.eval.fairness import flagged, threshold_for_fpr

    if args.threshold is not None:
        return {"value": args.threshold, "source": "given",
                "description": "set with --threshold"}
    cutoffs = getattr(model, "cutoffs", None)
    if isinstance(cutoffs, dict) and cutoffs.get("ai_threshold") is not None:
        max_fpr = cutoffs.get("max_fpr")
        return {"value": float(cutoffs["ai_threshold"]), "source": "model_cutoffs",
                "max_fpr": max_fpr,
                "description": "the model's own ai_threshold (stack.py: at most "
                               + (f"{max_fpr:.0%}" if isinstance(max_fpr, (int, float))
                                  else "max_fpr")
                               + " of the human texts it was fitted on score above it)"}
    t = threshold_for_fpr(p_val_human, args.max_fpr)
    return {"value": t, "source": "val_fpr", "split": args.threshold_split,
            "max_fpr": args.max_fpr, "n_human": int(len(p_val_human)),
            "fpr_on_split": float(flagged(p_val_human, t).mean()),
            "description": f"flags {args.max_fpr:.0%} of the {len(p_val_human):,} human texts "
                           f"of {args.threshold_split} (computed here; the model has no "
                           "cut-offs)"}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def _ci(row: dict[str, Any]) -> str:
    if row.get("ci_low") is None or row.get("ci_high") is None:
        return "n/a"
    return f"{100 * row['ci_low']:.1f}–{100 * row['ci_high']:.1f}%"


def _ratio(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.1f}×"


def _num(x: Any, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _status(row: dict[str, Any], min_n: int) -> str:
    return {"far_above": "**far above**", "above_uncertain": "above (not significant)",
            "ok": "ok", "too_small": f"too small (n < {min_n})"}.get(row.get("status"), "")


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    from skolfie.eval.report import _table as report_table

    return report_table(header, rows)


def _human_tables(res: dict[str, Any], section: str) -> list[str]:
    lines: list[str] = []
    min_n = res["flag_rule"]["min_n"]
    for t in res["tables"]:
        if t["kind"] != "human" or t["section"] != section:
            continue
        lines += [f"#### {t['title']}", ""]
        lines += _table(["group", "n", "flagged", "FPR", "95% CI", "vs reference", "status"],
                        [[r["group"], f"{r['n']:,}", f"{r['k']:,}", _pct(r["rate"]), _ci(r),
                          _ratio(r["ratio"]), _status(r, min_n)] for r in t["rows"]])
        lines.append("")
    return lines


def _feature_section(res: dict[str, Any]) -> list[str]:
    from skolfie.eval.fairness import FEATURE_TITLES

    feats = res.get("features")
    if not feats:
        return []
    names = feats["names"]
    if not names:
        return []
    header = ["group", "n"] + [FEATURE_TITLES.get(f, f) for f in names]
    lines = ["### Why: feature medians", "",
             "Median of label-free features per group, next to the corpus human and AI "
             "medians. Low perplexity (predictable wording), low vocabulary variety and short "
             "words are what AI text usually has, and what the literature blames for false "
             "alarms on non-native writing.", ""]
    rows = [["corpus human texts", f"{res['reference']['n']:,}"]
            + [_num(feats["corpus_human"].get(f)) for f in names]]
    if feats.get("corpus_ai"):
        rows.append(["corpus AI texts (sample)", f"{feats['n_corpus_ai']:,}"]
                    + [_num(feats["corpus_ai"].get(f)) for f in names])
    lines += _table(header, rows) + [""]
    titles = {t["key"]: t["title"] for t in res["tables"]}
    for key, groups in feats["groups"].items():
        if key in ("corpus_source",) or not groups:
            continue
        lines += [f"#### {titles.get(key, key)}", ""]
        lines += _table(header, [[g, f"{m.get('n', 0):,}"] + [_num(m.get(f)) for f in names]
                                 for g, m in groups.items()])
        lines.append("")
    return lines


def render_model(name: str, res: dict[str, Any]) -> list[str]:
    ref = res["reference"]
    thr = res["threshold_info"]
    lines = [f"## {name}", "",
             f"- threshold: P(ai) > {thr['value']:.4f}, {thr['description']}",
             f"- reference human FPR: **{_pct(ref['rate'])}** ({ref['k']:,} of {ref['n']:,} "
             f"corpus human texts, 95% CI {_ci(ref)})",
             f"- groups far above the reference: {len(res['far_above'])}"
             + (" (" + ", ".join(f"{r['grouping']}: {r['group']}" for r in res["far_above"])
                + ")" if res["far_above"] else ""), "",
             "### Human false-positive rate by subgroup (corpus)", ""]
    lines += _human_tables(res, "corpus")
    if res.get("external_note"):
        lines += ["### External sets", "", f"Skipped: {res['external_note']}", ""]
    else:
        lines += ["### External sets: human writing", "",
                  "The Hewlett and ELLIPSE essays keep their sources' anonymisation "
                  "placeholders (@CAPS1, Generic_Name, ...), which may shift their scores "
                  "slightly.", ""]
        lines += _human_tables(res, "external")
        for kind, title, col in (("ai_edited", "### Human texts edited by AI", "flagged as AI"),
                                 ("ai", "### AI texts: recall", "recall")):
            tables = [t for t in res["tables"] if t["kind"] == kind]
            if not tables:
                continue
            lines += [title, ""]
            if kind == "ai_edited":
                lines += ["Written by people, then polished or simplified by an AI: neither "
                          "clearly human nor AI, so not counted in any human FPR.", ""]
            for t in tables:
                lines += _table(["set", "n", "flagged", col, "95% CI"],
                                [[r["group"], f"{r['n']:,}", f"{r['k']:,}", _pct(r["rate"]),
                                  _ci(r)] for r in t["rows"]])
                lines.append("")
    lines += _feature_section(res)
    if res.get("features_note"):
        lines += [f"Feature medians: {res['features_note']}", ""]
    lines += ["### Known limitations", ""]
    if res["known_limitations"]:
        lines += [f"- {item['message']}" for item in res["known_limitations"]]
    else:
        lines.append("None: no user-facing group is far above the reference FPR.")
    lines.append("")
    return lines


def render_md(report: dict[str, Any]) -> str:
    models = report["models"]
    first = next(iter(models.values()))
    rule = first["flag_rule"]
    ext = report["fairness_sets"]
    sampling = report["sampling"]
    lines = [f"# Fairness check: {', '.join(models)}", "",
             f"- corpus `{report.get('corpus_version') or 'unversioned'}`, splits "
             f"`{report.get('splits_version') or 'unversioned'}`; human texts of "
             f"{', '.join(report['eval_splits'])}",
             "- a text is flagged when P(ai) is above the model's threshold (its AI verdict); "
             "on human text that is a false positive",
             f"- far above average: {rule['description']}",
             "- fairness sets: " + (f"`{ext['path']}` ({ext['n_rows']:,} texts)"
                                    if ext.get("n_rows") is not None else ext["note"]),
             "- sampling: " + (f"at most {sampling['max_per_group']:,} texts per stratum "
                               f"(corpus: domain x length; sets: set x proficiency), seed "
                               f"{sampling['seed']}" if sampling["max_per_group"]
                               else "all texts"), ""]
    if len(models) > 1:
        lines += ["## Summary", ""]
        lines += _table(["model", "threshold", "chosen by", "human FPR", "far above",
                         "known limitations"],
                        [[n, f"{r['threshold']:.4f}", r["threshold_info"]["source"],
                          _pct(r["reference"]["rate"]), str(len(r["far_above"])),
                          str(len(r["known_limitations"]))] for n, r in models.items()])
        lines.append("")
    for name, res in models.items():
        lines += render_model(name, res)
    lines += ["## Data sources", ""]
    for src in report["sources"]:
        parts = [f"**{src.get('name', 'unnamed')}**"]
        for key in ("title", "sets", "citation"):
            if src.get(key):
                parts.append(str(src[key]))
        licence = src.get("licence") or src.get("license")
        if licence:
            parts.append(f"licence: {licence}")
        if src.get("url"):
            parts.append(str(src["url"]))
        lines.append("- " + "; ".join(parts))
    lines += ["", f"(from `{report['sources_from']}`)", ""]
    git = report.get("git") or {}
    commit = (git.get("commit") or "unknown")[:12] + (" (dirty)" if git.get("dirty") else "")
    lines += ["## Run", "", f"- git: `{commit}`, created {report['created_at']}",
              f"- scoring {report['seconds']['score']:.0f}s, total "
              f"{report['seconds']['total']:.0f}s"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()
    started = time.time()

    names = [n for n, _ in args.model]
    if len(set(names)) != len(names):
        parser.error(f"duplicate model names: {names}; use NAME=PATH")
    for flag, path in [*(("--model", p) for _, p in args.model),
                       ("--features-model", args.features_model),
                       ("--corpus", args.corpus), ("--splits", args.splits)]:
        if path is not None and not path.is_file():
            parser.error(f"{flag} file not found: {path}")
    eval_splits = list(dict.fromkeys(s.strip() for s in args.eval_splits.split(",")
                                     if s.strip()))
    if not eval_splits:
        parser.error("--eval-splits is empty")

    import numpy as np
    import pandas as pd

    from skolfie.eval.fairness import (
        FlagRule,
        cap_per_stratum,
        evaluate_fairness,
        length_bucket,
        proficiency_band,
        word_counts,
    )
    from skolfie.eval.report import (
        _json_safe,
        atomic_path,
        git_info,
        load_splits,
        load_splits_manifest,
    )
    from skolfie.explain.explainer import load_model
    from skolfie.models.baselines import JoblibScorer

    rule = FlagRule(min_ratio=args.min_ratio, min_excess=args.min_excess, min_n=args.min_n)
    try:
        corpus = read_corpus(args.corpus)
    except ValueError as exc:
        parser.error(str(exc))
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)
    available = set(splits["split"])
    absent = [s for s in eval_splits if s not in available]
    if absent:
        parser.error(f"not in the splits file: {', '.join(absent)}; it has "
                     f"{', '.join(sorted(available))}")

    human = corpus_human(corpus, splits, eval_splits)
    if human.empty:
        parser.error(f"no corpus human texts in {', '.join(eval_splits)}")
    n_human_all = len(human)
    human = cap_per_stratum(human, [human.get("domain", pd.Series("", index=human.index)),
                                    length_bucket(word_counts(human))],
                            args.max_per_group, args.seed).reset_index(drop=True)
    print(f"[data] {len(human):,} corpus human texts" +
          (f" (of {n_human_all:,})" if len(human) < n_human_all else "") +
          f" from {', '.join(eval_splits)}")

    external, ext_info = None, {"path": None, "n_rows": None}
    if args.fairness_sets is None:
        ext_info["note"] = "no --fairness-sets given"
    elif not args.fairness_sets.is_file():
        ext_info.update(path=str(args.fairness_sets),
                        note=f"{args.fairness_sets} not found (build it with "
                             "scripts/build_fairness_sets.py)")
    else:
        try:
            external = read_fairness_sets(args.fairness_sets)
        except ValueError as exc:
            parser.error(str(exc))
        n_ext_all = len(external)
        external = cap_per_stratum(external, [external["set"],
                                              proficiency_band(external["proficiency"])],
                                   args.max_per_group, args.seed).reset_index(drop=True)
        ext_info.update(path=str(args.fairness_sets), n_rows=len(external),
                        n_available=n_ext_all, note=None,
                        by_set={str(k): int(v) for k, v in
                                external["set"].value_counts().sort_index().items()})
        print(f"[data] {len(external):,} fairness-set texts"
              + (f" (of {n_ext_all:,})" if len(external) < n_ext_all else "") + ": "
              + ", ".join(f"{k} {v}" for k, v in ext_info["by_set"].items()))
    if ext_info.get("note"):
        print(f"[data] external fairness sets skipped: {ext_info['note']}")
    fs_manifest = fairness_manifest(args.fairness_sets) if external is not None else {}
    sources, sources_from = data_sources(fs_manifest)

    val_h = None
    ai_ref = None
    if not args.no_features and args.feature_ai_n > 0:
        mem = splits.loc[splits["split"].isin(eval_splits), "id"]
        pool = corpus.loc[mem[mem.isin(corpus.index)].unique()]
        pool = pool[pool["label"] == "ai"]
        ai_ref = pool.sample(n=min(args.feature_ai_n, len(pool)), random_state=args.seed)

    shared_featurizer = (load_model(args.features_model)
                         if args.features_model is not None and not args.no_features else None)
    results: dict[str, Any] = {}
    score_seconds = 0.0
    for name, path in args.model:
        scorer = JoblibScorer(path)
        model = scorer.obj
        needs_val = args.threshold is None and not (
            isinstance(getattr(model, "cutoffs", None), dict)
            and model.cutoffs.get("ai_threshold") is not None)
        if needs_val and val_h is None:
            val_h = split_human(corpus, splits, args.threshold_split, args.max_val, args.seed)
            if val_h.empty:
                parser.error(f"no human texts in split {args.threshold_split!r} to set the "
                             f"threshold of {name}; pass --threshold")
        parts = [human["text"]] + ([external["text"]] if external is not None else []) \
            + ([val_h["text"]] if needs_val else [])
        texts = pd.concat(parts, ignore_index=True).fillna("").astype(str)
        uniq, inverse = np.unique(texts.to_numpy(dtype=object).astype(str), return_inverse=True)
        print(f"[score] {name}: {len(uniq):,} distinct texts")
        t0 = time.time()
        p = np.concatenate([
            np.asarray(scorer.predict_proba(list(uniq[i:i + args.batch_size])), dtype=np.float64)
            for i in range(0, len(uniq), args.batch_size)]) if len(uniq) else np.zeros(0)
        score_seconds += time.time() - t0
        if np.isnan(p).any():
            parser.error(f"{name} returned NaN for {int(np.isnan(p).sum())} texts")
        p = np.clip(p, 0.0, 1.0)[inverse]
        n_h = len(human)
        n_e = 0 if external is None else len(external)
        p_human, p_ext, p_val = p[:n_h], p[n_h:n_h + n_e], p[n_h + n_e:]

        info = choose_threshold(args, model, p_val if needs_val else None)
        print(f"[{name}] threshold {info['value']:.4f} ({info['source']})")

        feats: dict[str, Any] = {}
        features_note = None
        if not args.no_features:
            featurizer = shared_featurizer or find_featurizer(model)
            if featurizer is None:
                features_note = (f"skipped, {name} has no feature model inside (pass "
                                 "--features-model)")
            else:
                t0 = time.time()
                feats["features_corpus"] = key_features(featurizer, human["text"].tolist(),
                                                        args.batch_size)
                if external is not None:
                    feats["features_external"] = key_features(
                        featurizer, external["text"].tolist(), args.batch_size)
                if ai_ref is not None and len(ai_ref):
                    feats["features_ai"] = key_features(featurizer, ai_ref["text"].tolist(),
                                                        args.batch_size)
                print(f"[features] {name}: {time.time() - t0:.0f}s")
        else:
            features_note = "skipped (--no-features)"

        res = evaluate_fairness(human, p_human, info["value"], external=external,
                                p_external=p_ext if external is not None else None,
                                rule=rule, **feats)
        res.update(path=str(path), model_class=type(model).__name__, threshold_info=info,
                   external_note=ext_info.get("note"), features_note=features_note)
        results[name] = res
        ref = res["reference"]
        print(f"[{name}] human FPR {_pct(ref['rate'])} ({ref['k']}/{ref['n']}); "
              f"{len(res['far_above'])} group(s) far above; "
              f"{len(res['known_limitations'])} known limitation(s)")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = _json_safe({
        "created_at": created, "corpus_path": str(args.corpus), "splits_path": str(args.splits),
        "corpus_version": manifest.get("corpus_version"),
        "splits_version": manifest.get("splits_version"), "eval_splits": eval_splits,
        "fairness_sets": {**ext_info, "manifest": {k: v for k, v in fs_manifest.items()
                                                   if k != "sources"} or None},
        "sampling": {"max_per_group": args.max_per_group, "max_val": args.max_val,
                     "seed": args.seed, "n_corpus_human_available": n_human_all,
                     "feature_ai_n": 0 if ai_ref is None else len(ai_ref)},
        "models": results, "sources": sources, "sources_from": sources_from,
        "git": git_info(),
        "seconds": {"score": round(score_seconds, 1), "total": round(time.time() - started, 1)},
    })
    first = names[0]
    limitations = {
        "model": first, "path": str(args.model[0][1]),
        "threshold": report["models"][first]["threshold"],
        "threshold_source": report["models"][first]["threshold_info"]["source"],
        "reference_fpr": report["models"][first]["reference"]["rate"],
        "flag_rule": report["models"][first]["flag_rule"]["description"],
        "created_at": created, "git": report["git"],
        "limitations": report["models"][first]["known_limitations"],
    }

    out_dir = args.out_dir or Path("reports") / "fairness" / (
        first if len(names) == 1 else "compare-" + "-vs-".join(names))
    md = render_md(report)
    for fname, text in (
            ("fairness_report.json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
            ("fairness_report.md", md),
            ("known_limitations.json",
             json.dumps(limitations, indent=2, allow_nan=False) + "\n")):
        with atomic_path(out_dir / fname) as tmp:
            tmp.write_text(text, encoding="utf-8")
    print()
    print(md, end="")
    print(f"\n[save] {out_dir / 'fairness_report.md'}, {out_dir / 'fairness_report.json'}, "
          f"{out_dir / 'known_limitations.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
