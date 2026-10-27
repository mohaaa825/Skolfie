"""
scripts/eval_robust.py
-----------------------
Roadmap step 9: how much each attack lowers detection. Scores one or more
saved models on the clean sample and every attack set of an
`attacks.parquet` from `scripts/make_attacks.py`, and writes a robustness
table per attack.

Per model and attack (AI texts after the attack, compared with the same AI
texts before it, and with the clean human texts):

    AUROC             attacked AI vs clean human
    AUROC (both)      attacked AI vs attacked human, for attacks that also hit
                      human text (typos, synonyms, Unicode tricks)
    recall @0.5       share of attacked AI texts with P(ai) >= 0.5
    recall @thr       ... at the fixed clean-data threshold: the cut-off that
                      flags at most --max-fpr (default 1%) of the clean human texts
    FPR @0.5 / @thr   share of attacked human texts flagged as AI ("both" attacks)
    deltas            each value minus the same value on the clean versions of
                      the same texts (attacks that fail on a few texts are compared
                      on the texts they did attack)

With several `--model`s (the first is the reference, e.g. the model before
adversarial training) the report adds a side-by-side table and the roadmap
check: the AUROC loss under each paraphrase attack at most half the
reference's, and the clean-human FPR at 0.5 not higher than the reference's
by more than --fpr-tolerance.

Writes robust_report.json, robust_report.md and robust_predictions.parquet
(id, orig_id, attack, label, model, p_ai) into --out-dir.

Example:

    python scripts/eval_robust.py --attacks reports/robust/attacks-test/attacks.parquet \\
        --model transformer=models/transformer.joblib \\
        --model transformer-adv=models/transformer-adv.joblib \\
        --out-dir reports/robust/transformer-vs-adv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _fraction(value: str) -> float:
    x = float(value)
    if not 0 <= x < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
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
        prog="eval_robust.py",
        description="Robustness table per attack for one or more saved models.")
    parser.add_argument("--attacks", type=Path, required=True,
                        help="attacks.parquet from make_attacks.py (with the clean rows).")
    parser.add_argument("--model", type=_model, action="append", required=True,
                        metavar="[NAME=]PATH",
                        help="A saved model (joblib) with predict_proba(texts); repeat to "
                             "compare models (the first is the reference).")
    parser.add_argument("--only", help="Comma-separated attacks to evaluate (default: all).")
    parser.add_argument("--max-fpr", type=_fraction, default=0.01,
                        help="Clean-human FPR that sets the fixed threshold (default: 0.01).")
    parser.add_argument("--fpr-tolerance", type=_fraction, default=0.005,
                        help="Allowed rise of the clean-human FPR at 0.5 over the reference "
                             "model in the roadmap check (default: 0.005).")
    parser.add_argument("--batch-size", type=_positive_int, default=4096,
                        help="Texts per predict_proba call (default: 4096).")
    parser.add_argument("--out-dir", type=Path,
                        help="Default: reports/robust/<model> (or compare-<models>).")
    return parser


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def threshold_at_fpr(p_human: Any, max_fpr: float) -> float:
    """Lowest cut-off t (AI iff p >= t) that flags at most max_fpr of these human texts."""
    import numpy as np

    s = np.sort(np.asarray(p_human, dtype=np.float64))[::-1]
    if s.size == 0:
        return math.nan
    k = int(math.floor(max_fpr * s.size + 1e-9))  # texts allowed above the cut-off
    if k >= s.size:
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def _rate(p: Any, t: float) -> float:
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(p >= t)) if p.size else math.nan


def _auroc(p_ai: Any, p_human: Any) -> float:
    import numpy as np

    from skolfie.eval.metrics import auroc

    p_ai, p_human = np.asarray(p_ai), np.asarray(p_human)
    return auroc([1] * len(p_ai) + [0] * len(p_human), np.concatenate([p_ai, p_human]))


def _diff(a: float, b: float) -> float:
    return a - b if not (math.isnan(a) or math.isnan(b)) else math.nan


def robustness(df: Any, max_fpr: float, attacks: Sequence[str]) -> dict[str, Any]:
    """Metrics for one model; `df` has attack, orig_id, label, p_ai (and word_change)."""
    import numpy as np

    from skolfie.eval.metrics import binary_metrics
    from skolfie.robust.attacks import get_attack

    clean = df[df["attack"] == "clean"]
    c_ai = clean[clean["label"] == "ai"].set_index("orig_id")["p_ai"]
    c_h = clean[clean["label"] == "human"].set_index("orig_id")["p_ai"]
    if c_ai.empty or c_h.empty:
        raise ValueError("the attack file needs clean AI and clean human rows (attack 'clean'); "
                         "build it with make_attacks.py without --share")
    thr = threshold_at_fpr(c_h.to_numpy(), max_fpr)
    m = binary_metrics(["ai"] * len(c_ai) + ["human"] * len(c_h),
                       np.concatenate([c_ai.to_numpy(), c_h.to_numpy()]))
    out: dict[str, Any] = {
        "threshold": thr,
        "clean": {"n_ai": len(c_ai), "n_human": len(c_h), "auroc": m["auroc"],
                  "recall_05": _rate(c_ai, 0.5), "recall_thr": _rate(c_ai, thr),
                  "fpr_05": _rate(c_h, 0.5), "fpr_thr": _rate(c_h, thr),
                  "fpr_at_95tpr": m["fpr_at_95tpr"]},
        "attacks": {},
    }
    for name in attacks:
        att = df[df["attack"] == name]
        a_ai = att[att["label"] == "ai"].set_index("orig_id")["p_ai"]
        a_h = att[att["label"] == "human"].set_index("orig_id")["p_ai"]
        try:
            both = get_attack(name).applies_to == "both"
        except ValueError:
            both = not a_h.empty
        base_ai = c_ai[c_ai.index.isin(a_ai.index)]
        r: dict[str, Any] = {
            "applies_to": "both" if both else "ai", "n_ai": len(a_ai), "n_human": len(a_h),
            "mean_word_change": (float(att["word_change"].mean())
                                 if "word_change" in att and len(att) else None),
            "auroc": _auroc(a_ai, c_h) if len(a_ai) else math.nan,
            "auroc_clean": _auroc(base_ai, c_h) if len(base_ai) else math.nan,
            "recall_05": _rate(a_ai, 0.5), "recall_05_clean": _rate(base_ai, 0.5),
            "recall_thr": _rate(a_ai, thr), "recall_thr_clean": _rate(base_ai, thr),
            "mean_p_ai": float(a_ai.mean()) if len(a_ai) else math.nan,
            "mean_p_ai_clean": float(base_ai.mean()) if len(base_ai) else math.nan,
        }
        r["auroc_drop"] = _diff(r["auroc_clean"], r["auroc"])
        r["recall_05_drop"] = _diff(r["recall_05_clean"], r["recall_05"])
        r["recall_thr_drop"] = _diff(r["recall_thr_clean"], r["recall_thr"])
        if both and len(a_h):
            base_h = c_h[c_h.index.isin(a_h.index)]
            r.update({
                "auroc_both": _auroc(a_ai, a_h) if len(a_ai) else math.nan,
                "fpr_05": _rate(a_h, 0.5), "fpr_05_clean": _rate(base_h, 0.5),
                "fpr_thr": _rate(a_h, thr), "fpr_thr_clean": _rate(base_h, thr),
            })
            r["fpr_05_rise"] = _diff(r["fpr_05"], r["fpr_05_clean"])
            r["fpr_thr_rise"] = _diff(r["fpr_thr"], r["fpr_thr_clean"])
        out["attacks"][name] = r
    return out


def roadmap_check(ref: dict[str, Any], new: dict[str, Any], paraphrase: Sequence[str],
                  tolerance: float) -> dict[str, Any]:
    """Step 9's "done when": paraphrase AUROC loss at least halved, clean FPR unchanged."""
    rows = {}
    for name in paraphrase:
        if name not in ref["attacks"] or name not in new["attacks"]:
            continue
        before, after = ref["attacks"][name]["auroc_drop"], new["attacks"][name]["auroc_drop"]
        ok = (not math.isnan(before) and not math.isnan(after)
              and (after <= 0.5 * before if before > 0 else after <= 0))
        rows[name] = {"auroc_loss_ref": before, "auroc_loss": after,
                      "ratio": after / before if before and not math.isnan(after) else math.nan,
                      "pass": bool(ok)}
    fpr_ref, fpr_new = ref["clean"]["fpr_05"], new["clean"]["fpr_05"]
    fpr_ok = fpr_new <= fpr_ref + tolerance
    return {"paraphrase": rows, "clean_fpr_05_ref": fpr_ref, "clean_fpr_05": fpr_new,
            "clean_fpr_at_95tpr_ref": ref["clean"]["fpr_at_95tpr"],
            "clean_fpr_at_95tpr": new["clean"]["fpr_at_95tpr"],
            "fpr_tolerance": tolerance, "fpr_pass": bool(fpr_ok),
            "pass": bool(rows) and all(r["pass"] for r in rows.values()) and bool(fpr_ok)}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%"


def _signed(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if pct:
        return f"{round(100 * x, 1) + 0.0:+.1f} pp"  # + 0.0 turns -0.0 into 0.0
    return f"{round(x, 3) + 0.0:+.3f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_md(report: dict[str, Any]) -> str:
    models = report["models"]
    lines = [f"# Robustness: {', '.join(models)}", "",
             f"- attack sets: `{report['attacks_path']}`"
             + (f" (split `{report['split']}`)" if report.get("split") else ""),
             f"- fixed threshold: the cut-off flagging {report['max_fpr']:.0%} of the clean "
             "human texts, set per model",
             "- AUROC: attacked AI vs clean human texts; AUROC (both): attacked AI vs attacked "
             "human (attacks also applied to human text); deltas are against the clean "
             "versions of the same texts", ""]
    skipped = report.get("skipped") or {}
    if skipped:
        lines += ["Attacks not in this set:", ""]
        lines += [f"- {name}: {why}" for name, why in skipped.items()] + [""]
    for name, res in models.items():
        c = res["clean"]
        lines += [f"## {name}", "",
                  f"Clean: {c['n_ai']:,} AI / {c['n_human']:,} human texts, AUROC "
                  f"{_fmt(c['auroc'])}, recall {_pct(c['recall_05'])} at 0.5 and "
                  f"{_pct(c['recall_thr'])} at the fixed threshold {_fmt(res['threshold'], 4)}; "
                  f"human FPR {_pct(c['fpr_05'])} at 0.5, {_pct(c['fpr_thr'])} at the threshold.",
                  ""]
        rows = []
        for attack, r in res["attacks"].items():
            both = r["applies_to"] == "both"
            rows.append([
                attack, f"{r['n_ai']:,} / {r['n_human']:,}", _pct(r.get("mean_word_change")),
                _fmt(r["auroc"]), _signed(-r["auroc_drop"]),
                _fmt(r.get("auroc_both")) if both else "",
                _pct(r["recall_05"]), _signed(-r["recall_05_drop"], pct=True),
                _pct(r["recall_thr"]), _signed(-r["recall_thr_drop"], pct=True),
                _pct(r.get("fpr_05")) if both else "", _pct(r.get("fpr_thr")) if both else "",
                _signed(r.get("fpr_thr_rise"), pct=True) if both else ""])
        lines += _table(["attack", "AI / human", "words changed", "AUROC", "Δ", "AUROC (both)",
                         "recall @0.5", "Δ", "recall @thr", "Δ", "FPR @0.5", "FPR @thr", "Δ"],
                        rows) + [""]

    comp = report.get("comparison")
    if comp:
        names = list(models)
        lines += ["## Side by side", "",
                  f"AUROC change under each attack (attacked AI vs clean human, minus clean), "
                  f"and recall at each model's fixed threshold. Reference: `{names[0]}`.", ""]
        rows = []
        for attack in models[names[0]]["attacks"]:
            row = [attack]
            for n in names:
                r = models[n]["attacks"].get(attack)
                row += ["n/a", "n/a"] if r is None else [_signed(-r["auroc_drop"]),
                                                         _pct(r["recall_thr"])]
            rows.append(row)
        rows.append(["clean"] + [c for n in names for c in (
            _fmt(models[n]["clean"]["auroc"]), _pct(models[n]["clean"]["recall_thr"]))])
        lines += _table(["attack"] + [h for n in names for h in (f"{n} ΔAUROC",
                                                                  f"{n} recall @thr")], rows)
        for n, check in comp.items():
            lines += ["", f"### Roadmap check: {n} vs {names[0]}", "",
                      "Done when the AUROC loss under paraphrase is at least halved while the "
                      "FPR on clean human text stays the same.", ""]
            prow = [[a, _fmt(c["auroc_loss_ref"]), _fmt(c["auroc_loss"]), _fmt(c["ratio"], 2),
                     "yes" if c["pass"] else "no"] for a, c in check["paraphrase"].items()]
            if prow:
                lines += _table(["paraphrase attack", f"AUROC loss {names[0]}",
                                 f"AUROC loss {n}", "ratio", "halved"], prow)
            else:
                lines += ["No paraphrase attack in this set (t5_paraphrase, llm_paraphrase, "
                          "humanizer)."]
            lines += ["", f"- clean human FPR at 0.5: {_pct(check['clean_fpr_05_ref'])} → "
                          f"{_pct(check['clean_fpr_05'])} (allowed rise "
                          f"{_pct(check['fpr_tolerance'])}): "
                          f"{'ok' if check['fpr_pass'] else 'higher'}",
                      f"- clean FPR at 95% TPR: {_pct(check['clean_fpr_at_95tpr_ref'])} → "
                      f"{_pct(check['clean_fpr_at_95tpr'])}",
                      f"- overall: **{'pass' if check['pass'] else 'not yet'}** (on this "
                      "sample; confirm the FPR on the full test split with evaluate.py "
                      "--compare)"]
    lines += ["", f"Scoring took {report['seconds']['score']:.0f}s."]
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
    for _, path in args.model:
        if not path.is_file():
            parser.error(f"model file not found: {path}")
    if not args.attacks.is_file():
        parser.error(f"attack file not found: {args.attacks}")

    import numpy as np
    import pandas as pd

    from skolfie.eval.report import _json_safe, git_info
    from skolfie.models.baselines import JoblibScorer
    from skolfie.robust.attacks import PARAPHRASE_ATTACKS

    data = pd.read_parquet(args.attacks)
    missing = {"id", "orig_id", "attack", "text", "label"} - set(data.columns)
    if missing:
        parser.error(f"{args.attacks} is missing column(s): {', '.join(sorted(missing))}")
    attacks = [a for a in pd.unique(data["attack"]) if a != "clean"]
    if args.only:
        wanted = [a.strip() for a in args.only.split(",") if a.strip()]
        absent = [a for a in wanted if a not in attacks]
        if absent:
            parser.error(f"not in the attack file: {', '.join(absent)}; it has "
                         f"{', '.join(attacks)}")
        attacks = wanted
        data = data[data["attack"].isin(["clean", *attacks])]
    data = data.reset_index(drop=True)
    manifest_path = args.attacks.with_name("attacks_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    print(f"[data] {len(data):,} rows: " + ", ".join(
        f"{a} {n:,}" for a, n in data["attack"].value_counts(sort=False).items()))

    texts = data["text"].astype(str)
    uniq, inverse = np.unique(texts.to_numpy(dtype=object).astype(str), return_inverse=True)
    results, preds = {}, []
    t0 = time.time()
    for name, path in args.model:
        scorer = JoblibScorer(path)
        print(f"[score] {name}: {len(uniq):,} distinct texts")
        p = np.concatenate([
            np.asarray(scorer.predict_proba(list(uniq[i:i + args.batch_size])), dtype=np.float64)
            for i in range(0, len(uniq), args.batch_size)])
        df = data.assign(p_ai=np.clip(p, 0.0, 1.0)[inverse])
        try:
            results[name] = robustness(df, args.max_fpr, attacks)
        except ValueError as exc:
            parser.error(str(exc))
        results[name]["path"] = str(path)
        preds.append(df[["id", "orig_id", "attack", "label", "p_ai"]].assign(model=name))
        for a, r in results[name]["attacks"].items():
            print(f"[{name}] {a}: AUROC {_fmt(r['auroc'])} ({_signed(-r['auroc_drop'])}), "
                  f"recall@thr {_pct(r['recall_thr'])}")
    score_seconds = time.time() - t0

    comparison = None
    if len(results) > 1:
        ref = results[names[0]]
        comparison = {n: roadmap_check(ref, results[n], PARAPHRASE_ATTACKS, args.fpr_tolerance)
                      for n in names[1:]}
    report = _json_safe({
        "attacks_path": str(args.attacks), "split": manifest.get("split"),
        "corpus_version": manifest.get("corpus_version"), "max_fpr": args.max_fpr,
        "skipped": manifest.get("skipped") or {}, "models": results,
        "comparison": comparison, "git": git_info(),
        "seconds": {"score": round(score_seconds, 1), "total": round(time.time() - started, 1)},
    })
    out_dir = args.out_dir or Path("reports") / "robust" / (
        names[0] if len(names) == 1 else "compare-" + "-vs-".join(names))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "robust_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
        fh.write("\n")
    md = render_md(report)
    (out_dir / "robust_report.md").write_text(md, encoding="utf-8")
    pd.concat(preds, ignore_index=True).to_parquet(out_dir / "robust_predictions.parquet",
                                                   index=False)
    print()
    print(md, end="")
    print(f"\n[save] {out_dir / 'robust_report.md'}, {out_dir / 'robust_report.json'}, "
          f"{out_dir / 'robust_predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
