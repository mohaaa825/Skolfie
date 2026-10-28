"""
scripts/make_attacks.py
------------------------
Roadmap step 9: build attack sets from the corpus (`skolfie.robust.attacks`).

Evaluation sets (default). From one split (default `test`), sample `--n` AI
and `--n` human texts, stratified by (source, domain) and seeded, keeping
texts of at least `--min-words` words and cutting them at a sentence
boundary to at most `--max-words`. Then every attack is applied to the AI
texts, and the "both" attacks (typos, synonyms, Unicode tricks) also to the
human texts. The clean texts are written too (attack "clean"), so
`scripts/eval_robust.py` needs only this file.

Training augmentation (`--share S`, needs `--split train`). A share S of
the split's AI texts each get one attack (drawn by `--weight`), and for
every "both" attack the same share of human texts as of AI texts, so typos
and the like are not a label signal. Only attacked rows are written; pass
the file to `scripts/evaluate.py --extra-train`.

Output in `--out-dir`:

    attacks.parquet        id ("<attack>:<orig id>"), orig_id, attack, text, label,
                           source, domain, generator, n_words, word_change (share of
                           the original words not kept as they were), meta (JSON:
                           model, temperature, prompt kind for the LLM attacks)
    attacks_manifest.json  settings, counts per attack, attacks skipped and why
    parts/<attack>.parquet one file per attack, reused when the same command is run
                           again (the T5 and LLM attacks are slow); --force rebuilds

By default every attack whose requirements are met runs: the LLM attacks
(llm_paraphrase, humanizer, temp_low, temp_high) need an API key
(`--provider auto` takes the first of ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY that is set) and are skipped with a note otherwise;
t5_paraphrase is skipped without a GPU unless named in `--attacks`.

Examples:

    # Test-split attack sets, 1000 AI + 1000 human texts
    python scripts/make_attacks.py --corpus data/processed/corpus.parquet \\
        --splits data/processed/splits.parquet --out-dir reports/robust/attacks-test

    # Training augmentation: attacked copies of 5% of the train texts
    python scripts/make_attacks.py --corpus ... --splits ... --split train --share 0.05 \\
        --weight t5_paraphrase=3 --out-dir reports/robust/attacks-train

    python scripts/make_attacks.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N = 1000
OUT_COLUMNS = ["id", "orig_id", "attack", "text", "label", "source", "domain", "generator",
               "n_words", "word_change", "meta"]
API_KEYS = (("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"))


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
    if not 0 < x <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return x


def _weight(value: str) -> tuple[str, float]:
    name, sep, w = value.partition("=")
    try:
        weight = float(w)
    except ValueError:
        weight = -1.0
    if not sep or not name or weight < 0:
        raise argparse.ArgumentTypeError(f"expected ATTACK=WEIGHT (>= 0), got {value!r}")
    return name, weight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_attacks.py",
        description="Build evasion attack sets (or a training augmentation set) from a split.")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--split", default="test", help="Split to sample from (default: test).")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory (default: reports/robust/attacks-<split>).")
    parser.add_argument("--attacks", default="default",
                        help="Comma-separated attack names, or 'default' (every attack in the "
                             "suite whose requirements are met). See --list.")
    parser.add_argument("--n", type=_positive_int,
                        help=f"AI texts and human texts to sample, each (default: {DEFAULT_N}; "
                             "with --share: a cap on the AI texts, default none).")
    parser.add_argument("--share", type=_fraction,
                        help="Build a training augmentation set from this share of the split "
                             "instead of evaluation sets (needs --split train).")
    parser.add_argument("--weight", type=_weight, action="append", default=[],
                        metavar="ATTACK=WEIGHT",
                        help="With --share: relative weight of an attack when assigning one to "
                             "each sampled text (default 1 each; repeatable).")
    parser.add_argument("--min-words", type=_positive_int, default=40,
                        help="Only sample texts with at least N words (default: 40).")
    parser.add_argument("--max-words", type=_positive_int, default=400,
                        help="Cut texts (and attack outputs) at a sentence boundary to at most "
                             "N words (default: 400).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild every attack instead of reusing parts/ from a previous run.")
    parser.add_argument("--list", action="store_true", help="List the attacks and exit.")
    group = parser.add_argument_group("LLM attacks (llm_paraphrase, humanizer, temp_low/high)")
    group.add_argument("--provider", default="auto",
                       help="anthropic / openai / gemini / openai-compatible / together / groq / "
                            "ollama / vllm / fake, none (no LLM attacks), or auto (default): "
                            "the first of ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY "
                            "that is set.")
    group.add_argument("--model", help="Model id (default: the provider's default).")
    group.add_argument("--base-url", help="API base URL override.")
    group.add_argument("--workers", type=_positive_int, default=8,
                       help="Concurrent API requests (default: 8).")
    group = parser.add_argument_group("T5 paraphrase and synonyms")
    group.add_argument("--t5-model", help="Seq2seq paraphrase checkpoint (default: "
                                          "humarin/chatgpt_paraphraser_on_T5_base).")
    group.add_argument("--t5-batch-size", type=_positive_int, default=32)
    group.add_argument("--device", help="cuda / cpu for T5 (default: cuda if available).")
    group.add_argument("--synonym-source", choices=("auto", "wordnet", "table"), default="auto",
                       help="auto (default): WordNet, downloaded if missing, else the small "
                            "bundled table.")
    return parser


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def cut_text(text: str, max_words: int) -> str:
    """Leading whole sentences with at most max_words words (the first one cut at a word
    boundary if it alone is longer)."""
    from skolfie.mixed.chunking import split_sentences

    if len(text.split()) <= max_words:
        return text
    end, n = 0, 0
    for s, e in split_sentences(text):
        k = len(text[s:e].split())
        if n + k > max_words:
            break
        end, n = e, n + k
    if end == 0:
        return " ".join(text.split()[:max_words])
    return text[:end]


def stratified_take(df: Any, n: int, seed: int) -> Any:
    """n rows, allocated to (source, domain) groups in proportion (largest remainder)."""
    import numpy as np
    import pandas as pd

    if len(df) <= n:
        return df
    if n <= 0:
        return df.iloc[:0]
    sizes = df.groupby(["source", "domain"], sort=True).size()
    quotas = sizes / sizes.sum() * n
    take = np.floor(quotas).astype(int)
    for key in (quotas - take).sort_values(ascending=False, kind="stable").index[
            : n - int(take.sum())]:
        take[key] += 1
    parts = [g.sample(n=int(take[key]), random_state=seed)
             for key, g in df.groupby(["source", "domain"], sort=True) if take[key] > 0]
    return pd.concat(parts).sort_values("id", kind="stable")


def word_change(orig: str, new: str) -> float:
    """Share of the original's words not kept unchanged (in order) in the new text."""
    from difflib import SequenceMatcher

    a, b = orig.split(), new.split()
    if not a:
        return 0.0
    if len(a) == len(b):  # typos and Unicode tricks keep the word count: compare in place
        return sum(x != y for x, y in zip(a, b)) / len(a)
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False)
                  .get_matching_blocks())
    return 1.0 - matched / len(a)


# --------------------------------------------------------------------------- #
# Attack selection
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def make_llm_provider(args: argparse.Namespace) -> tuple[Any, str | None]:
    """(provider, None) or (None, why there is none)."""
    from skolfie.data.generation.providers import ProviderConfigError, make_provider

    name = args.provider
    if name == "none":
        return None, "LLM attacks turned off (--provider none)"
    if name == "auto":
        name = next((p for p, key in API_KEYS if os.environ.get(key)), None)
        if name is None:
            return None, ("no API key set (ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY; "
                          "or pass --provider)")
    try:
        return make_provider(name, args.model, base_url=args.base_url), None
    except ProviderConfigError as exc:
        return None, str(exc)


def select_attacks(args: argparse.Namespace, provider: Any, provider_problem: str | None,
                   log: Any = print) -> tuple[list[str], dict[str, str]]:
    """(attacks to run, {attack: reason} for the skipped ones)."""
    from skolfie.robust.attacks import ATTACKS, get_attack, temperature_problem

    explicit = args.attacks != "default"
    names = list(ATTACKS) if not explicit else [a.strip() for a in args.attacks.split(",")
                                                if a.strip()]
    run, skipped = [], {}
    for name in dict.fromkeys(names):
        spec = get_attack(name)
        if spec.needs_api and provider is None:
            skipped[name] = f"needs an LLM API: {provider_problem}"
        elif spec.family == "regenerate" and temperature_problem(
                provider, spec.params["temperature"]):
            skipped[name] = temperature_problem(provider, spec.params["temperature"])
        elif spec.needs_gpu and not _cuda_available():
            if explicit:
                log(f"[warn] {name}: no GPU, running on CPU (slow)")
                run.append(name)
            else:
                skipped[name] = "needs a GPU (name it in --attacks to run it on CPU anyway)"
        else:
            run.append(name)
    for name, why in skipped.items():
        log(f"[skip] {name}: {why}")
    return run, skipped


def _attack_config(name: str, spec: Any, config: dict[str, Any]) -> Any:
    """The part of the run's config an attack depends on (for its cache signature)."""
    if name == "t5_paraphrase":
        return config.get("t5_model")
    if spec.needs_api:
        return config.get("llm")
    return config.get({"synonym": "synonyms", "human_edit": "donors"}.get(spec.family, ""))


def _signature(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _read_signature(path: Path) -> str | None:
    """The signature saved next to a part file, or None if missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["signature"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _replace_atomically(path: Path, write: Any) -> None:
    """Call write(tmp_path), then move the finished file into place (never half-written)."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_repo_on_path()

    from skolfie.robust.attacks import ATTACKS, get_attack

    if args.list:
        for name, spec in ATTACKS.items():
            needs = ", ".join(n for n, f in (("GPU", spec.needs_gpu), ("API key", spec.needs_api))
                              if f) or "-"
            print(f"{name:15s} {spec.applies_to:5s} {needs:8s} {spec.description}")
        print("Rates are percentages and can be changed: typo_<pct>, synonym_<pct>, "
              "homoglyph_<pct>, zero_width_<pct>, human_edit_<pct>.")
        return 0
    for flag in ("corpus", "splits"):
        path = getattr(args, flag)
        if path is None:
            parser.error(f"--{flag} is required")
        if not path.is_file():
            parser.error(f"--{flag} file not found: {path}")
    if args.attacks != "default":
        for name in args.attacks.split(","):
            try:
                get_attack(name.strip())
            except ValueError as exc:
                parser.error(str(exc))
    augment = args.share is not None

    import numpy as np
    import pandas as pd

    from skolfie.eval.report import EVAL_SPLITS, load_corpus, load_splits, load_splits_manifest
    from skolfie.robust.attacks import T5_MODEL, AttackContext, DonorPool, apply_attack

    if augment and args.split in EVAL_SPLITS:
        parser.error(f"--share builds training data; {args.split!r} is an evaluation split "
                     "(use --split train)")
    started = time.time()
    corpus = load_corpus(args.corpus)
    splits = load_splits(args.splits)
    manifest = load_splits_manifest(args.splits)
    ids = splits.loc[splits["split"] == args.split, "id"]
    frame = corpus[corpus.index.isin(ids)].reset_index(drop=True)
    if frame.empty:
        parser.error(f"no corpus texts in split {args.split!r}")
    for col in ("generator", "source", "domain"):
        if col not in frame:
            frame[col] = "unknown"
    frame["text"] = frame["text"].astype(str)
    n_words = frame["text"].str.split().str.len()
    eligible = frame[n_words >= args.min_words]
    print(f"[data] {args.split}: {len(frame):,} texts, {len(eligible):,} with >= "
          f"{args.min_words} words")

    provider, provider_problem = make_llm_provider(args)
    if provider is not None:
        print(f"[llm] provider {getattr(provider, 'provider_name', '?')}, model "
              f"{provider.model_name}")
    attacks, skipped = select_attacks(args, provider, provider_problem)
    if not attacks:
        print("error: no attack can run", file=sys.stderr)
        return 2
    specs = {a: get_attack(a) for a in attacks}

    # --- which texts get which attack ------------------------------------- #
    ai_pool = eligible[eligible["label"] == "ai"]
    human_pool = eligible[eligible["label"] == "human"]
    assigned: dict[str, pd.DataFrame] = {}
    sample = None
    if not augment:
        n = args.n or DEFAULT_N
        sample = pd.concat([stratified_take(ai_pool, n, args.seed),
                            stratified_take(human_pool, n, args.seed)], ignore_index=True)
        for a, spec in specs.items():
            assigned[a] = sample if spec.applies_to == "both" else sample[sample["label"] == "ai"]
    else:
        weights = dict(args.weight)
        unknown = set(weights) - set(attacks)
        if unknown:
            parser.error(f"--weight for attacks not being run: {', '.join(sorted(unknown))}")
        w = np.array([weights.get(a, 1.0) for a in attacks])
        if w.sum() <= 0:
            parser.error("all attack weights are 0")
        n_ai = round(args.share * len(ai_pool))
        if args.n:
            n_ai = min(n_ai, args.n)
        if n_ai == 0:
            parser.error(f"--share {args.share} of {len(ai_pool):,} eligible AI texts is no text")
        rng = np.random.default_rng(args.seed)
        ai_rows = stratified_take(ai_pool, n_ai, args.seed).reset_index(drop=True)
        choice = rng.choice(len(attacks), size=len(ai_rows), p=w / w.sum())
        human_order = human_pool.iloc[rng.permutation(len(human_pool))]
        ratio = len(human_pool) / max(len(ai_pool), 1)
        used = 0
        for k, a in enumerate(attacks):
            part = ai_rows[choice == k]
            if specs[a].applies_to == "both":
                n_h = min(round(len(part) * ratio), len(human_order) - used)
                part = pd.concat([part, human_order.iloc[used:used + n_h]], ignore_index=True)
                used += n_h
            assigned[a] = part
    cap = args.max_words
    for a in assigned:
        assigned[a] = assigned[a].assign(text=[cut_text(t, cap) for t in assigned[a]["text"]])

    # --- resources -------------------------------------------------------- #
    ctx = AttackContext(seed=args.seed, provider=provider, workers=args.workers)
    config: dict[str, Any] = {}
    families = {s.family for s in specs.values()}
    if "synonym" in families:
        from skolfie.robust.attacks import load_synonyms

        ctx.synonyms = load_synonyms(args.synonym_source, download=True)
        config["synonyms"] = ctx.synonyms.name
        print(f"[synonym] using {ctx.synonyms.name}")
    if "human_edit" in families:
        donors = frame[frame["label"] == "human"].sort_values("id")
        if sample is not None:  # prefer donors that are not themselves in the clean sample
            rest = donors[~donors["id"].isin(sample["id"])]
            donors = rest if len(rest) else donors
        ctx.donors = DonorPool(donors["text"].tolist(), donors["source"].tolist(),
                               donors["domain"].tolist())
        config["donors"] = len(donors)
    if "t5_paraphrase" in attacks:
        from skolfie.robust.attacks import T5Paraphraser

        config["t5_model"] = args.t5_model or T5_MODEL
        ctx.paraphraser = T5Paraphraser(config["t5_model"], device=args.device,
                                        batch_size=args.t5_batch_size)
    if provider is not None:
        config["llm"] = f"{getattr(provider, 'provider_name', '?')}:{provider.model_name}"

    # --- run -------------------------------------------------------------- #
    from skolfie.data.generation.providers import generator_id

    out_dir = args.out_dir or Path("reports") / "robust" / f"attacks-{args.split}"
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    frames, stats = [], {}
    for a in attacks:
        rows = assigned[a]
        spec = specs[a]
        sig = _signature(attack=a, seed=args.seed, max_words=cap,
                         config=_attack_config(a, spec, config),
                         rows=hashlib.sha256("\x00".join(rows["id"] + "\x01" + rows["text"])
                                             .encode("utf-8")).hexdigest())
        part_path, sig_path = parts_dir / f"{a}.parquet", parts_dir / f"{a}.json"
        if not args.force and part_path.is_file() and _read_signature(sig_path) == sig:
            out = pd.read_parquet(part_path)
            print(f"[{a}] reused {len(out):,} rows from {part_path}")
        else:
            print(f"[{a}] attacking {len(rows):,} texts ({spec.description})")
            t0 = time.time()
            # The old signature goes first: if this run stops before the new one is written,
            # the part is rebuilt next time instead of being taken for the old one.
            sig_path.unlink(missing_ok=True)
            new = apply_attack(a, rows, ctx)
            meta = ctx.meta.get(a) or [{} for _ in new]
            keep = [i for i, t in enumerate(new) if t is not None and t.strip()]
            texts = [cut_text(new[i], cap) for i in keep]
            src = rows.iloc[keep]
            gen = src["generator"].tolist()
            if spec.family == "regenerate":
                gen = [generator_id(m["model"]) for m in (meta[i] for i in keep)]
            out = pd.DataFrame({
                "id": [f"{a}:{i}" for i in src["id"]], "orig_id": src["id"].tolist(),
                "attack": a, "text": texts, "label": src["label"].tolist(),
                "source": src["source"].tolist(), "domain": src["domain"].tolist(),
                "generator": gen, "n_words": [len(t.split()) for t in texts],
                "word_change": [word_change(o, t) for o, t in zip(src["text"], texts)],
                "meta": [json.dumps(meta[i], sort_keys=True) for i in keep],
            }, columns=OUT_COLUMNS)
            _replace_atomically(part_path, lambda tmp: out.to_parquet(tmp, index=False))
            _replace_atomically(sig_path, lambda tmp: tmp.write_text(
                json.dumps({"signature": sig}) + "\n", encoding="utf-8"))
            print(f"[{a}] {len(out):,} of {len(rows):,} attacked in {time.time() - t0:.0f}s"
                  + (f"; {len(rows) - len(out):,} failed and dropped" if len(out) < len(rows)
                     else ""))
        frames.append(out)
        stats[a] = {
            "description": spec.description, "applies_to": spec.applies_to,
            "n_input": len(rows), "n": len(out),
            "n_ai": int((out["label"] == "ai").sum()),
            "n_human": int((out["label"] == "human").sum()),
            "n_failed": len(rows) - len(out),
            "n_unchanged": int((out["word_change"] == 0).sum()),
            "mean_word_change": float(out["word_change"].mean()) if len(out) else None,
        }

    if sample is not None:
        clean = pd.DataFrame({
            "id": "clean:" + sample["id"], "orig_id": sample["id"], "attack": "clean",
            "text": [cut_text(t, cap) for t in sample["text"]], "label": sample["label"],
            "source": sample["source"], "domain": sample["domain"],
            "generator": sample["generator"], "word_change": 0.0, "meta": "{}"})
        clean["n_words"] = clean["text"].str.split().str.len()
        frames.insert(0, clean[OUT_COLUMNS])
    result = pd.concat(frames, ignore_index=True)
    _replace_atomically(out_dir / "attacks.parquet",
                        lambda tmp: result.to_parquet(tmp, index=False))

    from skolfie.eval.report import _json_safe, git_info

    info = _json_safe({
        "mode": "augment" if augment else "eval", "split": args.split, "seed": args.seed,
        "n": args.n or (None if augment else DEFAULT_N), "share": args.share,
        "weights": dict(args.weight) or None,
        "min_words": args.min_words, "max_words": cap,
        "corpus": str(args.corpus), "corpus_version": manifest.get("corpus_version"),
        "splits_version": manifest.get("splits_version"),
        "clean": None if sample is None else {
            "n_ai": int((sample["label"] == "ai").sum()),
            "n_human": int((sample["label"] == "human").sum())},
        "attacks": stats, "skipped": skipped, "config": config,
        "git": git_info(), "seconds": round(time.time() - started, 1),
    })
    _replace_atomically(out_dir / "attacks_manifest.json", lambda tmp: tmp.write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8"))
    print()
    print(f"{'attack':16s} {'ai':>6s} {'human':>6s} {'failed':>6s} {'changed':>8s}")
    for a, s in stats.items():
        change = s["mean_word_change"]
        print(f"{a:16s} {s['n_ai']:6,d} {s['n_human']:6,d} {s['n_failed']:6,d} "
              f"{'n/a' if change is None else f'{change:.1%}':>8s}")
    for a, why in skipped.items():
        print(f"{a:16s} skipped: {why}")
    print(f"\n[save] {out_dir / 'attacks.parquet'} ({len(result):,} rows), "
          f"{out_dir / 'attacks_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
