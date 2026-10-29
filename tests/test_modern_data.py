"""
tests/test_modern_data.py
-------------------------
The modern-data retrain: the PERSUADE 2.0 loader (csv, zip, dedupe, cap),
student-essay prompts, scripts/import_claude_essays.py,
configs/splits_modern.json (held-out Haiku essays that share PERSUADE topics
with trained essays) and training-time oversampling by source.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.data import corpus as corpus_mod
from skolfie.data.generation.student import PERSUADE_PROMPTS, student_prompt_plan
from skolfie.data.sources import fresh, persuade
from skolfie.eval import splits as splits_mod
from skolfie.eval.report import oversample_rows, run_evaluation

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "sources"

# The 15 prompt names exactly as PERSUADE 2.0 spells them.
PERSUADE_NAMES = [
    "Phones and driving", "Car-free cities", "Summer projects",
    '"A Cowboy Who Rode the Waves"', "Mandatory extracurricular activities",
    "Exploring Venus", "Facial action coding system", "The Face on Mars",
    "Community service", "Grades for extracurricular activities", "Driverless cars",
    "Does the electoral college work?", "Cell phones at school", "Distance learning",
    "Seeking multiple opinions",
]


def _script(name: str):
    module_name = f"skolfie_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name,
                                                  REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _essay(i: int, prompt: str) -> str:
    return (f"Essay number {i} about {prompt}. I think this matters because students like me "
            f"deal with it every day, and my friends agree with me about reason {i}. "
            "There are many reasons and I will explain them in this essay right now.")


def _write_persuade_csv(path: Path, sizes: dict[str, int], columns: dict[str, str] | None = None,
                        repeat: int = 2) -> None:
    """A discourse-level PERSUADE-like csv: `repeat` rows per essay, full text in each."""
    cols = {"id": "essay_id_comp", "text": "full_text", "prompt": "prompt_name",
            "grade": "grade_level", "ell": "ell_status", **(columns or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["discourse_id", cols["id"], "discourse_text", cols["text"], cols["prompt"],
                    cols["grade"], cols["ell"]])
        n = 0
        for prompt, k in sizes.items():
            for i in range(k):
                essay_id = f"P{list(sizes).index(prompt):02d}E{i:04d}"
                for _ in range(repeat):
                    n += 1
                    w.writerow([n, essay_id, "part", _essay(i, prompt), prompt, 6 + i % 7,
                                "Yes" if i % 3 == 0 else "No"])


# --------------------------------------------------------------------------- #
# PERSUADE loader
# --------------------------------------------------------------------------- #
def test_persuade_fixture_records() -> None:
    records = list(persuade.load(FIXTURE, download=False))
    assert len(records) == 3  # 9 discourse rows, 3 essays
    by_id = {r.id: r for r in records}
    rec = by_id["persuade:phones-and-driving:g10:ell-no:A1B2C3D4E5F6"]
    assert (rec.label, rec.generator, rec.domain, rec.subdomain, rec.source) == (
        "human", "human", "academic", "student_essay", "persuade")
    assert rec.split_hint == "train"
    assert "\n\n" in rec.text  # paragraph breaks kept for the corpus builder
    assert "persuade:cowboy-who-rode-the-waves:g8:ell-yes:0FF1CE000001" in by_id
    assert "persuade:electoral-college:g11:ell-no:0FF1CE000002" in by_id


def test_prompt_slugs_match_the_student_prompts() -> None:
    assert sorted(persuade.prompt_slug(n) for n in PERSUADE_NAMES) == sorted(PERSUADE_PROMPTS)


def test_persuade_reads_a_zip_and_tolerates_other_column_names(tmp_path: Path) -> None:
    sizes = {"Phones and driving": 3, "Exploring Venus": 2}
    plain = tmp_path / "plain"
    _write_persuade_csv(plain / "persuade" / "train.csv", sizes)
    expected = [r.to_dict() for r in persuade.load(plain, download=False, max_essays=0)]
    assert len(expected) == 5

    zipped = tmp_path / "zipped" / "persuade"
    zipped.mkdir(parents=True)
    with zipfile.ZipFile(zipped / "persuade.zip", "w") as zf:
        zf.write(plain / "persuade" / "train.csv", "persuade_2.0/train.csv")
    assert [r.to_dict() for r in persuade.load(zipped.parent, download=False,
                                               max_essays=0)] == expected

    # A zip next to its own unpacked csv is not read twice (ids would be dropped anyway).
    (zipped / "train.csv").write_bytes((plain / "persuade" / "train.csv").read_bytes())
    assert len(list(persuade.load(zipped.parent, download=False, max_essays=0))) == 5

    other = tmp_path / "other"
    _write_persuade_csv(other / "persuade" / "essays.csv", sizes,
                        columns={"id": "Essay_ID", "text": "Text", "prompt": "prompt",
                                 "grade": "Grade", "ell": "ELL"})
    recs = list(persuade.load(other, download=False, max_essays=0))
    assert sorted(r.id.split(":", 1)[1] for r in recs) == sorted(
        e["id"].split(":", 1)[1] for e in expected)


def test_persuade_missing_column_and_optional_columns(tmp_path: Path) -> None:
    d = tmp_path / "persuade"
    d.mkdir()
    (d / "x.csv").write_text("essay_id,prompt_name\n1,Exploring Venus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no text column"):
        list(persuade.load(tmp_path, download=False))
    (d / "x.csv").write_text(
        'essay_id,text,prompt_name\nE1,"' + _essay(1, "x") + '",Exploring Venus\n',
        encoding="utf-8")
    (rec,) = persuade.load(tmp_path, download=False)
    assert rec.id == "persuade:exploring-venus:gNA:ell-na:E1"


def test_persuade_cap_is_equal_per_prompt_and_deterministic(tmp_path: Path) -> None:
    sizes = {"Phones and driving": 1, "Exploring Venus": 4, "Distance learning": 9}
    _write_persuade_csv(tmp_path / "persuade" / "train.csv", sizes)
    recs = list(persuade.load(tmp_path, download=False, max_essays=7))
    per_prompt = Counter(r.id.split(":")[1] for r in recs)
    assert sum(per_prompt.values()) == 7
    # The 1-essay prompt gives its unused share to the others, which split it evenly.
    assert per_prompt == {"phones-and-driving": 1, "exploring-venus": 3,
                          "distance-learning": 3}
    again = list(persuade.load(tmp_path, download=False, max_essays=7))
    assert [r.id for r in recs] == [r.id for r in again]
    other_seed = list(persuade.load(tmp_path, download=False, max_essays=7, seed=1))
    assert {r.id for r in other_seed} != {r.id for r in recs}
    assert len(list(persuade.load(tmp_path, download=False, max_essays=0))) == 14
    assert len(list(persuade.load(tmp_path, download=False, max_essays=100))) == 14
    with pytest.raises(ValueError, match="max_essays"):
        list(persuade.load(tmp_path, download=False, max_essays=-1))


def test_stratified_cap_never_exceeds_and_fills() -> None:
    by_prompt = {f"p{k}": [f"p{k}-{i}" for i in range(n)]
                 for k, n in enumerate([0, 2, 5, 50, 51])}
    for cap in (1, 3, 7, 40, 100, 107):
        keep = persuade.stratified_cap(by_prompt, cap)
        assert len(keep) == cap
        counts = Counter(i.split("-")[0] for i in keep)
        open_counts = [counts[p] for p, ids in by_prompt.items() if counts[p] < len(ids)]
        assert not open_counts or max(open_counts) - min(open_counts) <= 1


def test_persuade_finalize_download(tmp_path: Path) -> None:
    html = tmp_path / "a" / "persuade_download"
    html.parent.mkdir()
    html.write_text("<!DOCTYPE html><html>Quota exceeded</html>")
    with pytest.raises(RuntimeError, match="web page"):
        persuade.finalize_download(html)
    assert not html.exists()

    data = tmp_path / "b" / "persuade_download"
    data.parent.mkdir()
    data.write_text("essay_id,full_text,prompt_name\n")
    assert persuade.finalize_download(data).name == "persuade_download.csv"

    z = tmp_path / "c" / "persuade_download"
    z.parent.mkdir()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("train.csv", "essay_id,full_text,prompt_name\n")
    assert persuade.finalize_download(z).name == "persuade_download.zip"


def test_build_corpus_passes_source_options(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_persuade_csv(raw / "persuade" / "train.csv",
                        {"Phones and driving": 6, "Exploring Venus": 6})
    registry = {"persuade": persuade}
    result = corpus_mod.build_corpus(["persuade"], raw, tmp_path / "out", download=False,
                                     registry=registry,
                                     source_options={"persuade": {"max_essays": 4}})
    assert result.n_records == 4
    assert result.manifest["parameters"]["source_options"] == {"persuade": {"max_essays": 4}}
    with pytest.raises(ValueError, match="not being built"):
        corpus_mod.build_corpus(["persuade"], raw, tmp_path / "out2", download=False,
                                registry=registry, source_options={"fresh": {}})

    build = _script("build_corpus")
    assert build.main(["--sources", "persuade", "--raw-dir", str(raw), "--out-dir",
                       str(tmp_path / "cli"), "--no-download", "--persuade-max-essays", "3",
                       "-q"]) == 0
    manifest = json.loads((tmp_path / "cli" / "manifest.json").read_text())
    assert manifest["n_records"] == 3
    assert build.main(["--sources", "persuade", "--raw-dir", str(raw), "--out-dir",
                       str(tmp_path / "cli0"), "--no-download", "--persuade-max-essays", "0",
                       "-q"]) == 0
    assert json.loads((tmp_path / "cli0" / "manifest.json").read_text())["n_records"] == 12


# --------------------------------------------------------------------------- #
# Student prompts
# --------------------------------------------------------------------------- #
def test_student_prompt_plan_is_deterministic_and_groupable() -> None:
    plan = student_prompt_plan(40, seed=3)
    assert plan == student_prompt_plan(40, seed=3)
    assert len(plan) == 40 and len({s.prompt_id for s in plan}) == 40
    persuade_specs = [s for s in plan if s.template_id.startswith("persuade-")]
    assert len(persuade_specs) == 24
    for spec in persuade_specs:
        assert spec.template_id[len("persuade-"):] in PERSUADE_PROMPTS
        rid = f"fresh:claude-opus-5-5:{spec.prompt_id}:0"
        grade = spec.style.split("+")[0]
        assert splits_mod.group_key(rid, "fresh") == f"fresh:{spec.template_id}.{grade}"


# --------------------------------------------------------------------------- #
# import_claude_essays.py
# --------------------------------------------------------------------------- #
def _plan_file(tmp_path: Path) -> tuple[Path, list]:
    specs = student_prompt_plan(6, seed=0)
    entries = []
    for i, spec in enumerate(specs):
        model, generator = (("opus", "claude-opus-5-5") if i % 2 == 0
                            else ("haiku", "claude-haiku-5-5"))
        entries.append({**dataclasses.asdict(spec), "model": model, "generator": generator})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(entries))
    return path, entries


def test_import_claude_essays(tmp_path: Path, capsys) -> None:
    imp = _script("import_claude_essays")
    plan, entries = _plan_file(tmp_path)
    essays = tmp_path / "essays"
    body = " ".join(["This is a sentence a student might write about the topic."] * 6)
    files = {
        0: f"# A Title For The Essay\n\n{body}\n\nAnother paragraph here, with more words.",
        1: f"Sure! Here's an essay about it:\n\n{body}",
        2: "",                                  # empty
        3: "Too short to count.",              # too short
        # 4: missing
        5: "I'm sorry, but I can't help with writing that essay for you today, sorry."
           + " " + body,                      # refusal
    }
    for i, text in files.items():
        e = entries[i]
        (essays / e["model"]).mkdir(parents=True, exist_ok=True)
        (essays / e["model"] / f"{e['prompt_id']}.txt").write_text(text, encoding="utf-8")

    out = tmp_path / "fresh"
    # A record from generate_ai_texts.py in the same file must survive.
    out.mkdir()
    foreign = {"id": "fresh:claude-opus-5-5:other-prompt:0", "text": body, "label": "ai",
               "generator": "claude-opus-5-5", "domain": "academic", "subdomain": "essay",
               "source": "fresh", "split_hint": "", "n_words": 60}
    (out / "claude-opus-5-5.jsonl").write_text(json.dumps(foreign) + "\n")

    assert imp.main(["--plan", str(plan), "--essays-dir", str(essays),
                     "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "claude-opus-5-5" in printed and "claude-haiku-5-5" in printed
    first = {p.name: p.read_bytes() for p in out.glob("*.jsonl")}
    assert set(first) == {"claude-opus-5-5.jsonl", "claude-haiku-5-5.jsonl"}

    opus = [json.loads(line) for line in first["claude-opus-5-5.jsonl"].decode().splitlines()]
    assert opus[0] == foreign
    rec = opus[1]
    assert rec["id"] == f"fresh:claude-opus-5-5:{entries[0]['prompt_id']}:0"
    assert not rec["text"].startswith("#") and rec["text"].startswith("This is a sentence")
    assert rec["subdomain"] == entries[0]["subdomain"] and rec["label"] == "ai"
    haiku = [json.loads(line) for line in first["claude-haiku-5-5.jsonl"].decode().splitlines()]
    assert len(haiku) == 1 and not haiku[0]["text"].startswith("Sure")

    counts = imp.import_essays(imp.load_plan(plan), essays, out)
    assert counts["claude-opus-5-5"]["written"] == 1
    assert counts["claude-opus-5-5"]["empty"] == 1
    assert counts["claude-haiku-5-5"]["written"] == 1
    assert counts["claude-haiku-5-5"]["too_short"] == 1
    assert counts["claude-haiku-5-5"]["refusal"] == 1
    assert counts["claude-opus-5-5"]["missing"] == 1
    # Idempotent: the second run left the same bytes.
    assert {p.name: p.read_bytes() for p in out.glob("*.jsonl")} == first

    # The fresh loader reads the result.
    loaded = list(fresh.load(tmp_path, download=False))
    assert len(loaded) == 3

    # --min-words skips an essay, but its earlier record is kept.
    counts = imp.import_essays(imp.load_plan(plan), essays, out, min_words=1000)
    assert counts["claude-opus-5-5"]["kept_old"] == 1
    assert {p.name: p.read_bytes() for p in out.glob("*.jsonl")} == first


def test_import_claude_essays_rejects_bad_plans(tmp_path: Path) -> None:
    imp = _script("import_claude_essays")
    plan, entries = _plan_file(tmp_path)
    (tmp_path / "essays").mkdir()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{k: v for k, v in entries[0].items() if k != "generator"}]))
    assert imp.main(["--plan", str(bad), "--essays-dir", str(tmp_path / "essays")]) == 2
    bad.write_text(json.dumps([entries[0], entries[0]]))
    assert imp.main(["--plan", str(bad), "--essays-dir", str(tmp_path / "essays")]) == 2
    bad.write_text(json.dumps([{**entries[0], "prompt_id": "../x"}]))
    assert imp.main(["--plan", str(bad), "--essays-dir", str(tmp_path / "essays")]) == 2
    # No usable essay at all: exit 1, nothing written.
    out = tmp_path / "out"
    assert imp.main(["--plan", str(plan), "--essays-dir", str(tmp_path / "essays"),
                     "--out-dir", str(out)]) == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# configs/splits_modern.json
# --------------------------------------------------------------------------- #
def _modern_frame() -> pd.DataFrame:
    """Claude essays on PERSUADE topics (all three models per topic) and on bank topics,
    PERSUADE human essays, and a few old-corpus rows."""
    rows = []

    def add(id_, label, gen, source, domain="academic"):
        rows.append({"id": id_, "label": label, "generator": gen, "domain": domain,
                     "source": source, "n_words": 300})

    for name in sorted(PERSUADE_PROMPTS):
        for grade in range(6, 13):
            for gen in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
                pid = f"persuade-{name}.grade{grade}.plain.w350.t07.n{len(rows):04d}"
                add(f"fresh:{gen}:{pid}:0", "ai", gen, "fresh")
            add(f"persuade:{name}:g{grade}:ell-no:E{len(rows)}", "human", "human", "persuade")
    for t in range(40):
        for gen in ("claude-opus-5-5", "claude-haiku-5-5"):
            add(f"fresh:{gen}:acad-essay.q{t:02d}.plain.w300.t10.college1:0", "ai", gen, "fresh")
    for i in range(200):
        add(f"m4gt:x:{i}", "ai", "gpt-4" if i % 4 == 0 else "gpt-3.5-turbo", "m4gt")
        add(f"m4gt:h:{i}", "human", "human", "m4gt")
    return pd.DataFrame(rows)


def test_modern_config_extends_the_old_one() -> None:
    old = json.loads((REPO_ROOT / "configs" / "splits.json").read_text())
    new = json.loads((REPO_ROOT / "configs" / "splits_modern.json").read_text())
    for key in ("seed", "fractions", "held_out_domains", "short_text_max_words"):
        assert new[key] == old[key]
    assert set(old["held_out_generators"]) < set(new["held_out_generators"])
    assert old["generator_families"].items() <= new["generator_families"].items()
    cfg = splits_mod.load_config(REPO_ROOT / "configs" / "splits_modern.json")
    assert cfg.family("claude-opus-5-5") == cfg.family("claude-sonnet-5-5") == "claude"
    assert "claude-haiku" in cfg.held_out_families and "claude" not in cfg.held_out_families


def test_old_config_hash_is_unchanged() -> None:
    # splits_version of the existing corpus must not move with the new option.
    cfg = splits_mod.load_config(REPO_ROOT / "configs" / "splits.json")
    assert cfg.config_hash() == (
        "a533dfd53b9625886d8c7758887c7920e91f3163c74b49dc5f3bd03ace370fb4")
    assert "shared_topic_groups" not in cfg.to_dict()


def test_modern_splits_hold_out_haiku_without_emptying_train() -> None:
    df = _modern_frame()
    cfg = splits_mod.load_config(REPO_ROOT / "configs" / "splits_modern.json")
    memberships = splits_mod.make_splits(df, cfg)  # runs check_guarantees
    primary = splits_mod.assign_splits(df, cfg)
    is_haiku = df["generator"] == "claude-haiku-5-5"
    assert (primary[is_haiku] == "test_ood_generator").all()
    train_ids = set(memberships.loc[memberships["split"] == "train", "id"])
    assert not train_ids & set(df.loc[is_haiku, "id"])

    persuade_ai = df["id"].str.contains(":persuade-") & ~is_haiku
    # Every PERSUADE group holds a Haiku essay, yet most opus/sonnet essays still train.
    assert (primary[persuade_ai] == "train").mean() > 0.6
    # Bank-topic groups (AI-only topics) are still tainted: no opus essay on a Haiku topic
    # is trained on.
    bank_opus = df["id"].str.contains(":acad-essay.") & ~is_haiku
    assert (primary[bank_opus] == "test").all()
    # Human PERSUADE test essays join test_ood_generator, for its false-positive rate.
    ood = memberships.loc[memberships["split"] == "test_ood_generator", "id"]
    assert ood.str.startswith("persuade:").any()

    # Without the option, every PERSUADE group would be pulled out of train.
    strict = dataclasses.replace(cfg, shared_topic_groups=[])
    strict_primary = splits_mod.assign_splits(df, strict)
    assert (strict_primary[persuade_ai] == "train").sum() == 0
    splits_mod.make_splits(df, strict)


def test_shared_topic_rows_are_still_checked() -> None:
    df = _modern_frame()
    cfg = splits_mod.load_config(REPO_ROOT / "configs" / "splits_modern.json")
    memberships = splits_mod.make_splits(df, cfg)
    haiku_id = df.loc[df["generator"] == "claude-haiku-5-5", "id"].iloc[0]
    leaked = memberships.copy()
    leaked.loc[leaked["id"] == haiku_id, "split"] = "train"
    with pytest.raises(splits_mod.SplitLeakError, match="held-out-family"):
        splits_mod.check_guarantees(df, leaked, cfg)
    # An opus essay in a non-shared (bank) group may not straddle its group.
    bank = df["id"].str.contains(":acad-essay.q00.") & (df["generator"] == "claude-opus-5-5")
    moved = memberships.copy()
    moved.loc[moved["id"] == df.loc[bank, "id"].iloc[0], "split"] = "train"
    with pytest.raises(splits_mod.SplitLeakError):
        splits_mod.check_guarantees(df, moved, cfg)


# --------------------------------------------------------------------------- #
# Oversampling
# --------------------------------------------------------------------------- #
def test_oversample_rows() -> None:
    df = pd.DataFrame({"text": list("abcde"), "label": ["ai", "human", "ai", "human", "ai"],
                       "source": ["fresh", "hc3", "fresh", "persuade", "hc3"]})
    out, info = oversample_rows(df, {"fresh": 3, "persuade": 1})
    assert Counter(out["text"]) == {"a": 3, "c": 3, "b": 1, "d": 1, "e": 1}
    assert info == {"column": "source", "factors": {"fresh": 3, "persuade": 1},
                    "rows": {"fresh": 2, "persuade": 1}, "n_added": 4}
    assert oversample_rows(df, None) == (df, {})
    with pytest.raises(ValueError, match="no train rows"):
        oversample_rows(df, {"frsh": 2})
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="integer"):
            oversample_rows(df, {"fresh": bad})


class _Recorder:
    name = "recorder"

    def fit(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)
        return self

    def predict_proba(self, texts):
        return np.full(len(texts), 0.5)


def test_run_evaluation_oversamples_train_only() -> None:
    ids = [f"r{i}" for i in range(8)]
    corpus = pd.DataFrame({
        "id": ids, "text": [f"text {i}" for i in range(8)],
        "label": ["ai", "human"] * 4,
        "generator": ["claude-opus-5-5", "human"] * 4,
        "domain": "academic", "source": ["fresh", "persuade", "hc3", "hc3"] * 2})
    corpus.index = pd.Index(ids)
    splits = pd.DataFrame({"id": ids, "split": ["train"] * 6 + ["test"] * 2})
    scorer = _Recorder()
    report, preds = run_evaluation(scorer, corpus, splits, oversample={"fresh": 4},
                                   eval_splits=("test",), log=lambda _: None)
    counts = Counter(scorer.texts)
    assert counts["text 0"] == counts["text 4"] == 4 and counts["text 1"] == 1
    assert len(scorer.texts) == 6 + 2 * 3
    assert report["train"]["n"] == 6
    assert report["train"]["oversample"]["n_added"] == 6
    assert report["sampling"]["oversample"] == {"fresh": 4}
    assert len(preds) == 2  # evaluation untouched


def test_evaluate_cli_oversample(tmp_path: Path, capsys) -> None:
    evaluate = _script("evaluate")
    assert evaluate.parse_oversample(["fresh=6", "persuade=1, x=2"]) == {
        "fresh": 6, "persuade": 1, "x": 2}
    for bad in (["fresh"], ["fresh=0"], ["=3"], ["fresh=two"], ["fresh=2,fresh=3"]):
        with pytest.raises(Exception):
            evaluate.parse_oversample(bad)

    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        ai = i % 2 == 0
        words = ["furthermore", "crucial", "robust"] if ai else ["lol", "kinda", "nope"]
        rows.append({"id": f"c{i:03d}", "text": " ".join(rng.choice(words, 40)),
                     "label": "ai" if ai else "human",
                     "generator": "claude-opus-5-5" if ai else "human", "domain": "academic",
                     "subdomain": "student_essay", "source": "fresh" if ai else "persuade",
                     "split_hint": "", "n_words": 40})
    corpus = pd.DataFrame(rows)
    corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
    pd.DataFrame({"id": corpus["id"], "split": ["train"] * 40 + ["val"] * 10 + ["test"] * 10}
                 ).to_parquet(tmp_path / "splits.parquet", index=False)
    args = ["--corpus", str(tmp_path / "corpus.parquet"), "--splits",
            str(tmp_path / "splits.parquet"), "--model", "length"]
    assert evaluate.main([*args, "--out-dir", str(tmp_path / "r"), "--oversample",
                          "fresh=3"]) == 0
    report = json.loads((tmp_path / "r" / "report.json").read_text())
    assert report["train"]["oversample"]["n_added"] == 40
    assert "oversampled by source: fresh 20 x3" in (tmp_path / "r" / "report.md").read_text()
    capsys.readouterr()
    with pytest.raises(SystemExit):
        evaluate.main([*args, "--out-dir", str(tmp_path / "bad"), "--oversample", "fresh=0"])
    with pytest.raises(ValueError, match="no train rows"):
        evaluate.main([*args, "--out-dir", str(tmp_path / "bad"), "--oversample", "nope=2"])
