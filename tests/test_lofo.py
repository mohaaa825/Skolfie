"""
tests/test_lofo.py
-------------------
Leave-one-family-out evaluation (`skolfie.eval.lofo`, scripts/eval_lofo.py):
split construction from configs/splits_modern.json (the family held out,
guarantees checked, PERSUADE human essays never moved), the metrics on fake
scores, one run per family with a fake scorer, the CLI, and the notebook
section in retrain_modern.ipynb.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.data.generation.student import PERSUADE_PROMPTS
from skolfie.eval import lofo
from skolfie.eval import splits as splits_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "splits_modern.json"
CLAUDE = ("claude-opus-5-5", "claude-sonnet-5-5")
GEMINI = ("gemini-3.1-flash-lite", "gemini-3.5-flash")


def modern_corpus() -> pd.DataFrame:
    """Claude (opus, sonnet, haiku) on PERSUADE and bank topics, Gemini on its own topics,
    PERSUADE human essays and old-corpus rows; texts carry their label as a marker."""
    rows = []

    def add(id_, label, gen, source, domain="academic"):
        marker = "zeta" if label == "ai" else "omega"
        rows.append({"id": id_, "label": label, "generator": gen, "domain": domain,
                     "source": source, "n_words": 300, "text": f"{marker} {id_}"})

    for name in sorted(PERSUADE_PROMPTS):
        for grade in range(6, 13):
            for gen in (*CLAUDE, "claude-haiku-5-5"):
                pid = f"persuade-{name}.grade{grade}.plain.w350.t07.n{len(rows):04d}"
                add(f"fresh:{gen}:{pid}:0", "ai", gen, "fresh")
            for k in range(2):
                add(f"persuade:{name}:g{grade}:ell-no:E{len(rows)}", "human", "human",
                    "persuade")
    for t in range(30):
        for gen in ("claude-opus-5-5", "claude-haiku-5-5"):
            add(f"fresh:{gen}:acad-essay.q{t:02d}.plain.w300.t10.college1:0", "ai", gen, "fresh")
    for t in range(40):
        gen = GEMINI[t % 2]
        add(f"fresh:{gen}:gemini-topic-{t}.9th-grader.persuasive-essay.w350.n0001:0", "ai", gen,
            "fresh")
    for i in range(200):
        add(f"m4gt:x:{i}", "ai", "gpt-4" if i % 4 == 0 else "gpt-3.5-turbo", "m4gt")
        add(f"m4gt:h:{i}", "human", "human", "m4gt")
    df = pd.DataFrame(rows)
    df.index = pd.Index(df["id"].to_numpy())
    return df


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return modern_corpus()


@pytest.fixture(scope="module")
def base() -> splits_mod.SplitConfig:
    # The synthetic texts are built from ids, so they look like near duplicates of each
    # other; near-duplicate merging is tested in test_near_dup.py.
    return dataclasses.replace(splits_mod.load_config(CONFIG), near_duplicate_threshold=0.0)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_family_generators(base, corpus) -> None:
    gens = corpus["generator"].unique()
    assert lofo.family_generators(base, "claude") == list(CLAUDE)
    assert lofo.family_generators(base, "claude-haiku") == ["claude-haiku-5-5"]
    assert set(GEMINI) <= set(lofo.family_generators(base, "gemini"))
    assert lofo.family_generators(base, "gpt-3.5") == ["chatgpt", "gpt-3.5-turbo"]
    # An unmapped generator is its own family.
    assert lofo.family_generators(base, "mystery-7b", ["mystery-7b", "human"]) == ["mystery-7b"]
    assert set(lofo.DEFAULT_FAMILIES) <= set(lofo.ai_families(corpus, base))
    cfg = lofo.lofo_config(base, "claude", gens)
    assert cfg.held_out_generators[:len(base.held_out_generators)] == base.held_out_generators
    assert set(CLAUDE) <= set(cfg.held_out_generators)
    assert lofo.lofo_config(base, "claude-haiku", gens).held_out_generators == \
        base.held_out_generators
    assert base.held_out_generators == ["gpt-4", "cohere", "claude-haiku-5-5"]  # unchanged


@pytest.mark.parametrize("family", ["claude", "claude-haiku", "gemini", "gpt-3.5"])
def test_lofo_splits_hold_out_the_family_and_keep_persuade(base, corpus, family) -> None:
    cfg, memberships = lofo.lofo_splits(corpus, base, family)  # check_guarantees ran
    splits_mod.check_guarantees(corpus, memberships, cfg)
    fam = lofo.family_mask(corpus, cfg, family)
    assert fam.any()
    fit_ids = set(memberships.loc[memberships["split"].isin(["train", "val"]), "id"])
    assert not fit_ids & set(corpus.loc[fam, "id"])
    primary = splits_mod.assign_splits(corpus, cfg)
    assert primary[fam & (corpus["domain"] != "news").to_numpy()].eq(
        "test_ood_generator").all()
    # PERSUADE human essays: the same splits as in the base config, most of them in train.
    assert lofo.persuade_primary_unchanged(corpus, base, cfg)
    counts = lofo.split_counts(corpus, memberships, cfg)
    n_persuade = int(lofo.persuade_human_mask(corpus).sum())
    assert counts["persuade_humans_by_split"]["train"] > 0.6 * n_persuade
    base_counts = lofo.split_counts(corpus, splits_mod.make_splits(corpus, base), base)
    assert counts["persuade_humans_by_split"] == base_counts["persuade_humans_by_split"]
    assert family not in counts["fresh_train_rows"]


def test_holding_out_claude_keeps_gemini_and_most_persuade_topics(base, corpus) -> None:
    cfg, memberships = lofo.lofo_splits(corpus, base, "claude")
    counts = lofo.split_counts(corpus, memberships, cfg)
    # Nothing of Claude trains; Gemini's own topics are untouched.
    assert set(counts["fresh_train_rows"]) == {"gemini"}
    base_counts = lofo.split_counts(corpus, splits_mod.make_splits(corpus, base), base)
    assert counts["fresh_train_rows"]["gemini"] == base_counts["fresh_train_rows"]["gemini"]
    # Opus essays on bank topics: the group is tainted, the essays go to test_ood_generator.
    bank = corpus["id"].str.contains(":acad-essay.") & corpus["generator"].eq("claude-opus-5-5")
    assert splits_mod.assign_splits(corpus, cfg)[bank].eq("test_ood_generator").all()


def test_scoring_memberships(base, corpus) -> None:
    cfg, memberships = lofo.lofo_splits(corpus, base, "gemini")
    scored = lofo.scoring_memberships(corpus, memberships, cfg, "gemini")
    by = scored.groupby("split")["id"].apply(set)
    assert by["train"] == set(memberships.loc[memberships["split"] == "train", "id"])
    assert by["val"] == set(memberships.loc[memberships["split"] == "val", "id"])
    gen = corpus.set_index("id")["generator"]
    assert {cfg.family(gen[i]) for i in by["test_ood_generator"]} == {"gemini"}
    assert all(i.startswith("persuade:") for i in by["test"])
    assert set(scored["split"]) == {"train", "val", "test", "test_ood_generator"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_family_metrics_on_fake_scores() -> None:
    # val: 100 humans at 0.00..0.99, so the 1% cut-off is the 99th-quantile human (0.99,
    # "higher" method) -> nothing at or below it counts as AI.
    y_val = ["human"] * 100 + ["ai"] * 10
    p_val = list(np.arange(100) / 100) + [1.0] * 10
    m = lofo.family_metrics(y_val, p_val, p_family=[0.995, 0.999, 0.5, 0.2],
                            p_human=[0.1, 0.2, 0.995, 0.3], max_fpr=0.01)
    assert m["ai_threshold"] == pytest.approx(0.99)
    assert m["recall"] == 0.5 and m["persuade_fpr"] == 0.25
    assert m["mean_p_ai"] == pytest.approx(np.mean([0.995, 0.999, 0.5, 0.2]))
    # AUROC of (family vs humans): 16 pairs; family 0.995 ties the human 0.995 (half).
    wins = sum((f > h) + 0.5 * (f == h) for f in [0.995, 0.999, 0.5, 0.2]
               for h in [0.1, 0.2, 0.995, 0.3])
    assert m["auroc_vs_persuade"] == pytest.approx(wins / 16)
    assert (m["n_val"], m["n_family"], m["n_persuade_human"]) == (110, 4, 4)
    # The same cut-off as the stacker's ai_threshold.
    from skolfie.models.stacker import operating_points

    assert m["ai_threshold"] == operating_points(y_val, p_val, max_fpr=0.01)["ai_threshold"]
    empty = lofo.family_metrics(y_val, p_val, [], [0.1])
    assert math.isnan(empty["recall"]) and math.isnan(empty["auroc_vs_persuade"])


def test_render_lofo_md() -> None:
    row = {"family": "claude", "held_out_generators": list(CLAUDE), "n_family": 12,
           "recall": 0.75, "auroc_vs_persuade": 0.9, "mean_p_ai": None, "n_persuade_human": 40,
           "persuade_fpr": 0.025, "ai_threshold": 0.8, "persuade_split_unchanged": True}
    md = lofo.render_lofo_md({"model": "features", "config": "c.json", "max_fpr": 0.01,
                              "base_held_out_generators": ["gpt-4"], "families": [row]})
    assert "| claude | claude-opus-5-5, claude-sonnet-5-5 | 12 | 75.0% | 0.900 | n/a | 40 |" in md
    assert "2.5%" in md and "kept their splits" in md


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
class MarkerScorer:
    """P(ai) high for texts with the AI marker, except for `blind` generators' ids."""

    name = "marker"
    pretrained = False

    def __init__(self, blind: tuple[str, ...] = ()) -> None:
        self.blind = blind
        self.fit_texts: list[str] = []

    def fit(self, texts, labels):
        self.fit_texts = list(texts)
        return self

    def predict_proba(self, texts):
        rng = np.random.default_rng(len(texts))
        p = np.array([0.9 if t.startswith("zeta") and not any(b in t for b in self.blind)
                      else 0.1 for t in texts])
        return np.clip(p + rng.uniform(-0.05, 0.05, len(texts)), 0, 1)


def test_run_family(base, corpus) -> None:
    scorers: list[MarkerScorer] = []

    def factory():
        scorers.append(MarkerScorer(blind=("gemini",)))
        return scorers[-1]

    logs: list[str] = []
    row, report, preds = lofo.run_family(factory, corpus, base, "gemini",
                                         oversample={"fresh": 3}, log=logs.append)
    assert row["family"] == "gemini" and row["persuade_split_unchanged"]
    assert row["recall"] <= 0.05 and row["persuade_fpr"] <= 0.05  # blind: chance level
    assert row["n_family"] == 40 and row["n_persuade_human"] > 0
    assert set(row["family_generators_scored"]) == set(GEMINI)
    assert not any("gemini" in t for t in scorers[0].fit_texts)
    assert report["train"]["oversample"]["factors"] == {"fresh": 3}
    assert set(preds["split"]) == set(lofo.LOFO_SPLITS)
    # A family the scorer does catch.
    row, _, _ = lofo.run_family(lambda: MarkerScorer(), corpus, base, "claude",
                                log=lambda m: None)
    assert row["recall"] > 0.9 and row["auroc_vs_persuade"] > 0.9
    assert row["mean_p_ai"] > 0.8


def test_run_family_skips_oversampling_a_source_with_no_train_rows(base, corpus) -> None:
    only_claude = corpus[~corpus["generator"].isin(GEMINI)]
    logs: list[str] = []
    row, report, _ = lofo.run_family(lambda: MarkerScorer(), only_claude, base, "claude",
                                     oversample={"fresh": 3}, log=logs.append)
    assert "oversample" not in report["train"]
    assert any("not oversampled" in m for m in logs)


def test_eval_lofo_cli(base, corpus, tmp_path, monkeypatch, capsys) -> None:
    import skolfie.models.baselines as baselines
    from tests.test_modern_data import _script

    corpus.reset_index(drop=True).to_parquet(tmp_path / "corpus.parquet", index=False)
    specs = []

    def fake_make_scorer(spec, seed=0, **options):
        specs.append((spec, {k: v for k, v in options.items() if v is not None}))
        return MarkerScorer(blind=("haiku",))

    monkeypatch.setattr(baselines, "make_scorer", fake_make_scorer)
    script = _script("eval_lofo")
    out = tmp_path / "lofo"
    assert script.main(["--corpus", str(tmp_path / "corpus.parquet"), "--config", str(CONFIG),
                        "--families", "gpt-3.5", "--oversample", "fresh=3",
                        "--keep-sources", "fresh,persuade", "--max-train", "500",
                        "--lm-batch-size", "8", "--out-dir", str(out)]) == 0
    result = json.loads((out / "lofo.json").read_text())
    assert [r["family"] for r in result["families"]] == [*lofo.DEFAULT_FAMILIES, "gpt-3.5"]
    assert [s for s, _ in specs] == ["features"] * 4
    assert specs[0][1] == {"lm_batch_size": 8}
    by = {r["family"]: r for r in result["families"]}
    assert by["claude-haiku"]["recall"] <= 0.05 and by["claude"]["recall"] > 0.9
    assert all(r["persuade_split_unchanged"] for r in result["families"])
    assert result["sampling"]["oversample"] == {"fresh": 3}
    md = (out / "lofo.md").read_text()
    assert md.count("\n| ") == 1 + 4 and "| gpt-3.5 |" in md
    assert md in capsys.readouterr().out
    for family in by:
        assert (out / family / "report.json").is_file()
        assert (out / family / "predictions.parquet").is_file()
    assert json.loads((out / "claude" / "report.json").read_text())["lofo"]["family"] == "claude"

    with pytest.raises(SystemExit):  # a family with no essays in the corpus
        script.main(["--corpus", str(tmp_path / "corpus.parquet"), "--families", "bloom",
                     "--no-default-families", "--out-dir", str(tmp_path / "bad")])
    with pytest.raises(SystemExit):
        script.main(["--corpus", str(tmp_path / "corpus.parquet"), "--no-default-families",
                     "--out-dir", str(tmp_path / "bad")])
    with pytest.raises(SystemExit):
        script.main(["--corpus", str(tmp_path / "corpus.parquet"), "--model", "transformer",
                     "--bino-observer", "Qwen/Qwen2.5-1.5B", "--out-dir", str(tmp_path / "bad")])


# --------------------------------------------------------------------------- #
# Notebook
# --------------------------------------------------------------------------- #
def _notebook() -> dict:
    return json.loads((REPO_ROOT / "notebooks" / "retrain_modern.ipynb").read_text("utf-8"))


def test_retrain_modern_notebook_is_valid_json_with_the_new_settings() -> None:
    nb = _notebook()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and "execution_count" in cell
            compile("".join(line for line in cell["source"] if not line.lstrip().startswith(
                ("!", "%"))), "cell", "exec")
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for name in ("BACKBONE = \"auto\"", "TRANSFORMER_MAX_TRAIN = 60000", "CHUNK_BACKBONE",
                 "RUN_LOFO = False", "scripts/eval_lofo.py", "--precision", "--backbone"):
        assert name in code, name


def test_notebook_backbone_choice() -> None:
    """The settings cell's choose_backbone(): DeBERTa-v3-large on >= 22 GB bf16 GPUs."""
    nb = _notebook()
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "def choose_backbone" in "".join(c["source"]))
    start = src.index("DEBERTA = ")
    ns: dict = {}
    exec(src[start:], ns)  # noqa: S102 - our own notebook code (the end of the cell)
    choose = ns["choose_backbone"]
    big = choose("auto", gpu_gb=40, bf16=True)
    assert big["model_name"] == "microsoft/deberta-v3-large" and big["precision"] == "bf16"
    assert (big["lr"], big["batch_size"], big["grad_accum"], big["max_length"]) == (
        1e-5, 8, 4, 512)
    assert big["max_train"] == 60000
    assert choose("auto", gpu_gb=22.5, bf16=True)["model_name"] == "microsoft/deberta-v3-large"
    for gb, bf16 in ((15, False), (40, False), (16, True)):
        small = choose("auto", gpu_gb=gb, bf16=bf16)
        assert small["model_name"] == "distilbert-base-cased" and small["max_train"] is None
        assert small["note"]
    explicit = choose("distilbert-base-cased", gpu_gb=80, bf16=True)
    assert explicit["model_name"] == "distilbert-base-cased"
    explicit = choose("microsoft/deberta-v3-large", gpu_gb=15, bf16=False)
    assert explicit["model_name"] == "microsoft/deberta-v3-large" and explicit["note"]
