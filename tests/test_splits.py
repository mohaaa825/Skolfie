"""
tests/test_splits.py
---------------------
Tests for skolfie/eval/splits.py and scripts/make_splits.py on a synthetic
corpus with HC3-style grouped ids, fresh-style prompt ids, several
generators (including generator families) and several domains.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from skolfie.eval import splits
from skolfie.eval.splits import (
    SPLITS,
    SplitConfig,
    SplitLeakError,
    assign_splits,
    build_memberships,
    check_guarantees,
    group_key,
    group_keys,
    load_config,
    make_splits,
    split_summary,
    write_splits,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "splits.json"

M4GT_GENERATORS = ["human", "gpt-4", "cohere", "gpt-3.5-turbo", "bloomz", "dolly-v2-12b",
                   "text-davinci-003"]
M4GT_DOMAINS = ["academic", "other", "qa", "technical"]
AUTEXT_GENERATORS = ["human", "bloom-1b7", "bloom-3b", "bloom-7b1", "babbage", "curie",
                     "text-davinci-003"]
AUTEXT_DOMAINS = ["legal", "news", "review", "technical"]
HC3_SUBSETS = {"open_qa": "qa", "reddit_eli5": "qa", "finance": "academic"}


# --------------------------------------------------------------------------- #
# synthetic corpus
# --------------------------------------------------------------------------- #


def make_corpus(n_questions: int = 1500, n_m4gt: int = 4000, n_autext: int = 2500,
                n_fresh_topics: int = 60, seed: int = 1, start: int = 0) -> pd.DataFrame:
    """A corpus frame with the schema columns. `start` offsets the local ids so
    that two calls with different starts give disjoint rows."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(id_: str, label: str, generator: str, domain: str, source: str) -> None:
        rows.append({"id": id_, "text": f"text of {id_}", "label": label,
                     "generator": generator, "domain": domain, "subdomain": "",
                     "source": source, "split_hint": "",
                     "n_words": int(rng.integers(30, 400))})

    subsets = list(HC3_SUBSETS)
    for line in range(start, start + n_questions):
        subset = subsets[line % len(subsets)]
        domain = HC3_SUBSETS[subset]
        for i in range(int(rng.integers(1, 3))):
            add(f"hc3:{subset}:{line}:human:{i}", "human", "human", domain, "hc3")
        for i in range(int(rng.integers(1, 3))):
            add(f"hc3:{subset}:{line}:chatgpt:{i}", "ai", "chatgpt", domain, "hc3")
    for k in range(start, start + n_m4gt):
        gen = M4GT_GENERATORS[int(rng.integers(len(M4GT_GENERATORS)))]
        add(f"m4gt:SubtaskA:{k}", "human" if gen == "human" else "ai", gen,
            M4GT_DOMAINS[int(rng.integers(len(M4GT_DOMAINS)))], "m4gt")
    for k in range(start, start + n_autext):
        gen = AUTEXT_GENERATORS[int(rng.integers(len(AUTEXT_GENERATORS)))]
        add(f"autextification:train:{k}", "human" if gen == "human" else "ai", gen,
            AUTEXT_DOMAINS[int(rng.integers(len(AUTEXT_DOMAINS)))], "autextification")
    for t in range(start, start + n_fresh_topics):
        topic = f"acad-essay.top{t:03d}-aud01"
        for gen in ("llama3.3:70b", "gpt-4"):
            for style in ("plain", "formal"):
                add(f"fresh:{gen}:{topic}.{style}.w300.t07:0", "ai", gen, "academic", "fresh")
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config() -> SplitConfig:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return make_corpus()


@pytest.fixture(scope="module")
def memberships(corpus: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    return make_splits(corpus, config)


def ids_in(memberships: pd.DataFrame, split: str) -> set[str]:
    return set(memberships.loc[memberships["split"] == split, "id"])


def by_id(corpus: pd.DataFrame) -> pd.DataFrame:
    return corpus.set_index("id")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_default_config(config: SplitConfig) -> None:
    assert config.held_out_generators == ["gpt-4", "cohere"]
    assert config.held_out_domains == ["news"]
    assert config.fractions == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert config.short_text_max_words == 100
    assert config.seed == 0
    assert config.family("chatgpt") == config.family("gpt-3.5-turbo") == "gpt-3.5"
    assert {config.family(g) for g in ("bloom-1b7", "bloom-3b", "bloom-7b1")} == {"bloom"}
    assert {config.family(g) for g in ("babbage", "curie", "text-davinci-003")} == {"gpt-3-base"}
    assert config.family("never-seen-model") == "never-seen-model"
    assert config.held_out_families == {"gpt-4", "cohere"}


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(fractions={"train": 0.8, "val": 0.1, "test": 0.2})
    with pytest.raises(ValueError, match="exactly the keys"):
        SplitConfig(fractions={"train": 0.9, "test": 0.1})
    with pytest.raises(ValueError, match="unknown split config key"):
        SplitConfig.from_dict({"seed": 0, "held_out_generator": ["gpt-4"]})
    with pytest.raises(ValueError, match="list of strings"):
        SplitConfig(held_out_domains="news")


def test_config_hash_ignores_order_but_not_content(config: SplitConfig) -> None:
    d = config.to_dict()
    swapped = SplitConfig.from_dict({**d, "held_out_generators": ["cohere", "gpt-4"]})
    assert swapped.config_hash() == config.config_hash()
    other = SplitConfig.from_dict({**d, "seed": 1})
    assert other.config_hash() != config.config_hash()


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("id_", "source", "expected"),
    [
        ("hc3:open_qa:12:human:0", "hc3", "hc3:open_qa:12"),
        ("hc3:open_qa:12:chatgpt:3", "hc3", "hc3:open_qa:12"),
        ("hc3:all:7:human:1", "hc3", "hc3:all:7"),
        ("m4gt:SubtaskA:991", "m4gt", "m4gt:SubtaskA:991"),
        ("autextification:train:5", "autextification", "autextification:train:5"),
        ("fresh:gpt-4:acad-essay.top01-aud02.plain.w300.t07:0", "fresh",
         "fresh:acad-essay.top01-aud02"),
        ("fresh:llama3.3:70b:acad-essay.top01-aud02.formal.w500.t10:2", "fresh",
         "fresh:acad-essay.top01-aud02"),
        ("fresh:oddly-formed", "fresh", "fresh:oddly-formed"),
        ("hc3:weird", "hc3", "hc3:weird"),
    ],
)
def test_group_key(id_: str, source: str, expected: str) -> None:
    assert group_key(id_, source) == expected


def test_group_keys_matches_scalar(corpus: pd.DataFrame) -> None:
    vec = group_keys(corpus["id"], corpus["source"])
    scalar = [group_key(i, s) for i, s in zip(corpus["id"], corpus["source"])]
    assert vec.tolist() == scalar
    # a question's human and chatgpt answers share a key
    hc3 = corpus[corpus["source"] == "hc3"]
    keys = group_keys(hc3["id"], hc3["source"])
    assert keys.nunique() == 1500


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #


def test_every_row_has_one_primary_split(corpus: pd.DataFrame, config: SplitConfig) -> None:
    primary = assign_splits(corpus, config)
    assert primary.index.equals(corpus.index)
    assert set(primary) <= set(splits.PRIMARY_SPLITS)
    assert primary.notna().all()


def test_deterministic(corpus: pd.DataFrame, config: SplitConfig,
                       memberships: pd.DataFrame) -> None:
    again = make_splits(corpus.copy(), config)
    pd.testing.assert_frame_equal(memberships, again)
    # row order does not matter
    shuffled = corpus.sample(frac=1.0, random_state=3).reset_index(drop=True)
    pd.testing.assert_frame_equal(memberships, make_splits(shuffled, config))
    # the seed does matter
    reseeded = SplitConfig.from_dict({**config.to_dict(), "seed": 1})
    other = make_splits(corpus, reseeded)
    assert ids_in(other, "train") != ids_in(memberships, "train")


def test_stable_when_rows_are_added(corpus: pd.DataFrame, config: SplitConfig,
                                    memberships: pd.DataFrame) -> None:
    extra = make_corpus(n_questions=400, n_m4gt=800, n_autext=500, n_fresh_topics=10, seed=9,
                        start=100_000)
    # also add a new answer to an existing HC3 question (same group, existing domain)
    extra_answer = corpus[corpus["id"] == "hc3:open_qa:0:human:0"].assign(
        id="hc3:open_qa:0:human:9")
    bigger = pd.concat([extra, corpus, extra_answer], ignore_index=True)
    new = make_splits(bigger, config)
    old_ids = set(corpus["id"])
    old_part = new[new["id"].isin(old_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(memberships, old_part)
    # the new answer follows its question
    primary = dict(zip(new.drop_duplicates("id")["id"], new.drop_duplicates("id")["split"]))
    assert primary["hc3:open_qa:0:human:9"] == primary["hc3:open_qa:0:human:0"]


def test_fractions_roughly_honoured(corpus: pd.DataFrame, config: SplitConfig,
                                    memberships: pd.DataFrame) -> None:
    core = memberships[memberships["split"].isin(["train", "val", "test"])]
    frac = core["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.8, abs=0.03)
    assert frac["val"] == pytest.approx(0.1, abs=0.02)
    assert frac["test"] == pytest.approx(0.1, abs=0.02)


# --------------------------------------------------------------------------- #
# guarantees
# --------------------------------------------------------------------------- #


def test_no_id_in_train_and_elsewhere(memberships: pd.DataFrame) -> None:
    train = ids_in(memberships, "train")
    for split in SPLITS:
        if split != "train":
            assert not train & ids_in(memberships, split), split


def test_no_group_straddles(corpus: pd.DataFrame, memberships: pd.DataFrame) -> None:
    m = memberships.merge(corpus[["id", "source"]], on="id")
    m["group"] = group_keys(m["id"], m["source"])
    m["side"] = m["split"].map(lambda s: s if s in ("train", "val") else "test")
    sides = m.groupby("group")["side"].nunique()
    assert (sides == 1).all()
    # HC3 groups with several answers exist and stay whole
    hc3 = m[m["id"].str.startswith("hc3:")]
    assert hc3.groupby("group")["id"].nunique().max() > 1


def test_held_out_family_not_in_train_or_val(corpus: pd.DataFrame, config: SplitConfig,
                                             memberships: pd.DataFrame) -> None:
    info = by_id(corpus)
    fit = memberships[memberships["split"].isin(["train", "val"])]
    gens = info.loc[fit["id"], "generator"]
    assert not set(gens) & {"gpt-4", "cohere"}
    ood = info.loc[sorted(ids_in(memberships, "test_ood_generator"))]
    ai = ood[ood["label"] == "ai"]
    assert set(ai["generator"]) == {"gpt-4", "cohere"}
    # every gpt-4/cohere AI row outside the held-out domain is in test_ood_generator
    held = corpus[corpus["generator"].isin(["gpt-4", "cohere"]) & (corpus["domain"] != "news")]
    assert set(held["id"]) == set(ai.index)


def test_held_out_domain_only_in_ood_domain(corpus: pd.DataFrame,
                                            memberships: pd.DataFrame) -> None:
    news = set(corpus.loc[corpus["domain"] == "news", "id"])
    assert news and ids_in(memberships, "test_ood_domain") == news
    news_rows = memberships[memberships["id"].isin(news)]
    assert set(news_rows["split"]) == {"test_ood_domain"}
    labels = set(corpus.loc[corpus["domain"] == "news", "label"])
    assert labels == {"human", "ai"}


def test_domain_rule_wins_over_generator_rule(config: SplitConfig) -> None:
    df = make_corpus(n_questions=50, n_m4gt=50, n_autext=50, n_fresh_topics=1)
    df.loc[0, ["label", "generator", "domain"]] = ["ai", "gpt-4", "news"]
    primary = assign_splits(df, config)
    assert primary.iloc[0] == "test_ood_domain"


def test_family_hold_out(corpus: pd.DataFrame, config: SplitConfig) -> None:
    cfg = SplitConfig.from_dict({**config.to_dict(), "held_out_generators": ["gpt-3.5-turbo"]})
    m = make_splits(corpus, cfg)
    info = by_id(corpus)
    ood_ai = info.loc[sorted(ids_in(m, "test_ood_generator"))]
    ood_ai = ood_ai[ood_ai["label"] == "ai"]
    assert set(ood_ai["generator"]) == {"gpt-3.5-turbo", "chatgpt"}
    fit = info.loc[sorted(ids_in(m, "train") | ids_in(m, "val"))]
    assert not set(fit["generator"]) & {"gpt-3.5-turbo", "chatgpt"}
    # every HC3 question has a held-out (chatgpt) answer, so its human answers
    # are kept out of train/val too: no group straddles.
    hc3_human = corpus[(corpus["source"] == "hc3") & (corpus["label"] == "human")]
    assert not set(hc3_human["id"]) & (ids_in(m, "train") | ids_in(m, "val"))
    assert set(hc3_human["id"]) <= ids_in(m, "test")
    # bloom family: one name holds out all three sizes
    cfg = SplitConfig.from_dict({**config.to_dict(), "held_out_generators": ["bloom-3b"],
                                 "held_out_domains": []})
    m = make_splits(corpus, cfg)
    ood = info.loc[sorted(ids_in(m, "test_ood_generator"))]
    assert {"bloom-1b7", "bloom-3b", "bloom-7b1", "bloomz"} <= set(ood["generator"])


def test_ood_generator_has_human_texts(corpus: pd.DataFrame, config: SplitConfig,
                                       memberships: pd.DataFrame) -> None:
    info = by_id(corpus)
    ood = info.loc[sorted(ids_in(memberships, "test_ood_generator"))]
    human = ood[ood["label"] == "human"]
    assert len(human) > 0
    covered = set(ood.loc[ood["label"] == "ai", "domain"])
    assert set(human["domain"]) <= covered
    # they are exactly the in-distribution test human rows of covered domains
    test = info.loc[sorted(ids_in(memberships, "test"))]
    expected = test[(test["label"] == "human") & test["domain"].isin(covered)]
    assert set(human.index) == set(expected.index)
    assert set(human.index) <= ids_in(memberships, "test")
    # and the covered domains are the held-out generators' domains
    assert covered == set(corpus.loc[corpus["generator"].isin(["gpt-4", "cohere"])
                                     & (corpus["domain"] != "news"), "domain"])


def test_test_short(corpus: pd.DataFrame, config: SplitConfig,
                    memberships: pd.DataFrame) -> None:
    info = by_id(corpus)
    test = info.loc[sorted(ids_in(memberships, "test"))]
    expected = set(test.index[test["n_words"] < config.short_text_max_words])
    assert expected and ids_in(memberships, "test_short") == expected
    assert (info.loc[sorted(expected), "n_words"] < 100).all()


def test_check_guarantees_catches_leaks(corpus: pd.DataFrame, config: SplitConfig,
                                        memberships: pd.DataFrame) -> None:
    check_guarantees(corpus, memberships, config)  # the real output passes

    def moved(id_: str, to: str) -> pd.DataFrame:
        m = memberships.copy()
        m.loc[m["id"] == id_, "split"] = to
        return m.drop_duplicates()

    held_ai = sorted(i for i in ids_in(memberships, "test_ood_generator")
                     - ids_in(memberships, "test") if i.startswith("m4gt:"))[0]
    with pytest.raises(SplitLeakError, match="held-out-family"):
        check_guarantees(corpus, moved(held_ai, "train"), config)
    news = sorted(ids_in(memberships, "test_ood_domain"))[0]
    with pytest.raises(SplitLeakError, match="held-out-domain"):
        check_guarantees(corpus, moved(news, "val"), config)
    # a train id also listed in test
    train_id = sorted(ids_in(memberships, "train"))[0]
    extra = pd.concat([memberships, pd.DataFrame({"id": [train_id], "split": ["test_short"]})])
    with pytest.raises(SplitLeakError, match="train and another split"):
        check_guarantees(corpus, extra, config)
    # one answer of a multi-answer HC3 question moved from train to test
    hc3 = corpus[corpus["source"] == "hc3"].copy()
    hc3["g"] = group_keys(hc3["id"], hc3["source"])
    train = ids_in(memberships, "train")
    multi = hc3[hc3["id"].isin(train)].groupby("g")["id"].apply(list)
    victim = next(ids[0] for ids in multi if len(ids) > 1)
    with pytest.raises(SplitLeakError, match="straddle"):
        check_guarantees(corpus, moved(victim, "test"), config)
    # a row with no split at all
    with pytest.raises(SplitLeakError, match="primary"):
        check_guarantees(corpus, memberships[memberships["id"] != train_id], config)
    with pytest.raises(SplitLeakError, match="unknown split"):
        check_guarantees(corpus, moved(train_id, "holdout"), config)


def test_rejects_duplicate_ids(corpus: pd.DataFrame, config: SplitConfig) -> None:
    with pytest.raises(ValueError, match="duplicate id"):
        assign_splits(pd.concat([corpus, corpus.head(1)]), config)


def test_non_default_index(corpus: pd.DataFrame, config: SplitConfig,
                           memberships: pd.DataFrame) -> None:
    df = corpus.set_axis(np.arange(len(corpus)) * 7 + 3)
    primary = assign_splits(df, config)
    assert primary.index.equals(df.index)
    pd.testing.assert_frame_equal(build_memberships(df, primary, config), memberships)


def test_fast_enough_on_corpus_scale(config: SplitConfig) -> None:
    import time

    big = make_corpus(n_questions=20_000, n_m4gt=100_000, n_autext=40_000, n_fresh_topics=0)
    assert len(big) > 190_000
    started = time.perf_counter()
    make_splits(big, config)
    assert time.perf_counter() - started < 20  # a few seconds in practice


# --------------------------------------------------------------------------- #
# summary, files, CLI
# --------------------------------------------------------------------------- #


def test_split_summary(corpus: pd.DataFrame, memberships: pd.DataFrame) -> None:
    summary = split_summary(corpus, memberships)
    assert list(summary.columns) == ["split", "label", "generator", "domain", "count"]
    assert summary["count"].sum() == len(memberships)
    totals = summary.groupby("split")["count"].sum()
    for split in SPLITS:
        assert totals[split] == len(ids_in(memberships, split))
    assert list(dict.fromkeys(summary["split"])) == list(SPLITS)


def write_corpus(tmp_path: Path, df: pd.DataFrame, version: str | None = "v-test123") -> Path:
    corpus_dir = tmp_path / "processed"
    corpus_dir.mkdir()
    path = corpus_dir / "corpus.parquet"
    df.to_parquet(path, index=False)
    if version:
        (corpus_dir / "manifest.json").write_text(json.dumps({"corpus_version": version}))
    return path


def test_write_splits(tmp_path: Path, corpus: pd.DataFrame, config: SplitConfig,
                      memberships: pd.DataFrame) -> None:
    corpus_path = write_corpus(tmp_path, corpus)
    result = write_splits(corpus, memberships, config, tmp_path / "out", corpus_path=corpus_path)

    table = pq.read_table(result.parquet_path)
    assert table.column_names == ["id", "split"]
    assert all(str(t) == "string" for t in table.schema.types)
    back = splits.load_splits(tmp_path / "out")
    pd.testing.assert_frame_equal(back.astype(object), memberships.astype(object),
                                  check_dtype=False)

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["corpus_version"] == "v-test123"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["config"] == config.to_dict()
    assert manifest["splits_version"].startswith("s-") and len(manifest["splits_version"]) == 18
    assert manifest["split_sizes"]["test_ood_domain"]["total"] == len(
        ids_in(memberships, "test_ood_domain"))
    assert sum(c["count"] for c in manifest["counts"]) == len(memberships)
    assert manifest["ood_generator_human_domains"] == sorted(M4GT_DOMAINS)
    assert manifest["held_out_generators_in_corpus"] == ["cohere", "gpt-4"]

    md = result.summary_path.read_text()
    assert manifest["splits_version"] in md
    # the synthetic corpus is small: some splits have < 500 texts of a label
    assert "WARNING: test_ood_domain: only" in md
    assert any("test_short" in w for w in manifest["warnings"])

    # same input -> same version; another config -> another version
    again = write_splits(corpus, memberships, config, tmp_path / "out2", corpus_path=corpus_path)
    assert again.splits_version == result.splits_version
    cfg2 = SplitConfig.from_dict({**config.to_dict(), "seed": 5})
    other = write_splits(corpus, make_splits(corpus, cfg2), cfg2, tmp_path / "out3")
    assert other.splits_version != result.splits_version
    assert other.manifest["corpus_version"] is None


def test_warns_about_missing_held_out_names(tmp_path: Path, corpus: pd.DataFrame,
                                            config: SplitConfig) -> None:
    cfg = SplitConfig.from_dict({**config.to_dict(), "held_out_generators": ["gpt-9"],
                                 "held_out_domains": ["creative"]})
    result = write_splits(corpus, make_splits(corpus, cfg), cfg, tmp_path)
    assert any("'gpt-9'" in w for w in result.manifest["warnings"])
    assert any("'creative'" in w for w in result.manifest["warnings"])


def load_script():
    spec = importlib.util.spec_from_file_location("skolfie_make_splits_script",
                                                  REPO_ROOT / "scripts" / "make_splits.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli(tmp_path: Path, corpus: pd.DataFrame, memberships: pd.DataFrame,
             capsys: pytest.CaptureFixture[str]) -> None:
    corpus_path = write_corpus(tmp_path, corpus)
    out_dir = tmp_path / "splits"
    code = load_script().main(["--corpus", str(corpus_path), "--config", str(DEFAULT_CONFIG),
                               "--out-dir", str(out_dir)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "## Split sizes" in printed and "test_ood_generator" in printed
    assert "splits.parquet" in printed
    for name in ("splits.parquet", "splits_manifest.json", "splits.md"):
        assert (out_dir / name).is_file()
    written = splits.load_splits(out_dir)
    assert written["id"].tolist() == memberships["id"].tolist()
    assert written["split"].tolist() == memberships["split"].tolist()
    assert json.loads((out_dir / "splits_manifest.json").read_text())["corpus_version"] == \
        "v-test123"


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = load_script()
    assert script.main(["--corpus", str(tmp_path / "missing.parquet"),
                        "--config", str(DEFAULT_CONFIG), "--out-dir", str(tmp_path)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"fractions": {"train": 1.0}}))
    assert script.main(["--corpus", str(tmp_path / "missing.parquet"),
                        "--config", str(bad), "--out-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "cannot read corpus" in err and "cannot read config" in err
    assert not (tmp_path / "splits.parquet").exists()
