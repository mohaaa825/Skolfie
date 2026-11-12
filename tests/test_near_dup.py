"""
tests/test_near_dup.py
----------------------
Near-duplicate clustering (skolfie/eval/near_dup.py), the split's use of it
(near_duplicate_threshold, HC3 question groups) and scripts/check_leaks.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.eval import splits
from skolfie.eval.near_dup import cluster_ids, shingle_hashes
from skolfie.eval.splits import (
    NEAR_DUP_COLUMN,
    SplitConfig,
    add_near_duplicate_clusters,
    group_key,
    group_keys,
    make_splits,
    merge_near_duplicate_groups,
    split_groups,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RNG_WORDS = [f"w{i}" for i in range(3000)]


def _text(seed: int, n: int = 120) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(rng.choice(RNG_WORDS, size=n))


def _edit(text: str, k: int = 1) -> str:
    words = text.split()
    for i in range(k):
        words[10 + 30 * i] = "changed"
    return " ".join(words)


# ------------------------------------------------------------------ clustering

def test_near_duplicates_are_clustered_and_others_not():
    base = _text(1)
    texts = [_text(0), base, _text(2), _edit(base), base.upper(), "too short"]
    c = cluster_ids(texts)
    assert c[1] == c[3] == c[4] == 1          # labelled by the smallest position
    assert c[0] == c[2] == c[5] == -1


def test_clusters_are_deterministic_and_order_free_in_membership():
    base = _text(5)
    texts = [_text(3), base, _text(4), _edit(base)]
    a = cluster_ids(texts)
    b = cluster_ids(texts)
    assert (a == b).all()
    rev = cluster_ids(texts[::-1])
    assert (rev[::-1] >= 0).tolist() == (a >= 0).tolist()


def test_unrelated_texts_rarely_join():
    texts = [_text(s) for s in range(300)]
    assert (cluster_ids(texts) >= 0).sum() == 0


def test_shingles_ignore_case_and_need_enough_words():
    assert shingle_hashes("one two three four") .size == 0
    assert (shingle_hashes("A b c d e f") == shingle_hashes("a B C d e f")).all()


def test_bad_parameters():
    with pytest.raises(ValueError):
        cluster_ids(["a", "b"], threshold=0)
    with pytest.raises(ValueError):
        cluster_ids(["a", "b"], num_perm=64, bands=7)


# ------------------------------------------------------------------ split groups

def test_hc3_question_hash_groups_across_lines_and_subsets():
    a = "hc3:open_qa:3q0123456789ab:human:0"
    b = "hc3:reddit_eli5:900q0123456789ab:chatgpt:1"
    assert group_key(a, "hc3") == group_key(b, "hc3") == "hc3:q0123456789ab"
    assert group_key("hc3:open_qa:3:human:0", "hc3") == "hc3:open_qa:3"   # old ids
    keys = group_keys(pd.Series([a, b, "hc3:open_qa:3:human:0", "m4gt:x:1"]),
                      pd.Series(["hc3", "hc3", "hc3", "m4gt"]))
    assert keys.tolist() == ["hc3:q0123456789ab", "hc3:q0123456789ab", "hc3:open_qa:3",
                             "m4gt:x:1"]


def test_merge_near_duplicate_groups_names_by_smallest_key():
    groups = pd.Series(["g3", "g1", "g2", "g4", "g3"])
    clusters = pd.Series([0, 0, -1, 5, -1])
    merged = merge_near_duplicate_groups(groups, clusters)
    assert merged.tolist() == ["g1", "g1", "g2", "g4", "g1"]


def _frame(texts: list[str]) -> pd.DataFrame:
    n = len(texts)
    return pd.DataFrame({
        "id": [f"m4gt:x:{i}" for i in range(n)], "text": texts,
        "label": ["human" if i % 2 else "ai" for i in range(n)],
        "generator": ["human" if i % 2 else "gpt-3.5-turbo" for i in range(n)],
        "domain": ["qa"] * n, "source": ["m4gt"] * n, "n_words": [120] * n,
    })


def test_near_duplicates_land_in_one_split():
    bases = [_text(100 + k) for k in range(60)]
    texts = bases + [_edit(b) for b in bases]
    cfg = SplitConfig(near_duplicate_threshold=0.8)
    df = add_near_duplicate_clusters(_frame(texts), texts, cfg)
    assert (df[NEAR_DUP_COLUMN] >= 0).sum() == 120
    m = make_splits(df.drop(columns="text"), cfg)
    primary = m.drop_duplicates("id").set_index("id")["split"]
    for k in range(60):
        assert primary[f"m4gt:x:{k}"] == primary[f"m4gt:x:{k + 60}"]
    # without the option the pairs are split by their own ids
    plain = make_splits(_frame(texts).drop(columns="text"), SplitConfig())
    p2 = plain.drop_duplicates("id").set_index("id")["split"]
    assert any(p2[f"m4gt:x:{k}"] != p2[f"m4gt:x:{k + 60}"] for k in range(60))


def test_threshold_requires_the_cluster_column():
    cfg = SplitConfig(near_duplicate_threshold=0.8)
    with pytest.raises(ValueError, match=NEAR_DUP_COLUMN):
        split_groups(_frame(["a b c"]).drop(columns="text"), cfg)


def test_config_threshold_validation_and_hash():
    assert "near_duplicate_threshold" not in SplitConfig().to_dict()
    cfg = SplitConfig(near_duplicate_threshold=0.8)
    assert cfg.to_dict()["near_duplicate_threshold"] == 0.8
    assert cfg.config_hash() != SplitConfig().config_hash()
    for bad in (-0.1, 1.5, True, "0.8"):
        with pytest.raises(ValueError):
            SplitConfig(near_duplicate_threshold=bad)
    modern = splits.load_config(REPO_ROOT / "configs" / "splits_modern.json")
    assert modern.near_duplicate_threshold == 0.8


# ------------------------------------------------------------------ leak script

def _load_script():
    spec = importlib.util.spec_from_file_location("check_leaks",
                                                  REPO_ROOT / "scripts" / "check_leaks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_leaks_reports_cross_split_duplicates(tmp_path):
    mod = _load_script()
    base, other = _text(7), _text(8)
    df = pd.DataFrame({"id": ["a", "b", "c", "d"],
                       "text": [base, _edit(base), base, other],
                       "label": ["human", "human", "ai", "ai"], "source": ["m4gt"] * 4})
    memberships = pd.DataFrame({"id": ["a", "b", "c", "d"],
                                "split": ["train", "test", "val", "test"]})
    corpus, split_path = tmp_path / "corpus.parquet", tmp_path / "splits.parquet"
    df.to_parquet(corpus)
    memberships.to_parquet(split_path)
    assert mod.main(["--corpus", str(corpus), "--splits", str(split_path),
                     "--out-dir", str(tmp_path / "out")]) == 0
    import json
    report = json.loads((tmp_path / "out" / "leaks.json").read_text())
    assert report["splits"]["test"] == {**report["splits"]["test"], "n": 2, "exact": 0,
                                        "near": 1, "leaked": 1}
    assert report["splits"]["val"]["exact"] == 1
    assert (tmp_path / "out" / "leaks.md").read_text().startswith("# Train/test leak check")
