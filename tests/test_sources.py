"""
tests/test_sources.py
---------------------
Source loaders against small fixture files in each dataset's on-disk format
(tests/fixtures/sources/<name>/...), always with download=False.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from skolfie.data.schema import COLUMNS, DOMAINS, Record
from skolfie.data.sources import SOURCES, autextification, fresh, get_source, hc3, m4gt

FIXTURES = Path(__file__).parent / "fixtures" / "sources"


def _load(module, raw_dir: Path = FIXTURES) -> list[Record]:
    return list(module.load(raw_dir, download=False))


def _by_id(records: list[Record]) -> dict[str, Record]:
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids)), "ids must be unique"
    return {r.id: r for r in records}


# --------------------------------------------------------------------- registry

def test_registry_and_interface():
    assert set(SOURCES) == {"hc3", "m4gt", "autextification", "persuade", "fresh"}
    assert get_source("HC3") is hc3
    with pytest.raises(KeyError, match="unknown source"):
        get_source("nope")
    for name, module in SOURCES.items():
        assert module.NAME == name
        for attr in ("DESCRIPTION", "CITATION", "LICENSE_NOTE", "MANUAL_DOWNLOAD"):
            assert isinstance(getattr(module, attr), str)
        assert callable(module.load)


@pytest.mark.parametrize("name", ["hc3", "m4gt", "autextification", "persuade", "fresh"])
def test_common_contract(name):
    records = _load(SOURCES[name])
    assert records
    _by_id(records)
    for r in records:
        assert isinstance(r, Record)
        assert r.id.startswith(f"{name}:")
        assert r.source == name
        assert r.domain in DOMAINS
        assert r.text and r.text == r.text.strip()
        assert set(r.to_dict()) == set(COLUMNS)
        assert (r.generator == "human") == (r.label == "human")


@pytest.mark.parametrize("name", ["hc3", "m4gt", "autextification", "persuade", "fresh"])
def test_ids_are_stable(name):
    first = [r.to_dict() for r in _load(SOURCES[name])]
    second = [r.to_dict() for r in _load(SOURCES[name])]
    assert first == second


@pytest.mark.parametrize("name", ["hc3", "m4gt", "autextification", "persuade", "fresh"])
def test_missing_files_raise_clear_error(name, tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        _load(SOURCES[name], tmp_path)
    message = str(info.value)
    assert str(tmp_path / name) in message
    assert SOURCES[name].MANUAL_DOWNLOAD.strip().splitlines()[0] in message


# -------------------------------------------------------------------------- hc3

def test_hc3_records():
    records = _by_id(_load(hc3))
    # eli5: 2 human (one empty answer dropped) + 3 chatgpt; wiki_csai 1+1; finance 1+1
    assert len(records) == 9
    counts = Counter((r.subdomain, r.label) for r in records.values())
    assert counts == {
        ("reddit_eli5", "human"): 2, ("reddit_eli5", "ai"): 3,
        ("wiki_csai", "human"): 1, ("wiki_csai", "ai"): 1,
        ("finance", "human"): 1, ("finance", "ai"): 1,
    }
    for r in records.values():
        assert r.domain == ("academic" if r.subdomain == "wiki_csai" else "qa")
        assert r.generator in {"human", "chatgpt"}
    # Local id is always the line number, even when the row carries its own id.
    assert "hc3:reddit_eli5:0:human:0" in records
    assert "hc3:reddit_eli5:1:chatgpt:1" in records
    assert "hc3:finance:0:human:0" in records


def test_hc3_repeated_row_ids_still_give_unique_record_ids(tmp_path):
    src = tmp_path / "hc3"
    src.mkdir()
    rows = [{"id": "0", "question": f"q{i}", "human_answers": [f"human {i}"],
             "chatgpt_answers": [f"ai {i}"]} for i in range(3)]
    (src / "finance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ids = [r.id for r in _load(hc3, tmp_path)]
    assert len(ids) == 6 and len(set(ids)) == 6


def test_hc3_cleans_tokenisation_artefacts_in_human_answers():
    records = _by_id(_load(hc3))
    text = records["hc3:reddit_eli5:0:human:0"].text
    assert text.startswith("It's a leftover reflex from when we had fur. Raising")
    assert "We don't have much fur now, so it doesn't do much (but the reflex" in text
    assert " ." not in text and " ," not in text


def test_hc3_clean_human_answer_leaves_normal_text_alone():
    normal = 'He said "yes" - and left. Cost: 5 % of $10 (approx.)'
    assert hc3.clean_human_answer(normal) == 'He said "yes" - and left. Cost: 5% of $10 (approx.)'
    assert hc3.clean_human_answer("rock 'n' roll isn't dead") == "rock 'n' roll isn't dead"


def test_hc3_uses_all_jsonl_only_without_subset_files(tmp_path):
    src = tmp_path / "hc3"
    src.mkdir()
    row = {"question": "q", "human_answers": ["a b c"], "chatgpt_answers": ["d e f"],
           "source": "open_qa"}
    (src / "all.jsonl").write_text(json.dumps(row) + "\n")
    records = _load(hc3, tmp_path)
    assert [r.id for r in records] == ["hc3:all:0:human:0", "hc3:all:0:chatgpt:0"]
    assert {r.subdomain for r in records} == {"open_qa"}
    shutil.copy(FIXTURES / "hc3" / "finance.jsonl", src / "finance.jsonl")
    assert {r.subdomain for r in _load(hc3, tmp_path)} == {"finance"}


def test_hc3_error_names_keys_seen(tmp_path):
    (tmp_path / "hc3").mkdir()
    (tmp_path / "hc3" / "open_qa.jsonl").write_text(json.dumps({"q": "x", "answers": []}) + "\n")
    with pytest.raises(ValueError, match=r"human_answers.*found keys \['answers', 'q'\]"):
        _load(hc3, tmp_path)


# ------------------------------------------------------------------------- m4gt

def test_m4gt_records():
    records = _by_id(_load(m4gt))
    assert len(records) == 11
    gens = Counter(r.generator for r in records.values())
    assert gens == {
        "human": 3, "gpt-3.5-turbo": 2, "text-davinci-003": 1, "gpt-4": 1, "bloomz": 2,
        "dolly-v2-12b": 1, "cohere": 1,
    }
    domains = {r.subdomain: r.domain for r in records.values()}
    assert domains == {
        "wikihow": "technical", "arxiv": "academic", "reddit": "qa", "peerread": "academic",
        "wikipedia": "other", "outfox": "academic",
    }
    first = records["m4gt:SubtaskA:0"]
    assert (first.label, first.generator, first.subdomain) == ("human", "human", "wikihow")
    assert records["m4gt:SubtaskA:1"].generator == "gpt-3.5-turbo"
    dev = records["m4gt:subtaskA_dev_monolingual:101"]
    assert (dev.label, dev.generator, dev.split_hint) == ("ai", "bloomz", "dev")
    assert records["m4gt:SubtaskA:0"].split_hint == ""


def test_m4gt_skips_non_english(tmp_path, caplog):
    src = tmp_path / "m4gt"
    src.mkdir()
    rows = [
        {"text": "An English abstract.", "label": 1, "model": "chatGPT", "source": "arxiv"},
        {"text": "یہ ایک خبر ہے", "label": 1, "model": "chatGPT", "source": "urdu"},
        {"text": "Ein Artikel.", "label": 0, "model": "human", "source": "wikipedia",
         "language": "german"},
    ]
    (src / "SubtaskA_multilingual.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        records = _load(m4gt, tmp_path)
    assert [r.text for r in records] == ["An English abstract."]
    assert "skipped 2" in caplog.text


def test_m4gt_label_fallback_and_key_error(tmp_path):
    src = tmp_path / "m4gt"
    src.mkdir()
    (src / "a.jsonl").write_text(
        json.dumps({"text": "x y", "label": 0, "source": "arxiv"}) + "\n"
        + json.dumps({"text": "y z", "label": 1, "source": "arxiv"}) + "\n"
    )
    assert [(r.label, r.generator) for r in _load(m4gt, tmp_path)] == [
        ("human", "human"), ("ai", "unknown"),
    ]
    (src / "b.jsonl").write_text(json.dumps({"body": "x", "source": "arxiv"}) + "\n")
    with pytest.raises(ValueError, match=r"no text field.*found keys \['body', 'source'\]"):
        _load(m4gt, tmp_path)


def test_m4gt_generator_normalisation():
    assert m4gt.normalise_generator("chatGPT") == "gpt-3.5-turbo"
    assert m4gt.normalise_generator("davinci-003") == "text-davinci-003"
    assert m4gt.normalise_generator("bigscience/bloomz") == "bloomz"
    assert m4gt.normalise_generator("flant5") == "flan-t5"
    assert m4gt.normalise_generator("Some-New-Model") == "some-new-model"


# -------------------------------------------------------------- autextification

def test_autextification_records():
    records = _by_id(_load(autextification))
    assert len(records) == 6  # the Spanish file and the subtask 2 file are not records
    assert {r.split_hint for r in records.values()} == {"train", "test"}
    legal = records["autextification:train:1"]
    assert (legal.label, legal.generator, legal.domain, legal.subdomain) == (
        "human", "human", "legal", "legal"
    )
    tweet = records["autextification:train:2"]
    assert (tweet.label, tweet.domain, tweet.subdomain) == ("ai", "casual", "tweets")
    # Generators recovered from subtask 2 letters (F, B) by matching the text.
    assert tweet.generator == "text-davinci-003"
    wiki = records["autextification:train:3"]
    assert (wiki.generator, wiki.domain) == ("bloom-3b", "technical")
    assert wiki.text.endswith("brush.\nDry it thoroughly.")
    quoted = records["autextification:train:4"].text
    assert quoted == "Don't use soap on it, \"seasoning\" is the whole point."
    review = records["autextification:test:2"]
    assert (review.label, review.generator, review.domain) == ("ai", "unknown", "review")
    assert records["autextification:test:1"].domain == "news"


def test_autextification_hf_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    features = {
        "id": {"dtype": "int64", "_type": "Value"},
        "prompt": {"dtype": "string", "_type": "Value"},
        "text": {"dtype": "string", "_type": "Value"},
        "label": {"names": ["generated", "human"], "_type": "ClassLabel"},
        "model": {"dtype": "string", "_type": "Value"},
        "domain": {"dtype": "string", "_type": "Value"},
    }
    table = pa.table({
        "id": [10, 11],
        "prompt": ["", "Write a tweet:"],
        "text": ["A human tweet lol", "An AI tweet about the weather."],
        "label": [1, 0],
        "model": ["human", "D"],
        "domain": ["tweets", "tweets"],
    })
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode()}
    out = tmp_path / "autextification" / "detection_en" / "train"
    out.mkdir(parents=True)
    pq.write_table(table.replace_schema_metadata(meta), out / "0000.parquet")

    records = _by_id(_load(autextification, tmp_path))
    assert records["autextification:train:10"].label == "human"
    ai = records["autextification:train:11"]
    assert (ai.label, ai.generator, ai.domain) == ("ai", "babbage", "casual")
    assert ai.split_hint == "train"


def test_autextification_unrecognised_files_are_listed(tmp_path):
    (tmp_path / "autextification").mkdir()
    (tmp_path / "autextification" / "data.tsv").write_text("id\ttext\tlabel\n1\tx\thuman\n")
    with pytest.raises(FileNotFoundError, match="seen but not recognised: data.tsv"):
        _load(autextification, tmp_path)


# ------------------------------------------------------------------------ fresh

def test_fresh_records():
    records = _by_id(_load(fresh))
    assert set(records) == {"fresh:0001", "fresh:0002"}
    ai = records["fresh:0002"]
    assert (ai.label, ai.generator, ai.domain) == ("ai", "gpt-4o-mini", "casual")
    assert ai.split_hint == "test"
    assert ai.n_words == len(ai.text.split())


def test_fresh_validates_through_record(tmp_path):
    (tmp_path / "fresh").mkdir()
    bad = {"id": "x", "text": "t", "label": "human", "generator": "gpt-4o", "domain": "news",
           "subdomain": "n"}
    (tmp_path / "fresh" / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:1: invalid record: human records"):
        _load(fresh, tmp_path)
    (tmp_path / "fresh" / "bad.jsonl").write_text(json.dumps({"id": "x", "text": "t"}) + "\n")
    with pytest.raises(ValueError, match="missing columns"):
        _load(fresh, tmp_path)


def test_fresh_skips_truncated_lines_and_keeps_colon_ids(tmp_path, caplog):
    (tmp_path / "fresh").mkdir()
    good = {"id": "fresh:llama3.3:70b:essay-03:1", "text": "Para one.\n\nPara two.",
            "label": "ai", "generator": "llama3.3:70b", "domain": "academic",
            "subdomain": "essay", "source": "fresh", "split_hint": "", "n_words": 4}
    path = tmp_path / "fresh" / "llama3.3_70b.jsonl"
    path.write_text(json.dumps(good) + "\n" + '{"id": "fresh:llama3.3:70b:essay-03:2", "te')
    with caplog.at_level("WARNING"):
        records = _load(fresh, tmp_path)
    assert [r.id for r in records] == ["fresh:llama3.3:70b:essay-03:1"]
    assert records[0].generator == "llama3.3:70b"
    assert records[0].id.rsplit(":", 2)[1:] == ["essay-03", "1"]
    assert "llama3.3_70b.jsonl:2: skipping" in caplog.text
