"""
tests/test_review_data.py
-------------------------
Regression tests from the data-pipeline review: provider robustness and
secret handling, loader edge cases (null ids, BOMs), all-or-nothing corpus
and split writes, and an end-to-end leakage check of corpus -> splits.
No network access.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from skolfie.data import corpus
from skolfie.data.corpus import build_corpus, normalized_hash
from skolfie.data.generation import providers
from skolfie.data.generation.providers import (
    HTTPResponse,
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderConfigError,
    ProviderError,
)
from skolfie.data.schema import Record
from skolfie.data.sources import autextification, fresh, m4gt
from skolfie.data.sources._common import iter_jsonl
from skolfie.eval import splits

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-live-0123456789abcdef"
OPENAI_OK = {"choices": [{"message": {"content": "Hi there."}, "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)


def _replay(*responses):
    calls = []

    def post(url, headers, payload, timeout):
        calls.append(headers)
        item = responses[min(len(calls), len(responses)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item(headers) if callable(item) else item

    post.calls = calls
    return post


def _openai(**kw) -> OpenAIProvider:
    return OpenAIProvider("gpt-4o", api_key=KEY, backoff_base=0.0, **kw)


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #


def test_body_cut_off_mid_transfer_is_retried(monkeypatch) -> None:
    requests = pytest.importorskip("requests")
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
        resp = SimpleNamespace(status_code=200, headers={}, text=json.dumps(OPENAI_OK))
        return resp

    monkeypatch.setattr(requests, "post", post)
    p = _openai(max_retries=2)
    assert p.generate("x", temperature=0.5, max_tokens=5) == "Hi there."
    assert p.stats.retries == 1


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", "[1, 2]", '"text"'])
def test_2xx_without_a_json_object_is_a_provider_error(monkeypatch, body) -> None:
    post = _replay(HTTPResponse(200, {}, body))
    monkeypatch.setattr(providers, "_http_post", post)
    p = _openai(max_retries=3)
    with pytest.raises(ProviderError, match="without a JSON object") as info:
        p.generate("x", temperature=0.5, max_tokens=5)
    assert info.value.status == 200
    assert len(post.calls) == 1 and p.stats.failures == 1


def test_api_key_echoed_by_server_is_masked(monkeypatch) -> None:
    echo = lambda headers: HTTPResponse(400, {}, f"bad auth header: {dict(headers)}")  # noqa: E731
    monkeypatch.setattr(providers, "_http_post", _replay(echo))
    p = _openai()
    with pytest.raises(ProviderError) as info:
        p.generate("x", temperature=0.5, max_tokens=5)
    assert KEY not in str(info.value) and KEY not in info.value.body
    assert "***" in info.value.body
    # nor in the stats, the repr or a transient error
    assert KEY not in repr(p) and KEY not in repr(p.stats)
    monkeypatch.setattr(providers, "_http_post",
                        _replay(providers.TransientHTTPError(f"reset while sending {KEY}")))
    with pytest.raises(ProviderError) as info:
        _openai(max_retries=0).generate("x", temperature=0.5, max_tokens=5)
    assert KEY not in str(info.value)


def test_short_test_keys_do_not_mangle_messages() -> None:
    assert providers.redact("max_tokens", "k") == "max_tokens"


@pytest.mark.parametrize("kwargs", [{"max_retries": -1}, {"timeout": 0}, {"timeout": -5.0}])
def test_bad_retry_or_timeout_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ProviderConfigError):
        _openai(**kwargs)


@pytest.mark.parametrize("url", ["localhost:11434/v1", "ftp://example.com", "https://", ""])
def test_base_url_needs_http_scheme_and_host(url) -> None:
    with pytest.raises(ProviderConfigError):
        OpenAIProvider("gpt-4o", api_key=KEY, base_url=url or "x")


def test_plain_http_to_remote_host_with_key_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=providers.__name__):
        OpenAICompatibleProvider("m", base_url="http://localhost:11434/v1", api_key=KEY)
        OpenAICompatibleProvider("m", base_url="http://example.com/v1")  # no key: nothing sent
        assert not caplog.records
        OpenAICompatibleProvider("m", base_url="http://example.com/v1", api_key=KEY)
    assert len(caplog.records) == 1 and "plain http" in caplog.text
    assert KEY not in caplog.text


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location(
        "skolfie_generate_script_review", REPO_ROOT / "scripts" / "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("flag", [["--timeout", "0"], ["--max-retries", "-1"]])
def test_generate_script_rejects_bad_flags(gen, flag) -> None:
    with pytest.raises(SystemExit) as info:
        gen.main(["--provider", "fake", "--n", "1", "--dry-run", *flag])
    assert info.value.code == 2


@pytest.mark.parametrize("generator", ["../../etc/passwd", "..", "a/b\\c", "/abs", "x:y\x00z"])
def test_generate_output_path_stays_in_out_dir(gen, tmp_path, generator) -> None:
    path = gen.output_path(tmp_path, generator)
    assert path.parent == tmp_path and path.name.endswith(".jsonl")
    assert not path.name.startswith(".")


# --------------------------------------------------------------------------- #
# source loaders
# --------------------------------------------------------------------------- #


def test_m4gt_null_ids_fall_back_to_the_row_index(tmp_path) -> None:
    d = tmp_path / "m4gt"
    d.mkdir()
    rows = [{"text": f"text {i}", "model": "human", "source": "wikihow", "id": v}
            for i, v in enumerate([None, "", None, "real-7"])]
    (d / "SubtaskA.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ids = [r.id for r in m4gt.load(tmp_path, download=False)]
    assert ids == ["m4gt:SubtaskA:0", "m4gt:SubtaskA:1", "m4gt:SubtaskA:2",
                   "m4gt:SubtaskA:real-7"]


def test_jsonl_with_byte_order_mark(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_bytes('﻿{"a": 1}\n{"a": 2}\n'.encode())
    assert [row for _, row in iter_jsonl(path)] == [{"a": 1}, {"a": 2}]

    d = tmp_path / "fresh"
    d.mkdir()
    row = {"id": "g:p:0", "text": "some text", "label": "ai", "generator": "g",
           "domain": "casual", "subdomain": "post"}
    (d / "g.jsonl").write_bytes(("﻿" + json.dumps(row) + "\n").encode())
    assert [r.id for r in fresh.load(tmp_path)] == ["fresh:g:p:0"]


def test_autextification_tsv_with_byte_order_mark_keeps_source_ids(tmp_path) -> None:
    d = tmp_path / "autextification" / "subtask_1" / "en"
    d.mkdir(parents=True)
    (d / "train.tsv").write_bytes(
        "﻿id\ttext\tlabel\tdomain\n7\thello there world\thuman\tnews\n".encode())
    assert [r.id for r in autextification.load(tmp_path, download=False)] == [
        "autextification:train:7"]


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def _source(records):
    return SimpleNamespace(load=lambda raw_dir, download=True: iter(records))


def _rec(source, i, text, label="human", domain="qa", generator=None) -> Record:
    return Record(id=f"{source}:{i}", text=text, label=label,
                  generator=generator or ("human" if label == "human" else "gpt-4"),
                  domain=domain, subdomain="", source=source)


def test_empty_text_is_dropped_even_with_min_words_zero(tmp_path) -> None:
    reg = {"s": _source([_rec("s", 1, " ​\x00 "), _rec("s", 2, "two words", "ai")])}
    res = build_corpus(["s"], tmp_path, tmp_path / "out", min_words=0, registry=reg)
    assert pq.read_table(res.parquet_path).column("text").to_pylist() == ["two words"]
    assert res.manifest["sources"]["s"]["too_short"] == 1


def test_interrupted_corpus_write_keeps_the_previous_build(tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    reg = {"s": _source([_rec("s", 1, "word " * 40)])}
    first = build_corpus(["s"], tmp_path, out, registry=reg)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    reg = {"s": _source([_rec("s", 2, "other " * 40)])}

    def crash():
        raise KeyboardInterrupt

    monkeypatch.setattr(corpus, "_git_info", crash)  # runs after the parquet is written
    with pytest.raises(KeyboardInterrupt):
        build_corpus(["s"], tmp_path, out, registry=reg)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_version"] == first.corpus_version


def test_interrupted_splits_write_keeps_the_previous_splits(tmp_path, monkeypatch) -> None:
    df = pd.DataFrame({"id": ["a:1", "a:2"], "label": ["human", "ai"],
                       "generator": ["human", "g"], "domain": ["qa", "qa"],
                       "source": ["a", "a"], "n_words": [50, 50]})
    cfg = splits.SplitConfig(fractions={"train": 1.0, "val": 0.0, "test": 0.0})
    splits.write_splits(df, splits.make_splits(df, cfg), cfg, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    cfg2 = splits.SplitConfig(fractions={"train": 0.0, "val": 0.0, "test": 1.0})

    def disk_full(self, *args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_text", disk_full)  # after splits.parquet is written
    with pytest.raises(OSError):
        splits.write_splits(df, splits.make_splits(df, cfg2), cfg2, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_splits_reject_null_ids() -> None:
    df = pd.DataFrame({"id": ["a:1", None], "label": ["human", "ai"],
                       "generator": ["human", "g"], "domain": ["qa", "qa"],
                       "source": ["a", "a"], "n_words": [50, 50]})
    with pytest.raises(ValueError, match="no id"):
        splits.make_splits(df, splits.SplitConfig())


# --------------------------------------------------------------------------- #
# end-to-end leakage: corpus -> splits
# --------------------------------------------------------------------------- #


def test_no_text_or_question_leaks_across_splits(tmp_path) -> None:
    """Variants of one text that differ only in case / punctuation / spacing, spread
    over sources, end up as one row; every answer to one HC3 question stays on one
    side; test_short and the human part of test_ood_generator come from test."""
    hc3, other = [], []
    for q in range(120):
        base = " ".join(f"q{q}w{j}" for j in range(40))
        for i in range(2):
            hc3.append(Record(id=f"hc3:open_qa:{q}:human:{i}", text=f"{base} human{i}",
                              label="human", generator="human", domain="qa",
                              subdomain="open_qa", source="hc3"))
            hc3.append(Record(id=f"hc3:open_qa:{q}:chatgpt:{i}", text=f"{base} bot{i}",
                              label="ai", generator="chatgpt", domain="qa",
                              subdomain="open_qa", source="hc3"))
        gen = "held" if q % 3 == 0 else "gpt-4"
        # same text as an HC3 human answer, re-cased, re-punctuated, re-spaced
        other.append(_rec("m4gt", f"h{q}", f"{base.upper()}   HUMAN0!!", "human"))
        other.append(_rec("m4gt", f"a{q}", " ".join(f"m{q}x{j}" for j in range(q % 90 + 30)),
                          "ai", generator=gen))
    reg = {"hc3": _source(hc3), "m4gt": _source(other)}
    res = build_corpus(["hc3", "m4gt"], tmp_path, tmp_path / "out", registry=reg)
    table = pq.read_table(res.parquet_path).to_pandas()
    assert res.manifest["sources"]["m4gt"]["duplicate"] == 120

    cfg = splits.SplitConfig(held_out_generators=["held"])
    m = splits.make_splits(table, cfg)
    text = dict(zip(table["id"], table["text"]))
    side = {"train": "train", "val": "val"}
    seen: dict[bytes, set[str]] = {}
    for rid, split in zip(m["id"], m["split"]):
        seen.setdefault(normalized_hash(text[rid]), set()).add(side.get(split, "test"))
    assert all(len(s) == 1 for s in seen.values())

    hc3_rows = m[m["id"].str.startswith("hc3:")]
    question = hc3_rows["id"].str.rsplit(":", n=2).str[0]
    hc3_side = hc3_rows["split"].map(lambda s: side.get(s, "test"))
    assert (hc3_side.groupby(question).nunique() == 1).all()

    by_split = m.groupby("split")["id"].apply(set)
    assert by_split["test_short"] <= by_split["test"]
    human = set(table.loc[table["label"] == "human", "id"])
    assert by_split["test_ood_generator"] & human <= by_split["test"]
    assert not by_split["train"] & (by_split["val"] | by_split["test"])
