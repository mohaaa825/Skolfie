"""
tests/test_corpus.py
---------------------
Tests for skolfie/data/corpus.py and scripts/build_corpus.py, using fake
in-memory sources (no network, no real dataset loaders).
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from skolfie.data import corpus
from skolfie.data.corpus import EmptyCorpusError, SourceError, build_corpus, clean_text
from skolfie.data.schema import COLUMNS, Record

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def words(tag: str, n: int = 40) -> str:
    """A text of exactly n words, unique per tag."""
    return " ".join(f"{tag}w{j}" for j in range(n))


def rec(source: str, i: int | str, text: str, label: str = "human", domain: str = "news",
        generator: str | None = None) -> Record:
    gen = generator or ("human" if label == "human" else "gpt-4")
    return Record(id=f"{source}:{i}", text=text, label=label, generator=gen, domain=domain,
                  subdomain=f"{domain}-sub", source=source, split_hint="train")


def fake_source(name: str, records: list | Callable[[], Iterator]) -> SimpleNamespace:
    def load(raw_dir: Path, download: bool = True) -> Iterator:
        yield from (records() if callable(records) else records)

    return SimpleNamespace(NAME=name, load=load)


def failing_source(name: str, exc: Exception, after: list | None = None) -> SimpleNamespace:
    def load(raw_dir: Path, download: bool = True) -> Iterator:
        yield from after or []
        raise exc

    return SimpleNamespace(NAME=name, load=load, URL=f"https://example.org/{name}.zip")


def mixed_records(source: str, n_human: int, n_ai: int, domain: str = "news") -> list[Record]:
    out = [rec(source, f"{domain}-h{i}", words(f"{source}{domain}h{i}"), "human", domain)
           for i in range(n_human)]
    out += [rec(source, f"{domain}-a{i}", words(f"{source}{domain}a{i}"), "ai", domain)
            for i in range(n_ai)]
    return out


def build(tmp_path: Path, registry: dict, **kw) -> corpus.CorpusResult:
    kw.setdefault("download", False)
    return build_corpus(list(registry), tmp_path / "raw", tmp_path / "out",
                        registry=registry, **kw)


def read_ids(result: corpus.CorpusResult) -> list[str]:
    return pq.read_table(result.parquet_path).column("id").to_pylist()


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Hello,   World!  ", "Hello, World!"),
        ("tabs\tand nbsp thin", "tabs and nbsp thin"),
        ("line one\nline two", "line one line two"),
        ("para one.\n\npara two.", "para one.\n\npara two."),
        ("para one.\n \n\n\t\npara two.", "para one.\n\npara two."),
        ("windows\r\n\r\nbreaks\rhere", "windows\n\nbreaks here"),
        ("a b c", "a\n\nb c"),
        ("ctrl\x00\x07\x1b chars\x7f\x9f", "ctrl chars"),
        ("﻿BOM and zero​width", "BOM and zerowidth"),
        ("Keep CASE, punctuation; and \"quotes\"!?", "Keep CASE, punctuation; and \"quotes\"!?"),
        ("\n\n\nleading and trailing\n\n\n", "leading and trailing"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_clean_text(raw, expected) -> None:
    assert clean_text(raw) == expected


def test_clean_text_nfc_and_zwnj() -> None:
    decomposed = "café"  # e + combining acute
    assert clean_text(decomposed) == "café"
    persian = "می‌خواهم"  # ZWNJ must survive
    assert clean_text(persian) == persian


def test_clean_text_idempotent() -> None:
    raw = " A\r\n\r\n  b\tc  \n\n\n d\x00 "
    once = clean_text(raw)
    assert clean_text(once) == once == "A\n\nb c\n\nd"


# --------------------------------------------------------------------------- #
# build_corpus
# --------------------------------------------------------------------------- #


def test_length_filter_and_n_words_recomputed(tmp_path: Path) -> None:
    records = [
        rec("s", 1, words("short", 5)),
        rec("s", 2, words("ok", 10)),
        rec("s", 3, "  " + words("spaced", 12).replace(" ", "   \n ") + "  "),
        rec("s", 4, words("long", 50)),
    ]
    result = build(tmp_path, {"s": fake_source("s", records)}, min_words=10, max_words=20)
    table = pq.read_table(result.parquet_path)
    assert table.column_names == COLUMNS
    assert table.column("id").to_pylist() == ["s:2", "s:3"]
    assert table.column("n_words").to_pylist() == [10, 12]
    assert "   " not in table.column("text")[1].as_py()
    stats = result.manifest["sources"]["s"]
    assert (stats["raw"], stats["too_short"], stats["too_long"], stats["kept"]) == (4, 1, 1, 2)


def test_exact_dedupe_normalised_across_sources(tmp_path: Path) -> None:
    base = words("dup")
    variant = base.upper().replace(" ", " , ") + "!!!"  # same after normalisation
    registry = {
        "b": fake_source("b", [rec("b", 1, base), rec("b", 2, words("other"))]),
        "a": fake_source("a", [rec("a", 1, variant), rec("a", 2, base)]),
    }
    result = build(tmp_path, registry, min_words=10)
    # smallest id wins, independent of source order
    assert sorted(read_ids(result)) == ["a:1", "b:2"]
    assert result.manifest["sources"]["a"]["duplicate"] == 1
    assert result.manifest["sources"]["b"]["duplicate"] == 1
    assert result.n_records == 2


def test_conflicting_labels_drop_all_copies(tmp_path: Path) -> None:
    text = words("conflict")
    records = [
        rec("s", 1, text, "human"),
        rec("s", 2, text, "ai"),
        rec("s", 3, text + " .", "human"),
        rec("s", 4, words("fine"), "ai"),
    ]
    result = build(tmp_path, {"s": fake_source("s", records)})
    assert read_ids(result) == ["s:4"]
    assert result.manifest["sources"]["s"]["conflicting_label"] == 3
    assert result.manifest["sources"]["s"]["duplicate"] == 0


def test_invalid_and_duplicate_ids(tmp_path: Path) -> None:
    good = rec("s", 1, words("good"))
    records = [
        good,
        rec("s", 1, words("same id other text")),
        {"id": "s:9", "text": words("baddomain"), "label": "human", "generator": "human",
         "domain": "not-a-domain", "subdomain": "", "source": "s"},
        {**rec("s", 5, words("asdict")).to_dict()},
        "not a record",
    ]
    result = build(tmp_path, {"s": fake_source("s", records)})
    stats = result.manifest["sources"]["s"]
    assert stats["invalid"] == 2
    assert stats["duplicate_id"] == 1
    assert read_ids(result) == ["s:1", "s:5"]
    assert any("reusing an id" in w for w in result.manifest["warnings"])


def test_balance_downsample(tmp_path: Path) -> None:
    records = (mixed_records("s", 10, 4, "news") + mixed_records("s", 3, 7, "qa")
               + mixed_records("s", 5, 0, "legal"))
    reg = {"s": fake_source("s", records)}
    result = build(tmp_path, reg, balance="downsample", seed=1)
    table = pq.read_table(result.parquet_path)
    pairs = Counter(zip(table.column("domain").to_pylist(), table.column("label").to_pylist()))
    assert pairs == {("news", "human"): 4, ("news", "ai"): 4, ("qa", "human"): 3, ("qa", "ai"): 3}
    assert result.manifest["sources"]["s"]["balance"] == 6 + 4 + 5
    assert any("legal" in w for w in result.manifest["warnings"])

    # seeded: same seed gives the same sample, another seed (very likely) not
    again = build(tmp_path / "again", reg, balance="downsample", seed=1)
    assert read_ids(again) == read_ids(result)
    other = build(tmp_path / "other", reg, balance="downsample", seed=2)
    assert read_ids(other) != read_ids(result)

    with pytest.raises(ValueError):
        build(tmp_path / "bad", reg, balance="upsample")


def test_manifest_contents_and_outputs(tmp_path: Path) -> None:
    registry = {
        "s1": fake_source("s1", mixed_records("s1", 3, 2)),
        "s2": fake_source("s2", mixed_records("s2", 1, 2, "academic")),
    }
    result = build(tmp_path, registry, min_words=5, max_words=100, seed=3)
    m = json.loads(result.manifest_path.read_text())
    assert m == result.manifest
    for key in ("build_time", "git_commit", "parameters", "sources", "counts", "parquet_sha256",
                "corpus_version", "content_sha256", "n_records"):
        assert key in m
    assert m["parameters"]["min_words"] == 5
    assert m["parameters"]["max_words"] == 100
    assert m["parameters"]["seed"] == 3
    assert m["parameters"]["sources"] == ["s1", "s2"]
    assert m["sources"]["s1"]["raw"] == 5 and m["sources"]["s1"]["kept"] == 5
    assert m["n_records"] == 8
    assert m["corpus_version"].startswith("v-")
    assert m["corpus_version"][2:] == m["content_sha256"][:16]
    assert m["parquet_sha256"] == corpus._file_sha256(result.parquet_path)
    assert {"label": "ai", "generator": "gpt-4", "domain": "academic", "count": 2} in m["counts"]
    assert sum(c["count"] for c in m["counts"]) == 8
    assert m["totals"]["label"] == {"ai": 4, "human": 4}

    ids = read_ids(result)
    assert ids == sorted(ids)
    assert pq.read_table(result.parquet_path).schema.metadata is None

    md = result.stats_path.read_text()
    assert md == result.stats_markdown
    assert m["corpus_version"] in md
    assert "| s1 |" in md and "| academic |" in md and "| gpt-4 |" in md


def test_determinism_same_input_same_version(tmp_path: Path) -> None:
    recs = mixed_records("s1", 6, 6) + mixed_records("s1", 2, 3, "qa")
    first = build(tmp_path / "1", {"s1": fake_source("s1", recs), "s2": fake_source("s2", [])})
    # same records in a different order and source order: same corpus
    second = build(tmp_path / "2",
                   {"s2": fake_source("s2", []), "s1": fake_source("s1", recs[::-1])})
    assert first.corpus_version == second.corpus_version
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    assert first.parquet_path.read_bytes() == second.parquet_path.read_bytes()

    changed = list(recs)
    changed[0] = rec("s1", "news-h0", words("s1newsh0") + " extra", "human")
    third = build(tmp_path / "3", {"s1": fake_source("s1", changed)})
    assert third.corpus_version != first.corpus_version


def test_failing_source_is_skipped_or_strict(tmp_path: Path) -> None:
    registry = {
        "ok": fake_source("ok", mixed_records("ok", 2, 2)),
        "bad": failing_source("bad", FileNotFoundError("data/raw/bad/train.jsonl not found"),
                              after=mixed_records("bad", 1, 1)),
    }
    result = build(tmp_path, registry)
    assert result.n_records == 4  # partial output of the failed source is discarded
    assert "bad" in result.errors
    assert "data/raw/bad/train.jsonl" in result.errors["bad"]
    assert "https://example.org/bad.zip" in result.errors["bad"]
    assert result.manifest["sources"]["bad"]["kept"] == 0
    assert result.manifest["sources"]["bad"]["error"]

    with pytest.raises(SourceError, match="bad"):
        build(tmp_path / "strict", registry, strict=True)
    assert not (tmp_path / "strict" / "out" / "corpus.parquet").exists()


def test_empty_corpus_raises_and_writes_nothing(tmp_path: Path) -> None:
    registry = {"s": fake_source("s", [rec("s", 1, "too short")])}
    with pytest.raises(EmptyCorpusError):
        build(tmp_path, registry)
    assert not (tmp_path / "out").exists()


def test_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        build_corpus(["nope"], tmp_path, tmp_path / "out", registry={})


def test_streams_generator_once(tmp_path: Path) -> None:
    calls = []

    def gen() -> Iterator[Record]:
        calls.append(1)
        for i in range(500):
            yield rec("g", i, words(f"g{i}", 35), "human" if i % 2 else "ai", "casual")

    result = build(tmp_path, {"g": fake_source("g", gen)})
    assert calls == [1]
    assert result.n_records == 500


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_cli():
    spec = importlib.util.spec_from_file_location("skolfie_build_corpus_script",
                                                  REPO_ROOT / "scripts" / "build_corpus.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli(monkeypatch):
    registry = {
        "hc3": fake_source("hc3", mixed_records("hc3", 4, 6, "qa")),
        "m4gt": fake_source("m4gt", mixed_records("m4gt", 5, 3, "academic")),
        "broken": failing_source("broken", ConnectionError("could not reach https://hf.co/x")),
    }
    monkeypatch.setattr(corpus, "load_registry", lambda: registry)
    return _load_cli()


def test_cli_builds_and_continues_past_failure(cli, tmp_path: Path, capsys) -> None:
    out = tmp_path / "processed"
    code = cli.main(["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(out),
                     "--min-words", "10", "--max-words", "500", "--no-download", "--quiet"])
    captured = capsys.readouterr()
    assert code == 0
    assert (out / "corpus.parquet").exists()
    assert (out / "manifest.json").exists()
    assert (out / "stats.md").exists()
    assert "source 'broken' failed: ConnectionError" in captured.err
    assert "https://hf.co/x" in captured.err
    assert captured.err.count("source 'broken' failed") == 1
    assert "## Sources" in captured.out and "| hc3 |" in captured.out
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_records"] == 18
    assert m["parameters"]["download"] is False
    assert m["parameters"]["sources"] == ["hc3", "m4gt", "broken"]


def test_cli_sources_balance_and_strict(cli, tmp_path: Path, capsys) -> None:
    out = tmp_path / "bal"
    code = cli.main(["--sources", "hc3,m4gt", "--out-dir", str(out), "--raw-dir",
                     str(tmp_path), "--balance", "downsample", "-q"])
    assert code == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_records"] == 4 * 2 + 3 * 2
    assert m["parameters"]["balance"] == "downsample"

    code = cli.main(["--strict", "--out-dir", str(tmp_path / "strict"), "-q"])
    assert code != 0
    assert "broken" in capsys.readouterr().err
    assert not (tmp_path / "strict" / "corpus.parquet").exists()


def test_cli_no_records_and_bad_args(cli, tmp_path: Path, capsys) -> None:
    code = cli.main(["--sources", "broken", "--out-dir", str(tmp_path / "none"), "-q"])
    assert code == 1
    assert "broken" in capsys.readouterr().err

    code = cli.main(["--sources", "hc3", "--min-words", "1000", "--max-words", "2000",
                     "--out-dir", str(tmp_path / "none2"), "-q"])
    assert code == 1
    assert "hc3: 10 raw records, 0 kept" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        cli.main(["--sources", "hc3,nope"])
    assert exc.value.code == 2
    assert "nope" in capsys.readouterr().err
