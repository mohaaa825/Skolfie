"""
tests/test_fairness_sets.py
---------------------------
The fairness test-set loaders (skolfie/data/fairness_sets.py) and
scripts/build_fairness_sets.py, without network: fixture files are written
into the cache directory, and `download` / `urlopen` are monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from skolfie.data import fairness_sets as fs
from skolfie.data.corpus import normalized_hash
from skolfie.data.schema import COLUMNS, DOMAINS

REPO_ROOT = Path(__file__).resolve().parent.parent

ELLIPSE_HEADER = (
    "text_id_kaggle,full_text,gender,grade,race_ethnicity,num_words,task,SES,prompt,Overall,"
    "Cohesion,Syntax,Vocabulary,Phraseology,Grammar,Conventions\n"
)


def _doc(set_name: str, i: int) -> str:
    return f"Essay {i} of the {set_name} set talks about cooking at home and why it matters."


def _write_cache(cache: Path, *, skip: tuple[str, ...] = ()) -> None:
    """Two documents per Liang set, plus a three-row ELLIPSE CSV."""
    for fset in fs.FAIRNESS_SETS.values():
        if fset.source != "liang2023" or fset.name in skip:
            continue
        path = cache / fset.source / fset.path
        path.parent.mkdir(parents=True, exist_ok=True)
        docs = [{"document": _doc(fset.name, i)} for i in range(2)]
        if fset.label == "ai":
            docs[0]["prompt"] = "Write a college essay."  # never part of the text
        path.write_text(json.dumps(docs), encoding="utf-8")
    ellipse = cache / "ellipse" / fs.FAIRNESS_SETS["ellipse"].path
    ellipse.parent.mkdir(parents=True, exist_ok=True)
    ellipse.write_text(
        ELLIPSE_HEADER
        + '0012345678,"First paragraph here.\r\n\r\nSecond  paragraph here.",Male,8,'
          "Hispanic/Latino,6,Independent,x,Distance learning,2.5,3,3,3,3,3,3\n"
        + 'ABCDEF123456,"Another learner essay, itâ going fine.",Female,12,White,6,'
          "Independent,x,Career,4.0,3,3,3,3,3,3\n"
        + "EMPTY0000001,,Male,10,White,0,Independent,x,Career,3.0,3,3,3,3,3,3\n",
        encoding="utf-8",
    )


@pytest.fixture()
def no_network(monkeypatch):
    def refuse(url, dest, **kw):
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(fs, "download", refuse)


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    _write_cache(path)
    return path


def _build(cache: Path, **kw) -> pd.DataFrame:
    return fs.build_fairness_sets(cache, log=lambda msg: None, **kw)


# ------------------------------------------------------------------ contract

def test_column_contract_and_dtypes(cache, no_network):
    df = _build(cache)
    assert list(df.columns) == fs.FAIRNESS_COLUMNS
    assert fs.FAIRNESS_COLUMNS[: len(COLUMNS)] == COLUMNS
    for col in fs.FAIRNESS_COLUMNS:
        if col in ("n_words", "edited_by_ai", "proficiency", "grade"):
            continue
        assert pd.api.types.is_string_dtype(df[col]), col
    assert df["n_words"].dtype == "int64"
    assert df["edited_by_ai"].dtype == bool
    assert df["proficiency"].dtype == "float64" and df["grade"].dtype == "float64"

    assert df["id"].is_unique
    assert set(df["set"]) == set(fs.FAIRNESS_SETS)
    assert set(df["label"]) == {"human", "ai"}
    assert set(df["domain"]) <= set(DOMAINS)
    assert set(df["writer"]) <= set(fs.WRITERS)
    assert (df["split_hint"] == "").all()
    assert ((df["generator"] == "human") == (df["label"] == "human")).all()
    assert (df["n_words"] == df["text"].str.split().str.len()).all()
    for row in df.itertuples():
        assert row.id.startswith(f"{row.source}:{row.set}:")
        assert row.source == fs.FAIRNESS_SETS[row.set].source
    # Liang rows: one per document, in order, the prompt field ignored
    toefl = df[df["set"] == "toefl_real"]
    assert list(toefl["id"]) == ["liang2023:toefl_real:0", "liang2023:toefl_real:1"]
    assert not df["text"].str.contains("Write a college essay").any()


def test_set_writer_and_edited_mapping(cache, no_network):
    df = _build(cache)
    per_set = df.groupby("set")[["label", "writer", "edited_by_ai", "group", "source"]].first()
    expected = {
        "toefl_real": ("human", "non-native", False),
        "toefl_gpt4polished": ("human", "non-native", True),
        "college_real": ("human", "native", False),
        "cs224n_real": ("human", "unknown", False),
        "hewlett_real": ("human", "native", False),
        "hewlett_gptsimplify": ("human", "native", True),
        "college_gpt3": ("ai", "ai", False),
        "college_gpt3prompteng": ("ai", "ai", False),
        "cs224n_gpt3": ("ai", "ai", False),
        "cs224n_gpt3prompteng": ("ai", "ai", False),
        "ellipse": ("human", "non-native", False),
    }
    assert set(per_set.index) == set(expected)
    for name, (label, writer, edited) in expected.items():
        row = per_set.loc[name]
        assert (row["label"], row["writer"], bool(row["edited_by_ai"])) == (label, writer, edited)
        assert row["group"]
    assert set(df.loc[df["edited_by_ai"], "set"]) == {"toefl_gpt4polished", "hewlett_gptsimplify"}
    assert set(df.loc[df["source"] == "ellipse", "set"]) == {"ellipse"}
    assert (df.loc[df["label"] == "ai", "generator"] == "chatgpt-3.5").all()


def test_ellipse_proficiency_grade_and_text(cache, no_network):
    df = _build(cache, sources=("ellipse",))
    assert set(df["set"]) == {"ellipse"}
    # the empty essay is dropped; the all-digit id keeps its leading zero
    assert list(df["id"]) == ["ellipse:ellipse:0012345678", "ellipse:ellipse:ABCDEF123456"]
    assert list(df["proficiency"]) == [2.5, 4.0]
    assert list(df["grade"]) == [8.0, 12.0]
    first, second = df["text"]
    assert first == "First paragraph here.\n\nSecond paragraph here."  # paragraph break kept
    assert second == "Another learner essay, it' going fine."  # mojibake repaired
    assert df.attrs["build_stats"]["empty"] == {"ellipse": 1}

    liang = _build(cache, sources=("liang2023",))
    assert liang["proficiency"].isna().all() and liang["grade"].isna().all()


def test_unknown_source_rejected(cache, no_network):
    with pytest.raises(ValueError, match="unknown fairness source"):
        _build(cache, sources=("nope",))


# ------------------------------------------------------------------ text repairs

def test_repairs():
    hewlett = fs.FAIRNESS_SETS["hewlett_real"]
    assert fs.repair_text("they\x92ve got it", hewlett) == "they’ve got it"
    assert fs.repair_text("soft\xadhyphen", hewlett) == "softhyphen"
    assert fs.repair_text(None, hewlett) == ""
    cs = fs.FAIRNESS_SETS["cs224n_real"]
    assert fs.repair_text("Title: RobustQA\nAbstract: In recent years, QA.", cs) == (
        "In recent years, QA."
    )
    # the header is only cut for the human CS224N set
    assert fs.repair_text("Title: X\nAbstract: Y", hewlett) == "Title: X Abstract: Y"


def test_empty_and_duplicates_dropped(tmp_path, no_network):
    cache = tmp_path / "cache"
    _write_cache(cache)
    toefl = cache / "liang2023" / fs.FAIRNESS_SETS["toefl_real"].path
    polished = cache / "liang2023" / fs.FAIRNESS_SETS["toefl_gpt4polished"].path
    toefl.write_text(json.dumps([
        {"document": "I prefer to cook at home. It is cheaper."},
        {"document": "   "},
        {"document": "I prefer to cook at home, it is cheaper!"},  # same normalised text
        {"document": "Another essay."},
    ]), encoding="utf-8")
    # a duplicate in a later set is dropped from that set (first kept)
    polished.write_text(json.dumps([{"document": "Another essay."}, {"document": "New."}]),
                        encoding="utf-8")
    messages: list[str] = []
    df = fs.build_fairness_sets(cache, sources=("liang2023",), log=messages.append)
    assert list(df.loc[df["set"] == "toefl_real", "id"]) == [
        "liang2023:toefl_real:0", "liang2023:toefl_real:3",
    ]
    assert list(df.loc[df["set"] == "toefl_gpt4polished", "text"]) == ["New."]
    stats = df.attrs["build_stats"]
    assert stats["empty"] == {"toefl_real": 1}
    assert stats["duplicate"] == {"toefl_real": 1, "toefl_gpt4polished": 1}
    assert any("dropped 2 exact duplicates" in m for m in messages)
    assert any("dropped 1 empty" in m for m in messages)


def test_bad_liang_file(tmp_path, no_network):
    cache = tmp_path / "cache"
    _write_cache(cache)
    path = cache / "liang2023" / fs.FAIRNESS_SETS["college_real"].path
    path.write_text(json.dumps([{"text": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="document"):
        _build(cache, sources=("liang2023",))


# ------------------------------------------------------------------ downloads

def test_missing_files_downloaded_once_then_cached(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    _write_cache(seed)
    calls: list[str] = []

    def fake_download(url, dest, **kw):
        calls.append(url)
        rel = Path(dest).relative_to(tmp_path / "cache")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((seed / rel).read_bytes())
        return dest

    monkeypatch.setattr(fs, "download", fake_download)
    first = _build(tmp_path / "cache")
    assert len(calls) == len(fs.FAIRNESS_SETS)
    assert fs.LIANG_BASE_URL + "Human_Data/TOEFL_real_91/data.json" in calls
    assert fs.ELLIPSE_URL in calls

    second = _build(tmp_path / "cache")  # everything cached now: no new download
    assert len(calls) == len(fs.FAIRNESS_SETS)
    pd.testing.assert_frame_equal(first, second)


def test_no_download_names_url(tmp_path, no_network):
    with pytest.raises(fs.FairnessDownloadError, match="downloading is off.*raw.githubusercontent"):
        _build(tmp_path / "empty", sources=("ellipse",), download=False)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.payload


def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    attempts: list[float] = []
    sleeps: list[float] = []

    def flaky(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise urllib.error.URLError("connection reset")
        return _Response(b"ok")

    monkeypatch.setattr(fs.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fs, "_sleep", sleeps.append)
    dest = tmp_path / "sub" / "file.json"
    assert fs.download("https://example.invalid/f", dest, timeout=5) == dest
    assert dest.read_bytes() == b"ok"
    assert attempts == [5, 5, 5]
    assert sleeps == [1.0, 2.0]
    assert [p.name for p in dest.parent.iterdir()] == ["file.json"]  # no temp file left


def test_download_gives_up_with_url_in_error(tmp_path, monkeypatch):
    calls: list[str] = []

    def down(request, timeout):
        calls.append(request.full_url)
        raise TimeoutError("timed out")

    monkeypatch.setattr(fs.urllib.request, "urlopen", down)
    monkeypatch.setattr(fs, "_sleep", lambda s: None)
    dest = tmp_path / "file.json"
    url = "https://example.invalid/data.json"
    with pytest.raises(fs.FairnessDownloadError,
                       match=r"could not download https://example\.invalid/data\.json "
                             r"\(3 attempts\): TimeoutError"):
        fs.download(url, dest)
    assert calls == [url] * 3
    assert not dest.exists()


def test_download_does_not_retry_404(tmp_path, monkeypatch):
    calls: list[str] = []

    def missing(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(fs.urllib.request, "urlopen", missing)
    monkeypatch.setattr(fs, "_sleep", lambda s: None)
    with pytest.raises(fs.FairnessDownloadError, match=r"\(1 attempt\): HTTPError"):
        fs.download("https://example.invalid/x", tmp_path / "x")
    assert len(calls) == 1


# ------------------------------------------------------------------ metadata

def test_source_metadata():
    assert set(fs.FAIRNESS_SOURCES) == {"liang2023", "ellipse"}
    for meta in fs.SOURCES.values():
        assert meta.license and meta.citation and meta.url.startswith("https://")
    assert "CC BY 4.0" in fs.SOURCES["ellipse"].license
    assert "MIT" in fs.SOURCES["liang2023"].license


# ------------------------------------------------------------------ script

def _load_script():
    spec = importlib.util.spec_from_file_location(
        "skolfie_build_fairness_sets_script", REPO_ROOT / "scripts" / "build_fairness_sets.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_drops_corpus_overlap_and_writes_manifest(cache, tmp_path, no_network, capsys):
    script = _load_script()
    # The corpus holds one TOEFL essay (with different case/punctuation) and one
    # ELLIPSE essay: both must be dropped from the fairness sets.
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({
        "id": ["c:1", "c:2", "c:3"],
        "text": [_doc("toefl_real", 0).upper().replace(".", "!"),
                 "First paragraph here. Second paragraph here.",
                 "Unrelated corpus text."],
    }).to_parquet(corpus, index=False)
    out = tmp_path / "out" / "fairness_sets.parquet"
    rc = script.main(["--out", str(out), "--cache-dir", str(cache), "--corpus", str(corpus)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == fs.FAIRNESS_COLUMNS
    assert "liang2023:toefl_real:0" not in set(df["id"])
    assert "ellipse:ellipse:0012345678" not in set(df["id"])
    assert not set(df["text"].map(normalized_hash)) & {
        normalized_hash(t) for t in pd.read_parquet(corpus)["text"]
    }

    manifest = json.loads((out.parent / "fairness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_rows"] == len(df)
    assert manifest["corpus_overlap"] == {
        "corpus": str(corpus), "dropped": 2,
        "dropped_by_set": {"toefl_real": 1, "ellipse": 1},
    }
    assert manifest["by_set"]["toefl_real"]["n"] == 1
    assert manifest["by_set"]["ellipse"]["n"] == 1
    assert manifest["by_set"]["toefl_gpt4polished"]["edited_by_ai"] is True
    assert sum(v["n"] for v in manifest["by_set"].values()) == len(df)
    assert manifest["by_label"] == df["label"].value_counts().to_dict()
    assert manifest["by_writer"] == df["writer"].value_counts().to_dict()
    assert manifest["by_label_writer"]["human/non-native"] == int(
        ((df["label"] == "human") & (df["writer"] == "non-native")).sum()
    )
    assert manifest["edited_by_ai"] == int(df["edited_by_ai"].sum())
    assert set(manifest["sources"]) == {"liang2023", "ellipse"}
    assert manifest["sources"]["ellipse"]["license"] == "CC BY 4.0"
    assert manifest["sources"]["liang2023"]["citation"].startswith("Liang")
    assert manifest["build_stats"]["empty"] == {"ellipse": 1}
    assert manifest["created_at"]
    # written atomically: no temporary file left behind
    assert {p.name for p in out.parent.iterdir()} == {
        "fairness_sets.parquet", "fairness_manifest.json"
    }
    assert "toefl_real" in capsys.readouterr().out


def test_script_without_corpus_and_errors(cache, tmp_path, monkeypatch, capsys):
    script = _load_script()
    out = tmp_path / "o" / "f.parquet"
    monkeypatch.setattr(fs, "download", lambda url, dest, **kw: pytest.fail("download"))
    assert script.main(["-q", "--out", str(out), "--cache-dir", str(cache),
                        "--sources", "liang2023"]) == 0
    manifest = json.loads((out.parent / "fairness_manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus_overlap"] == {"corpus": None, "dropped": 0, "dropped_by_set": {}}
    assert set(manifest["sources"]) == {"liang2023"}

    # unreadable corpus -> 2; unknown source -> 2 (argparse)
    assert script.main(["--out", str(out), "--cache-dir", str(cache),
                        "--corpus", str(tmp_path / "missing.parquet")]) == 2
    with pytest.raises(SystemExit) as exc:
        script.main(["--out", str(out), "--sources", "nope"])
    assert exc.value.code == 2

    # a failed download -> 1 with the URL in the message
    def fail(url, dest, **kw):
        raise fs.FairnessDownloadError(f"could not download {url} (3 attempts): boom")

    monkeypatch.setattr(fs, "download", fail)
    capsys.readouterr()
    assert script.main(["--out", str(out), "--cache-dir", str(tmp_path / "nocache"),
                        "--sources", "ellipse"]) == 1
    assert fs.ELLIPSE_URL in capsys.readouterr().err
