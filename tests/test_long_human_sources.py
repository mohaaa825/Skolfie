"""
tests/test_long_human_sources.py
--------------------------------
The long human academic sources, pes2o (peS2o shards) and theses (OAI-PMH
harvest of CC BY / BY-SA student theses), fully offline: fixtures are built
in tmp_path (gzipped JSON lines, a tiny PDF written with pypdf, OAI-PMH XML
served by a fake requests session).
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import shutil
import textwrap
from pathlib import Path

import pytest

from skolfie.data.schema import COLUMNS, Record
from skolfie.data.sources import SOURCES, get_source, pes2o, theses
from skolfie.data.sources.theses import Endpoint, OaiRecord

# --------------------------------------------------------------------- helpers

_SUBJECTS = ["The analysis", "This study", "Our model", "The survey", "The committee",
             "Each participant", "The method", "The proposed approach", "The data set",
             "Previous research", "The teacher", "The experiment"]
_VERBS = ["shows", "suggests", "indicates", "confirms", "challenges", "explains",
          "describes", "examines", "supports", "questions"]
_OBJECTS = ["that the effect is stronger in rural regions than in the cities",
            "how the students respond to the new form of feedback in class",
            "the role of trust in the relationship between the firm and its customers",
            "a clear difference between the two groups of the sample",
            "why the results of the earlier work could not be reproduced",
            "the importance of careful measurement in the field",
            "that the policy has had only a modest effect on the outcome",
            "the way in which the network adapts to changes in its environment"]


def para(rng: random.Random, n_sentences: int = 7) -> str:
    """An English prose paragraph of about 16 words per sentence."""
    return " ".join(f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}."
                    for _ in range(n_sentences))


def prose(seed: int, n_paragraphs: int, n_sentences: int = 7) -> list[str]:
    rng = random.Random(seed)
    return [para(rng, n_sentences) for _ in range(n_paragraphs)]


REFERENCES = "\n".join([
    "[1] Smith, J., Brown, K. et al. (2015). A study of things. J. Things 12(3), 45-67.",
    "[2] Jones, A. et al. (2017). Another study. Proc. Conf. 2017, 101-110.",
    "[3] Lee, B. et al. (2019). More results. Nature 555, 1-10.",
    "[4] Wang, C. et al. (2012). Old results. Science 300, 20-30.",
])
TABLE = "\n".join(["Group  N  Mean  SD", "A  12  3.4  0.9", "B  15  2.9  1.1", "C  9  3.1  0.7"])


def _words(text: str) -> int:
    return len(text.split())


def _check_records(records: list[Record], name: str, subdomain: str,
                   min_words: int = 800, max_words: int = 1500) -> None:
    ids = [r.id for r in records]
    assert len(ids) == len(set(ids))
    for r in records:
        assert isinstance(r, Record)
        assert r.id.startswith(f"{name}:")
        assert (r.label, r.generator, r.domain, r.subdomain, r.source) == (
            "human", "human", "academic", subdomain, name)
        assert set(r.to_dict()) == set(COLUMNS)
        assert r.text == r.text.strip()
        assert min_words <= r.n_words <= max_words
        assert "et al." not in r.text and "Mean  SD" not in r.text


# -------------------------------------------------------------------- registry

def test_registry_contains_both_sources():
    assert SOURCES["pes2o"] is pes2o and SOURCES["theses"] is theses
    assert get_source("PES2O") is pes2o
    for module in (pes2o, theses):
        for attr in ("DESCRIPTION", "CITATION", "LICENSE_NOTE", "MANUAL_DOWNLOAD"):
            assert isinstance(getattr(module, attr), str) and getattr(module, attr)


@pytest.mark.parametrize("module", [pes2o, theses])
def test_missing_files_raise_clear_error(module, tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        list(module.load(tmp_path, download=False))
    assert str(tmp_path / module.NAME) in str(info.value)


# ---------------------------------------------------------- shared heuristics

def test_junk_and_heading_detection():
    assert pes2o.looks_like_junk(REFERENCES)
    assert pes2o.looks_like_junk(TABLE)
    assert pes2o.looks_like_junk("12.3 45.6 78.9 10.1 11.2 13.4 15.6")
    assert not pes2o.looks_like_junk(para(random.Random(0)))
    assert pes2o.looks_like_heading("2.1 Related work")
    assert not pes2o.looks_like_heading(para(random.Random(1)))


def test_looks_english():
    assert pes2o.looks_english(" ".join(prose(0, 3)))
    german = ("Die Analyse zeigt, dass die Wirkung in ländlichen Gebieten stärker ist als in "
              "den Städten, und die Ergebnisse früherer Arbeiten wurden bestätigt. ") * 5
    assert not pes2o.looks_english(german)


def test_make_sections_word_ranges_and_boundaries():
    paragraphs = prose(1, 30)               # ~112 words each
    sections = pes2o.make_sections(paragraphs, 300, 500)
    assert len(sections) >= 5
    for s in sections:
        assert 300 <= _words(s) <= 500
    # a junk paragraph ends the run: no section spans it, nor contains it
    mixed = prose(2, 3) + [REFERENCES] + prose(3, 3)
    sections = pes2o.make_sections(mixed, 300, 500)
    assert all("et al." not in s for s in sections)
    assert all(_words(s) >= 300 for s in sections)
    # one paragraph over max_words is dropped
    assert pes2o.make_sections([" ".join(prose(4, 10))], 300, 500) == []
    # headings are skipped without ending the run
    with_heads = [p for pair in zip(["Results", "Discussion", "Method"], prose(5, 3))
                  for p in pair]
    assert pes2o.make_sections(with_heads, 200, 500) == ["\n\n".join(prose(5, 3))]
    with pytest.raises(ValueError):
        pes2o.make_sections([], 500, 300)


def test_year_of():
    assert pes2o.year_of("2019-03-01T00:00:00.000Z") == 2019
    assert pes2o.year_of(2021) == 2021
    assert pes2o.year_of("unknown") is None
    assert pes2o.year_of(None) is None


# ------------------------------------------------------------------------ pes2o

def _paper(pid: str, created: str | None, paragraphs: list[str], source: str = "s2orc/valid"):
    row = {"id": pid, "text": "\n\n".join(["A Title of a Paper", *paragraphs]),
           "source": source, "version": "v2", "added": "2023-01-01"}
    if created is not None:
        row["created"] = created
    return row


def _pes2o_rows() -> list[dict]:
    rows = []
    for i in range(12):
        body = prose(100 + i, 20)
        body.insert(10, REFERENCES)
        rows.append(_paper(f"10{i}", f"{2010 + i % 10}-05-01T00:00:00.000Z", body))
    rows.append(_paper("200", "2022-05-01", prose(200, 14)))      # too recent
    rows.append(_paper("201", "2023", prose(201, 14)))            # too recent
    rows.append(_paper("202", None, prose(202, 14)))              # no date: kept
    rows.append(_paper("203", "2015", prose(203, 3)))             # too short
    rows.append(_paper("204", "2015", prose(204, 14), source="s2ag/valid"))  # abstracts
    rows.append(_paper("205", "2015", [p + " Über straße für Größe Äpfel öl ü ä ö ß é è"
                                       " ñ ç ø å æ œ " * 4 for p in prose(205, 14)]))
    return rows


def _write_pes2o(raw: Path, rows: list[dict], name: str = "validation-00000-of-00001.json.gz"):
    path = raw / "pes2o" / "data" / "v2" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_pes2o_load_filters_and_sections(tmp_path):
    _write_pes2o(tmp_path, _pes2o_rows())
    records = list(pes2o.load(tmp_path, download=False, max_texts=100))
    _check_records(records, "pes2o", "research_paper")
    papers = [r.id.split(":")[1] for r in records]
    assert len(papers) == len(set(papers))                       # one section per paper
    assert set(papers) == {f"10{i}" for i in range(12)} | {"202"}
    assert all(r.id.rsplit(":", 1)[1].startswith("s") for r in records)


def test_pes2o_before_year_none_keeps_recent_papers(tmp_path):
    _write_pes2o(tmp_path, _pes2o_rows())
    papers = {r.id.split(":")[1] for r in pes2o.load(tmp_path, download=False, max_texts=100,
                                                     before_year=None)}
    assert {"200", "201"} <= papers


def test_pes2o_is_deterministic_and_capped(tmp_path):
    _write_pes2o(tmp_path, _pes2o_rows())
    a = [r.to_dict() for r in pes2o.load(tmp_path, download=False, max_texts=5, seed=3)]
    b = [r.to_dict() for r in pes2o.load(tmp_path, download=False, max_texts=5, seed=3)]
    assert a == b and len(a) == 5
    everything = {r.id for r in pes2o.load(tmp_path, download=False, max_texts=0, seed=3)}
    assert {r["id"] for r in a} <= everything


def test_pes2o_reads_jsonl_and_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _pes2o_rows()
    directory = tmp_path / "pes2o"
    directory.mkdir()
    with open(directory / "a.jsonl", "w") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    table = pa.Table.from_pylist([{k: r.get(k) for k in ("id", "text", "created")}
                                  for r in rows[1:3]])
    pq.write_table(table, directory / "b.parquet")
    papers = {r.id.split(":")[1] for r in pes2o.load(tmp_path, download=False)}
    assert papers == {"100", "101", "102"}


def test_pes2o_choose_shards():
    files = [".gitattributes", "README.md", "peS2o.py",
             "data/v1/train-00000-of-00020.json.gz", "data/v1/valid-00000-of-00001.json.gz",
             "data/v2/train-00001-of-00020.json.gz", "data/v2/train-00000-of-00020.json.gz",
             "data/v2/validation-00001-of-00002.json.gz",
             "data/v2/validation-00000-of-00002.json.gz",
             "data/v3/train-0000-of-0136.zst"]
    assert pes2o.choose_shards(files) == ["data/v2/validation-00000-of-00002.json.gz",
                                          "data/v2/validation-00001-of-00002.json.gz"]
    no_valid = [f for f in files if "valid" not in f]
    assert pes2o.choose_shards(no_valid) == ["data/v2/train-00000-of-00020.json.gz",
                                             "data/v2/train-00001-of-00020.json.gz"]


def test_pes2o_download_stops_when_pool_is_full(tmp_path, monkeypatch):
    source = tmp_path / "src"
    shard = _write_pes2o(source, _pes2o_rows())
    shards = ["data/v2/validation-00000-of-00002.json.gz",
              "data/v2/validation-00001-of-00002.json.gz"]
    fetched = []

    def fake_download(repo_id, filename, local_dir):
        assert repo_id == "allenai/peS2o"
        fetched.append(filename)
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(shard, target)
        return target

    monkeypatch.setattr(pes2o, "_list_repo_files", lambda: shards + ["README.md"])
    monkeypatch.setattr(pes2o, "hf_download_file", fake_download)
    raw = tmp_path / "raw"
    records = list(pes2o.load(raw, download=True, max_texts=2))
    assert fetched == shards[:1]                       # 2 * POOL_FACTOR papers in shard 1
    assert len(records) == 2
    # the downloaded shard is reused without the network
    monkeypatch.setattr(pes2o, "_list_repo_files", lambda: pytest.fail("network"))
    again = list(pes2o.load(raw, download=True, max_texts=2))
    assert [r.id for r in again] == [r.id for r in records]


# ---------------------------------------------------------------- theses: pure

def test_cc_license_accepts_by_and_by_sa_only():
    url = "https://creativecommons.org/licenses/by/4.0/"
    assert theses.cc_license([url]) == url
    assert theses.cc_license(["http://creativecommons.org/licenses/by-sa/3.0/"]) == \
        "https://creativecommons.org/licenses/by-sa/3.0/"
    assert theses.cc_license(["Licence: CC BY 4.0"]) == url
    assert theses.cc_license(["CC-BY-SA"]) == "https://creativecommons.org/licenses/by-sa/4.0/"
    for bad in ("https://creativecommons.org/licenses/by-nc/4.0/",
                "https://creativecommons.org/licenses/by-nd/4.0/",
                "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "CC BY-NC-ND 4.0", "info:eu-repo/semantics/openAccess", "All rights reserved"):
        assert theses.cc_license([bad]) is None, bad
    # a record naming both BY and an NC licence is rejected
    assert theses.cc_license([url, "CC BY-NC 4.0"]) is None
    assert theses.has_nc_nd(["CC BY-ND"]) and not theses.has_nc_nd([url, "no licence"])


def test_student_thesis_and_language_checks():
    assert theses.is_student_thesis(["info:eu-repo/semantics/bachelorThesis"])
    assert theses.is_student_thesis(["Student thesis", "text"])
    assert theses.is_student_thesis(["Independent thesis Advanced level (degree of Master)"])
    assert not theses.is_student_thesis(["Doctoral thesis, monograph"])
    assert not theses.is_student_thesis(["Article in journal"])
    assert theses.is_english_language(["eng"]) and theses.is_english_language([])
    assert not theses.is_english_language(["swe"])


def test_pdf_candidates_and_landing_page():
    rec = OaiRecord("oai:DiVA.org:kth-123456", "2020-01-01", False, {
        "identifier": ["http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-123456",
                       "https://example.org/thesis.pdf"]})
    assert theses.pdf_candidates(rec, "diva") == [
        "https://example.org/thesis.pdf",
        "https://www.diva-portal.org/smash/get/diva2:123456/FULLTEXT01.pdf"]
    assert theses.landing_page(rec, "diva").startswith("http://urn.kb.se/")
    lup = OaiRecord("oai:lup.lub.lu.se:8901234", "", False, {
        "identifier": ["https://lup.lub.lu.se/student-papers/record/8901234"],
        "relation": ["https://lup.lub.lu.se/student-papers/record/8901234/file/8901235"]})
    assert theses.pdf_candidates(lup, "lup") == [
        "https://lup.lub.lu.se/student-papers/record/8901234/file/8901235"]
    assert theses.record_slug(lup) == "8901234"


def _wrap(paragraph: str, width: int = 90) -> list[str]:
    return textwrap.wrap(paragraph, width)


def thesis_pages(seed: int = 7, n_paragraphs: int = 24) -> list[list[str]]:
    """Lines per page of a fake thesis: front matter, body, references."""
    header = "Lund University, Department of Things"
    front = ["A Study of Feedback in Rural Schools", "Bachelor thesis, 15 credits",
             "Abstract", *_wrap(para(random.Random(1), 4)), "Table of contents",
             "1 Introduction 3", "2 Method 5", "References 12"]
    body = ["1 Introduction"]
    for i, p in enumerate(prose(seed, n_paragraphs)):
        if i == 3:
            body.append("2 Method")
        lines = _wrap(p)
        if i == 5:   # a hyphenated line break
            lines[0] = lines[0] + " inter-"
            lines[1] = "national " + lines[1]
        body.extend(lines)
    body += ["References", *REFERENCES.splitlines()]
    pages = [[header, *front, "i"]]
    for start in range(0, len(body), 45):
        pages.append([header, *body[start:start + 45], str(len(pages) + 1)])
    return pages


def test_clean_pdf_text_strips_front_matter_headers_and_references():
    pages = ["\n".join(lines) for lines in thesis_pages()]
    text = theses.clean_pdf_text(pages)
    assert text.startswith(prose(7, 1)[0][:40])
    assert "Abstract" not in text and "Table of contents" not in text
    assert "Lund University, Department" not in text
    assert "et al." not in text and "References" not in text
    assert "international" in text and "inter- national" not in text
    paragraphs = text.split("\n\n")
    assert len(paragraphs) >= 15 and "2 Method" in paragraphs
    assert not any(p.strip().isdigit() for p in paragraphs)


# ------------------------------------------------------------- theses: harvest

def _write_pdf(path: Path, pages: list[list[str]]) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        ops = ["BT", "/F1 9 Tf", "13 TL", "40 770 Td"]
        ops += [f"({esc(line)}) Tj T*" for line in lines]
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    with open(path, "wb") as fh:
        writer.write(fh)
    return path.read_bytes()


BY = "https://creativecommons.org/licenses/by/4.0/"


def _oai_record(ident: str, *, rights: str = BY, lang: str = "eng", date: str = "2015",
                type_: str = "info:eu-repo/semantics/bachelorThesis",
                pdf: str | None = None, page: str | None = None, deleted: bool = False) -> str:
    if deleted:
        return (f'<record><header status="deleted"><identifier>{ident}</identifier>'
                f'<datestamp>2020-01-01</datestamp></header></record>')
    ids = "".join(f"<dc:identifier>{u}</dc:identifier>" for u in (page, pdf) if u)
    rights_el = f"<dc:rights>{rights}</dc:rights>" if rights else ""
    return (
        f"<record><header><identifier>{ident}</identifier><datestamp>2020-01-01</datestamp>"
        f"</header><metadata><oai_dc:dc xmlns:oai_dc='http://www.openarchives.org/OAI/2.0/"
        f"oai_dc/' xmlns:dc='http://purl.org/dc/elements/1.1/'>"
        f"<dc:title>Thesis {ident}</dc:title><dc:language>{lang}</dc:language>"
        f"<dc:date>{date}</dc:date><dc:type>{type_}</dc:type>{rights_el}{ids}"
        f"</oai_dc:dc></metadata></record>")


def _oai_page(records: list[str], token: str | None) -> bytes:
    tok = f"<resumptionToken>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return ("<?xml version='1.0'?><OAI-PMH xmlns='http://www.openarchives.org/OAI/2.0/'>"
            f"<responseDate>2024-01-01</responseDate><ListRecords>{''.join(records)}{tok}"
            "</ListRecords></OAI-PMH>").encode()


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, url: str = "") -> None:
        self.content = body
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))}
        self.url = url

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def iter_content(self, chunk_size: int = 1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class FakeSession:
    """Serves OAI pages by (url, resumptionToken), PDFs and landing pages by url."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url, params=None, timeout=None, stream=False):
        key = url if not params else f"{url}?{params.get('resumptionToken') or ''}"
        self.calls.append(key)
        value = self.routes.get(key)
        if value is None:
            return FakeResponse(b"not found", 404, url)
        if isinstance(value, Exception):
            raise value
        return FakeResponse(value, 200, url)


def _harvest_routes(tmp_path: Path) -> dict:
    pdf = _write_pdf(tmp_path / "thesis.pdf", thesis_pages())
    pdf2 = _write_pdf(tmp_path / "thesis2.pdf", thesis_pages(seed=8))
    base = "https://repo.example.org/oai"
    landing = "https://repo.example.org/record/77"
    page1 = _oai_page([
        _oai_record("oai:repo:11", pdf="https://repo.example.org/11.pdf"),
        _oai_record("oai:repo:12", rights="https://creativecommons.org/licenses/by-nc/4.0/",
                    pdf="https://repo.example.org/12.pdf"),
        _oai_record("oai:repo:13", date="2023", pdf="https://repo.example.org/13.pdf"),
        _oai_record("oai:repo:14", lang="swe", pdf="https://repo.example.org/14.pdf"),
    ], "tok1")
    page2 = _oai_page([
        _oai_record("oai:repo:15", type_="Doctoral thesis", pdf="https://repo.example.org/15.pdf"),
        _oai_record("oai:repo:16", deleted=True),
        _oai_record("oai:repo:77", rights="", page=landing),   # licence + PDF on landing page
        _oai_record("oai:repo:18", rights="https://creativecommons.org/licenses/by-nd/4.0/",
                    pdf="https://repo.example.org/18.pdf"),
    ], None)
    html = (f'<html><a href="{BY}">CC BY</a> <a href="/files/77/FULLTEXT01.pdf">Full text</a>'
            "</html>").encode()
    return {
        f"{base}?": page1, f"{base}?tok1": page2,
        "https://repo.example.org/11.pdf": pdf,
        landing: html, "https://repo.example.org/files/77/FULLTEXT01.pdf": pdf2,
        "https://broken.example.org/oai?": ConnectionError("down"),
    }


ENDPOINTS = (Endpoint("broken", "https://broken.example.org/oai"),
             Endpoint("repo", "https://repo.example.org/oai"))


def test_theses_harvest_filters_caches_and_reloads_offline(tmp_path, caplog):
    session = FakeSession(_harvest_routes(tmp_path))
    raw = tmp_path / "raw"
    with caplog.at_level(logging.WARNING):
        records = list(theses.load(raw, download=True, endpoints=ENDPOINTS, delay=0,
                                   session=session))
    assert "broken" in caplog.text                      # failing endpoint only warns
    _check_records(records, "theses", "student_thesis")
    slugs = {r.id.split(":")[2] for r in records}
    assert slugs == {"11", "77"}
    assert all(r.id.startswith("theses:repo:") for r in records)
    per_thesis = {s: sum(r.id.split(":")[2] == s for r in records) for s in slugs}
    assert all(1 <= n <= theses.SECTIONS_PER_THESIS for n in per_thesis.values())
    # rejected records never had their PDFs fetched
    for n in (12, 13, 14, 15, 18):
        assert f"https://repo.example.org/{n}.pdf" not in session.calls
    # cache: one line per kept thesis, with licence and landing page
    cache = raw / "theses" / "theses.jsonl"
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert {row["slug"] for row in rows} == {"11", "77"}
    assert all(row["license"] == BY for row in rows)
    assert {row["landing_page"] for row in rows} == {"", "https://repo.example.org/record/77"}
    assert len(list((raw / "theses" / "pdf").glob("*.pdf"))) == 2
    # rerun: read from the cache, no network at all
    again = list(theses.load(raw, download=True, endpoints=ENDPOINTS, delay=0,
                             session=FakeSession({})))
    assert [r.to_dict() for r in again] == [r.to_dict() for r in records]
    # before_year applied to the cache too
    assert list(theses.load(raw, download=False, before_year=2015)) == []


def test_theses_cache_sections_are_deterministic_and_capped(tmp_path):
    directory = tmp_path / "theses"
    directory.mkdir()
    rows = []
    for i in range(4):
        text = "\n\n".join(prose(300 + i, 40))
        rows.append({"repo": "lup", "slug": f"r{i}", "year": 2012, "license": BY,
                     "landing_page": "", "pdf_url": "", "text": text})
    rows.append({**rows[0], "slug": "nc", "license": "https://creativecommons.org/licenses/by-nc/4.0/"})
    rows.append({**rows[0], "slug": "new", "year": 2023})
    (directory / "theses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    records = list(theses.load(tmp_path, download=False, max_texts=0))
    _check_records(records, "theses", "student_thesis")
    assert len(records) == 8                            # 2 sections x 4 theses
    assert {r.id.split(":")[2] for r in records} == {"r0", "r1", "r2", "r3"}
    a = [r.id for r in theses.load(tmp_path, download=False, max_texts=3)]
    b = [r.id for r in theses.load(tmp_path, download=False, max_texts=3)]
    assert a == b and len(a) == 3 and set(a) <= {r.id for r in records}
    other_seed = {r.id for r in theses.load(tmp_path, download=False, max_texts=0, seed=1)}
    assert len(other_seed) == 8 and other_seed != {r.id for r in records}


def test_theses_nothing_harvested_raises(tmp_path):
    session = FakeSession({"https://broken.example.org/oai?": ConnectionError("down")})
    with pytest.raises(RuntimeError, match="no thesis passed the filters"):
        list(theses.load(tmp_path, download=True, endpoints=ENDPOINTS[:1], delay=0,
                         session=session))
    assert not (tmp_path / "theses" / "theses.jsonl").exists()


def test_theses_rejected_set_is_retried_without_set(tmp_path):
    error = (b"<?xml version='1.0'?><OAI-PMH xmlns='http://www.openarchives.org/OAI/2.0/'>"
             b"<error code='badArgument'>bad set</error></OAI-PMH>")
    calls = []

    class SetSession(FakeSession):
        def get(self, url, params=None, timeout=None, stream=False):
            if params is not None:
                calls.append(dict(params))
                if "set" in params:
                    return FakeResponse(error)
            return super().get(url, params, timeout, stream)

    routes = _harvest_routes(tmp_path)
    endpoint = Endpoint("repo", "https://repo.example.org/oai", set_spec="nope")
    records = list(theses.load(tmp_path / "raw", download=True, endpoints=[endpoint], delay=0,
                               session=SetSession(routes)))
    assert records
    assert calls[0].get("set") == "nope" and "set" not in calls[1]


def test_parse_list_records_reads_fields_token_and_errors():
    xml = _oai_page([_oai_record("oai:x:1", pdf="https://x.org/1.pdf"),
                     _oai_record("oai:x:2", deleted=True)], "abc")
    records, token, error = theses.parse_list_records(xml)
    assert token == "abc" and error is None
    assert records[0].fields["rights"] == [BY]
    assert records[0].fields["identifier"] == ["https://x.org/1.pdf"]
    assert records[1].deleted
    no_match = (b"<OAI-PMH xmlns='http://www.openarchives.org/OAI/2.0/'>"
                b"<error code='noRecordsMatch'/></OAI-PMH>")
    assert theses.parse_list_records(no_match) == ([], None, "noRecordsMatch")
