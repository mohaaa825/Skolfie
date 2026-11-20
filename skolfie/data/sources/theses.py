"""
skolfie/data/sources/theses.py
------------------------------
Loader for open-access student theses (bachelor and master level) under
Creative Commons BY or BY-SA licences, harvested over OAI-PMH from university
repositories. Like peS2o, this source supplies long human academic prose
(800-1,500 word sections, before ChatGPT) so the detector cannot learn
"long text = AI"; unlike peS2o it is written by students, close to the essays
the detector is meant for.

Endpoints (`DEFAULT_ENDPOINTS`; each is an `Endpoint(repo, base_url, set_spec,
all_student_theses)`):

  lup   https://lup.lub.lu.se/student-papers/search/oai
        Lund University Student Papers (LUP). Verified from the repository's
        API page (lup.lub.lu.se/student-papers/search/doc/api): OAI-PMH 2.0,
        `metadataPrefix=oai_dc`, resumption tokens; every record is a student
        paper, so the type check is skipped. Its JSON schema has file entries
        with an `access` level and a free-text `copyright_statement` (no
        licence field); how files and licences appear in oai_dc was NOT
        verified (the harvesting sandbox had no network) and is handled
        heuristically (below).
  diva  https://www.diva-portal.org/dice/oai, set "publicationTypeCode:studentThesis"
        DiVA, the portal of about 50 Swedish universities. The `/dice/oai`
        base URL is verified from registry entries (e.g. Uppsala's
        http://uu.diva-portal.org/dice/oai in the COAR repository registry).
        The set name is an ASSUMPTION; if the endpoint rejects it
        (badArgument / noSetHierarchy / noRecordsMatch) the harvest retries
        once without a set and relies on the dc:type filter.

Harvest (`download=True` and no cache): plain `requests` (`pip install -e
".[generate]"`), `verb=ListRecords&metadataPrefix=oai_dc`, following
`resumptionToken`, with `delay` seconds between requests, a timeout, a retry
on HTTP 503/429 (honouring Retry-After up to a minute) and at most
`max_records` records read per endpoint. Any endpoint failure is logged as a
warning and the next endpoint is tried.

Record filters (`record_filter_reason`; all on oai_dc fields):
  * not deleted;
  * a licence: some field mentions creativecommons.org/licenses/by/ or
    /by-sa/ (or "CC BY" / "CC BY-SA" text); NC and ND licences are rejected
    (`cc_license`). When the metadata has no licence at all and
    `check_landing_page` is on, the landing page HTML is searched for a CC
    licence link instead;
  * language English (dc:language en/eng/English, or missing; the extracted
    text must also pass `looks_english`);
  * year (first year in dc:date) in [MIN_YEAR, before_year);
  * dc:type looks like a student thesis (bachelor/master/student
    thesis/degree project/essay/uppsats/examensarbete), never doctoral or
    licentiate, unless the endpoint holds only student theses.

PDF links (`pdf_candidates`): http(s) values of dc:identifier, dc:relation,
dc:source and dc:format that end in .pdf or match a known full-text pattern
(DiVA `/smash/get/diva2:N/FULLTEXT01.pdf`, LUP and other LibreCat/Biblio
`/record/N/file/M`, DSpace `/bitstream/`), in that order; for DiVA records
`https://www.diva-portal.org/smash/get/diva2:N/FULLTEXT01.pdf` is derived from
the `diva-N` / `diva2:N` number in the identifiers; finally, the landing page
(the first non-PDF http identifier, or the DiVA record page) is fetched and
its links matching the same patterns are tried. PDFs are cached as
`raw_dir/theses/pdf/<sha1 of url>.pdf` and capped at `MAX_PDF_BYTES`; a
response that does not start with %PDF is rejected.

Text (`extract_pdf_text` with pypdf, then `clean_pdf_text`): repeated
header/footer lines and page numbers are dropped, words hyphenated across
line breaks are rejoined, lines are joined into paragraphs (a paragraph ends
at a blank line or a short line ending a sentence), everything before the
Introduction heading is dropped when one is found (front matter, abstract,
table of contents), and everything from the reference list or appendix
heading in the second half is cut.

Cache: every thesis that passes the filters and yields text is written to
`raw_dir/theses/theses.jsonl` (repo, oai identifier, slug, title, year,
language, licence URL, landing page, PDF URL, cleaned text). This file is
also the licence manifest. If it exists it is read without any network
access; delete it to harvest again.

Sections: `pes2o.make_sections` over the cleaned paragraphs, English only, at
most `SECTIONS_PER_THESIS` (2) per thesis, chosen with an RNG seeded by (seed,
repo, slug). If more than `max_texts` sections result, `max_texts` are
sampled with `random.Random(seed)`.

Records: label/generator "human", domain "academic", subdomain
"student_thesis", source "theses", id `theses:<repo>:<slug>:s<k>`, k being
the section's index among the thesis's candidate sections.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from skolfie.data.schema import Record
from skolfie.data.sources._common import iter_jsonl, missing_files_error
from skolfie.data.sources.pes2o import looks_english, make_sections, year_of

log = logging.getLogger(__name__)

NAME = "theses"
DESCRIPTION = (
    "Open-access CC BY / CC BY-SA bachelor and master theses (Lund University Student Papers, "
    "DiVA) harvested over OAI-PMH; up to two 800-1,500 word sections per thesis (human only)."
)
CITATION = (
    "Student theses from Lund University Student Papers (https://lup.lub.lu.se/student-papers) "
    "and DiVA (https://www.diva-portal.org); each thesis is cited by its landing page in "
    "theses.jsonl."
)
LICENSE_NOTE = (
    "Per thesis: only CC BY or CC BY-SA (any version) is kept; the licence URL and landing page "
    "of every thesis are recorded in <raw_dir>/theses/theses.jsonl."
)


@dataclass(frozen=True)
class Endpoint:
    repo: str
    base_url: str
    set_spec: str | None = None
    #: every record of this endpoint/set is a student thesis (skip the dc:type check)
    all_student_theses: bool = False


DEFAULT_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("lup", "https://lup.lub.lu.se/student-papers/search/oai", None, True),
    Endpoint("diva", "https://www.diva-portal.org/dice/oai", "publicationTypeCode:studentThesis"),
)
URLS = tuple(e.base_url for e in DEFAULT_ENDPOINTS)

MANUAL_DOWNLOAD = f"""\
Theses are harvested over OAI-PMH ({", ".join(URLS)}) with `requests`
(pip install -e ".[generate]") when download is allowed; the result is cached in
<raw_dir>/{NAME}/theses.jsonl (one thesis per line with keys repo, slug, year,
license, landing_page, pdf_url, text). Copy such a file there to load offline.
"""

CACHE_NAME = "theses.jsonl"
DEFAULT_MAX_TEXTS = 2000
DEFAULT_MAX_RECORDS = 20000
SECTIONS_PER_THESIS = 2
MIN_YEAR = 2005
MAX_PDF_BYTES = 15 * 1024 * 1024
USER_AGENT = "skolfie-corpus-harvester/0.1 (research; OAI-PMH)"

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_DC = "{http://purl.org/dc/elements/1.1/}"

# --------------------------------------------------------------------------- #
# OAI-PMH parsing and record filters
# --------------------------------------------------------------------------- #


@dataclass
class OaiRecord:
    identifier: str
    datestamp: str
    deleted: bool
    fields: dict[str, list[str]]


def parse_list_records(xml: bytes) -> tuple[list[OaiRecord], str | None, str | None]:
    """(records, resumption token or None, OAI error code or None) of one response."""
    root = ET.fromstring(xml)
    error = root.find(f"{_OAI}error")
    if error is not None:
        return [], None, error.get("code") or "unknown"
    records: list[OaiRecord] = []
    list_records = root.find(f"{_OAI}ListRecords")
    if list_records is None:
        return [], None, None
    for rec in list_records.findall(f"{_OAI}record"):
        header = rec.find(f"{_OAI}header")
        if header is None:
            continue
        fields: dict[str, list[str]] = {}
        metadata = rec.find(f"{_OAI}metadata")
        if metadata is not None:
            for el in metadata.iter():
                if el.tag.startswith(_DC) and el.text and el.text.strip():
                    fields.setdefault(el.tag[len(_DC):], []).append(el.text.strip())
        records.append(OaiRecord(
            identifier=(header.findtext(f"{_OAI}identifier") or "").strip(),
            datestamp=(header.findtext(f"{_OAI}datestamp") or "").strip(),
            deleted=header.get("status") == "deleted",
            fields=fields,
        ))
    token_el = list_records.find(f"{_OAI}resumptionToken")
    token = (token_el.text or "").strip() if token_el is not None else ""
    return records, token or None, None


_CC_URL = re.compile(r"creativecommons\.org/licen[cs]es/([a-z-]+)(?:/(\d(?:\.\d)?))?", re.I)
_CC_TEXT = re.compile(r"\bCC[\s-]+(BY(?:[\s-]+(?:NC|ND|SA))*)(?:[\s-]+(\d\.\d))?\b", re.I)


def cc_license(values: Iterable[str]) -> str | None:
    """Normalised licence URL if the values name CC BY or CC BY-SA (never NC/ND), else None.

    If any value names an NC or ND licence the record is rejected outright.
    """
    found: list[tuple[str, str]] = []
    for value in values:
        for kind, version in _CC_URL.findall(value):
            found.append((kind.lower(), version))
        for kind, version in _CC_TEXT.findall(value):
            found.append((re.sub(r"[\s-]+", "-", kind.lower()), version))
    if not found or any("nc" in k.split("-") or "nd" in k.split("-") for k, _ in found):
        return None
    for kind, version in found:
        if kind in ("by", "by-sa"):
            return f"https://creativecommons.org/licenses/{kind}/{version or '4.0'}/"
    return None


_ENGLISH = {"en", "eng", "english", "en-us", "en-gb", "en_us", "en_gb"}


def is_english_language(values: Sequence[str]) -> bool:
    """True if dc:language says English or is missing."""
    if not values:
        return True
    return any(v.strip().lower() in _ENGLISH for v in values)


_STUDENT = re.compile(
    r"bachelor|master|student|degree project|independent thesis|undergraduate|essay|"
    r"uppsats|examensarbete|magister|kandidat|candidate", re.I)
_NOT_STUDENT = re.compile(r"doctoral|licentiate|dissertation|phd", re.I)


def is_student_thesis(types: Sequence[str]) -> bool:
    text = " | ".join(types)
    return bool(_STUDENT.search(text)) and not _NOT_STUDENT.search(text)


def record_year(fields: dict[str, list[str]]) -> int | None:
    for value in fields.get("date", []):
        year = year_of(value)
        if year is not None:
            return year
    return None


def record_filter_reason(rec: OaiRecord, endpoint: Endpoint, before_year: int | None,
                         require_license: bool = True) -> str | None:
    """Why the record is rejected, or None if it passes the metadata filters."""
    if rec.deleted:
        return "deleted"
    if not is_english_language(rec.fields.get("language", [])):
        return "language"
    year = record_year(rec.fields)
    if year is None or year < MIN_YEAR or (before_year is not None and year >= before_year):
        return "year"
    if not endpoint.all_student_theses and not is_student_thesis(rec.fields.get("type", [])):
        return "type"
    if require_license and not cc_license(v for vs in rec.fields.values() for v in vs):
        return "license"
    return None


def has_nc_nd(values: Iterable[str]) -> bool:
    """True if the values name some CC licence that is not BY/BY-SA (e.g. NC, ND)."""
    values = list(values)
    mentions = any(_CC_URL.search(v) or _CC_TEXT.search(v) for v in values)
    return mentions and cc_license(values) is None


_PDF_PATTERNS = (
    re.compile(r"\.pdf(?:$|[?#])", re.I),
    re.compile(r"/smash/get/diva2:\d+/FULLTEXT\d+", re.I),
    re.compile(r"/record/\d+/file/\d+", re.I),
    re.compile(r"/bitstream/", re.I),
)
_DIVA_NUMBER = re.compile(r"diva(?:2:|-)(\d+)", re.I)


def _is_http(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def looks_like_pdf_url(url: str) -> bool:
    return any(p.search(url) for p in _PDF_PATTERNS)


def pdf_candidates(rec: OaiRecord, repo: str) -> list[str]:
    """Full-text PDF URLs to try, best first (see the module docstring)."""
    values = [v for key in ("identifier", "relation", "source", "format")
              for v in rec.fields.get(key, []) if _is_http(v)]
    out = [v for v in values if _PDF_PATTERNS[0].search(v)]
    out += [v for v in values if looks_like_pdf_url(v) and v not in out]
    if repo == "diva":
        number = diva_number(rec)
        if number:
            url = f"https://www.diva-portal.org/smash/get/diva2:{number}/FULLTEXT01.pdf"
            if url not in out:
                out.append(url)
    return out


def diva_number(rec: OaiRecord) -> str | None:
    for value in [rec.identifier, *rec.fields.get("identifier", [])]:
        match = _DIVA_NUMBER.search(value)
        if match:
            return match.group(1)
    return None


def landing_page(rec: OaiRecord, repo: str) -> str | None:
    for value in rec.fields.get("identifier", []):
        if _is_http(value) and not looks_like_pdf_url(value):
            return value
    if repo == "diva":
        number = diva_number(rec)
        if number:
            return f"https://www.diva-portal.org/smash/record.jsf?pid=diva2:{number}"
    return None


def record_slug(rec: OaiRecord) -> str:
    local = rec.identifier.rsplit(":", 1)[-1] if rec.identifier else ""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", local).strip("-").lower()
    if not slug:
        slug = hashlib.sha1(json.dumps(rec.fields, sort_keys=True).encode()).hexdigest()[:12]
    return slug


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in ("href", "content", "src") and value:
                self.links.append(value)


def page_links(html: str, base_url: str) -> list[str]:
    """Absolute URLs of the links (href/content/src) of an HTML page."""
    parser = _Links()
    try:
        parser.feed(html)
    except Exception:  # pragma: no cover - HTMLParser is lenient
        pass
    return [urljoin(base_url, link) for link in parser.links]


# --------------------------------------------------------------------------- #
# PDF text
# --------------------------------------------------------------------------- #


def extract_pdf_text(path: Path) -> list[str]:
    """The text of each page (pypdf); pages that fail to extract are empty."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pypdf can fail on odd fonts; skip the page
            pages.append("")
    return pages


_PAGE_NUMBER = re.compile(
    r"^\W*(?:page\s*)?(?:\d{1,4}|[ivxlc]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?\W*$", re.I)
_INTRO = re.compile(
    r"^(?:(?:chapter\s+)?(?:\d{1,2}(?:\.\d{1,2})*|[ivx]{1,4})\.?\s+)?introduction\.?$", re.I)
_END_MATTER = re.compile(
    r"^(?:(?:\d{1,2}|[ivx]{1,4})\.?\s+)?(?:references|bibliography|reference list|"
    r"list of references|works cited|literature cited|literature|appendix|"
    r"appendices)(?:\s+[a-z0-9]{1,3})?\.?$", re.I)
_SENT_END = (".", "?", "!", ":", '"', "”")


def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.split())).lower()


def _drop_headers_footers(pages: list[list[str]]) -> list[list[str]]:
    n_pages = len(pages)
    counts: dict[str, int] = {}
    for lines in pages:
        edge = {_norm_line(ln) for ln in lines[:2] + lines[-2:]}
        for key in edge:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(2, math.ceil(0.3 * n_pages))
    repeated = {k for k, c in counts.items() if c >= threshold and len(k) < 120}
    out = []
    for lines in pages:
        kept = []
        for i, line in enumerate(lines):
            edge = i < 2 or i >= len(lines) - 2
            if edge and _norm_line(line) in repeated:
                continue
            if _PAGE_NUMBER.match(line.strip()):
                continue
            kept.append(line)
        out.append(kept)
    return out


def _join_lines(lines: list[str]) -> list[str]:
    """Join PDF lines into paragraphs (blank line, or a short line ending a sentence)."""
    lengths = sorted(len(ln) for ln in lines if ln.strip())
    typical = lengths[int(len(lengths) * 0.75)] if lengths else 0
    paragraphs: list[str] = []
    current = ""
    for raw in lines:
        line = " ".join(raw.split())
        if not line:
            if current:
                paragraphs.append(current)
            current = ""
            continue
        heading = (len(line.split()) < 10 and not line.endswith(_SENT_END)
                   and not line[:1].islower())
        if heading and current.endswith(_SENT_END):
            paragraphs.append(current)
            current = ""
        if current.endswith("-") and len(current) > 1 and current[-2].isalpha() \
                and line[:1].islower():
            current = current[:-1] + line
        else:
            current = f"{current} {line}" if current else line
        short = len(raw.strip()) < 0.8 * typical
        if short and (line.endswith(_SENT_END) or len(line.split()) < 10):
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


def clean_pdf_text(pages: list[str]) -> str:
    """Body text of a thesis, paragraphs separated by blank lines (see the module docstring)."""
    split = [[ln.rstrip() for ln in page.splitlines()] for page in pages]
    split = [[ln for ln in lines if ln.strip()] for lines in split]
    lines = [ln for page in _drop_headers_footers(split) for ln in page]
    stripped = [" ".join(ln.split()) for ln in lines]
    n = len(stripped)
    intros = [i for i, ln in enumerate(stripped) if _INTRO.match(ln) and i < n / 2]
    start = intros[-1] + 1 if intros else 0
    ends = [i for i, ln in enumerate(stripped) if _END_MATTER.match(ln) and i > max(start, n / 2)]
    end = ends[0] if ends else n
    paragraphs = _join_lines(lines[start:end])
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- #
# Harvest
# --------------------------------------------------------------------------- #


def _requests_session() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "requests is needed to harvest theses; install it (pip install -e \".[generate]\") "
            f"or place {CACHE_NAME} manually.\n{MANUAL_DOWNLOAD}") from exc
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


class _Harvester:
    def __init__(self, session: Any, directory: Path, delay: float, timeout: float,
                 check_landing_page: bool) -> None:
        self.session = session
        self.directory = directory
        self.delay = delay
        self.timeout = timeout
        self.check_landing_page = check_landing_page
        self.stats: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def _get(self, url: str, **kwargs: Any) -> Any:
        for attempt in range(3):
            if self.delay:
                time.sleep(self.delay)
            resp = self.session.get(url, timeout=self.timeout, **kwargs)
            if resp.status_code in (429, 503) and attempt < 2:
                wait = resp.headers.get("Retry-After", "10")
                time.sleep(min(60.0, float(wait) if str(wait).isdigit() else 10.0))
                continue
            return resp
        return resp  # pragma: no cover

    def list_records(self, endpoint: Endpoint, max_records: int) -> Iterator[OaiRecord]:
        params: dict[str, str] = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        if endpoint.set_spec:
            params["set"] = endpoint.set_spec
        seen = 0
        first = True
        while True:
            resp = self._get(endpoint.base_url, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"{endpoint.base_url}: HTTP {resp.status_code}")
            records, token, error = parse_list_records(resp.content)
            if error:
                if first and "set" in params and error in ("badArgument", "noSetHierarchy",
                                                           "noRecordsMatch"):
                    log.warning("theses: %s rejected set %r (%s); harvesting without a set",
                                endpoint.base_url, params["set"], error)
                    params.pop("set")
                    continue
                if error != "noRecordsMatch":
                    raise RuntimeError(f"{endpoint.base_url}: OAI-PMH error {error}")
                return
            first = False
            for rec in records:
                yield rec
                seen += 1
                if seen >= max_records:
                    return
            if not token:
                return
            params = {"verb": "ListRecords", "resumptionToken": token}

    def fetch_page(self, url: str) -> tuple[str, str]:
        """(final url, html) of a landing page, or ('', '') on failure."""
        try:
            resp = self._get(url)
        except Exception as exc:
            log.debug("theses: landing page %s failed: %s", url, exc)
            return "", ""
        if resp.status_code != 200:
            return "", ""
        return str(getattr(resp, "url", "") or url), resp.text

    def fetch_pdf(self, url: str) -> Path | None:
        pdf_dir = self.directory / "pdf"
        target = pdf_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.pdf"
        if target.is_file():
            return target
        try:
            resp = self._get(url, stream=True)
            if resp.status_code != 200:
                return None
            size = int(resp.headers.get("Content-Length") or 0)
            if size > MAX_PDF_BYTES:
                return None
            data = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    return None
        except Exception as exc:
            log.debug("theses: PDF %s failed: %s", url, exc)
            return None
        if not bytes(data[:1024]).lstrip().startswith(b"%PDF"):
            return None
        pdf_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(bytes(data))
        tmp.replace(target)
        return target

    def thesis(self, rec: OaiRecord, endpoint: Endpoint, before_year: int | None,
               min_words: int) -> dict | None:
        reason = record_filter_reason(rec, endpoint, before_year, require_license=False)
        if reason:
            self._count(f"rejected_{reason}")
            return None
        all_values = [v for vs in rec.fields.values() for v in vs]
        if has_nc_nd(all_values):
            self._count("rejected_license")
            return None
        license_url = cc_license(all_values)
        landing = landing_page(rec, endpoint.repo)
        page_html, page_url = "", ""
        if not license_url and self.check_landing_page and landing:
            page_url, page_html = self.fetch_page(landing)
            links = page_links(page_html, page_url) if page_html else []
            if has_nc_nd(links):
                self._count("rejected_license")
                return None
            license_url = cc_license(links)
        if not license_url:
            self._count("rejected_license")
            return None
        candidates = pdf_candidates(rec, endpoint.repo)
        if landing and not candidates:
            if not page_html:
                page_url, page_html = self.fetch_page(landing)
            if page_html:
                candidates = [u for u in page_links(page_html, page_url)
                              if looks_like_pdf_url(u)]
        text, pdf_url = "", None
        for url in dict.fromkeys(candidates):
            path = self.fetch_pdf(url)
            if path is None:
                continue
            try:
                text = clean_pdf_text(extract_pdf_text(path))
            except Exception as exc:
                log.debug("theses: %s: text extraction failed: %s", url, exc)
                continue
            pdf_url = url
            break
        if not pdf_url:
            self._count("no_pdf")
            return None
        if len(text.split()) < min_words or not looks_english(text):
            self._count("rejected_text")
            return None
        self._count("kept")
        titles = rec.fields.get("title", [])
        return {
            "repo": endpoint.repo, "oai_identifier": rec.identifier, "slug": record_slug(rec),
            "title": titles[0] if titles else "", "year": record_year(rec.fields),
            "language": (rec.fields.get("language") or [""])[0], "license": license_url,
            "landing_page": landing or "", "pdf_url": pdf_url, "text": text,
        }


def harvest(directory: Path, endpoints: Sequence[Endpoint], max_theses: int, max_records: int,
            before_year: int | None, min_words: int, delay: float = 1.0,
            timeout: float = 60.0, check_landing_page: bool = True,
            session: Any = None) -> list[dict]:
    """Harvest theses from `endpoints` (see the module docstring); failures only warn."""
    harvester = _Harvester(session or _requests_session(), directory, delay, timeout,
                           check_landing_page)
    rows: list[dict] = []
    for endpoint in endpoints:
        before = len(rows)
        try:
            for rec in harvester.list_records(endpoint, max_records):
                if len(rows) >= max_theses:
                    break
                row = harvester.thesis(rec, endpoint, before_year, min_words)
                if row is not None:
                    rows.append(row)
        except Exception as exc:
            log.warning("theses: endpoint %s (%s) failed: %s: %s", endpoint.repo,
                        endpoint.base_url, type(exc).__name__, exc)
        log.info("theses: %s: %d theses kept", endpoint.repo, len(rows) - before)
    log.info("theses: harvest stats %s", dict(sorted(harvester.stats.items())))
    return rows


def _write_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def thesis_sections(row: dict, seed: int, min_words: int,
                    max_words: int) -> list[tuple[int, str]]:
    """Up to SECTIONS_PER_THESIS (index, text) sections of one cached thesis, in order."""
    paragraphs = [p for p in str(row.get("text") or "").split("\n\n") if p.strip()]
    sections = [s for s in make_sections(paragraphs, min_words, max_words) if looks_english(s)]
    if not sections:
        return []
    rng = random.Random(f"{seed}:{row.get('repo')}:{row.get('slug')}")
    picked = rng.sample(range(len(sections)), min(SECTIONS_PER_THESIS, len(sections)))
    return [(k, sections[k]) for k in sorted(picked)]


def load(raw_dir: Path, download: bool = True, max_texts: int | None = DEFAULT_MAX_TEXTS,
         seed: int = 0, min_words: int = 800, max_words: int = 1500,
         before_year: int | None = 2022, max_records: int = DEFAULT_MAX_RECORDS,
         endpoints: Sequence[Endpoint] = DEFAULT_ENDPOINTS, delay: float = 1.0,
         timeout: float = 60.0, check_landing_page: bool = True,
         session: Any = None) -> Iterator[Record]:
    """Yield human student-thesis sections (see the module docstring)."""
    if max_texts is not None and max_texts < 0:
        raise ValueError(f"max_texts must be >= 0 (0 keeps all), got {max_texts}")
    if min_words < 1 or max_words < min_words:
        raise ValueError(f"need 1 <= min_words <= max_words, got {min_words}, {max_words}")
    directory = Path(raw_dir) / NAME
    cache = directory / CACHE_NAME
    if cache.is_file():
        rows = [row for _, row in iter_jsonl(cache)]
    elif download:
        max_theses = (max_texts + 1) // 2 + max_texts // 10 + 1 if max_texts else 10**9
        rows = harvest(directory, endpoints, max_theses, max_records, before_year, min_words,
                       delay=delay, timeout=timeout, check_landing_page=check_landing_page,
                       session=session)
        if not rows:
            raise RuntimeError(
                "theses: no thesis passed the filters (CC BY/BY-SA, English, student thesis, "
                f"{MIN_YEAR} <= year < {before_year}, PDF with >= {min_words} words) at "
                f"{', '.join(e.base_url for e in endpoints)}; see the warnings above "
                f"(network, OAI-PMH errors) or raise max_records.\n{MANUAL_DOWNLOAD}")
        _write_cache(cache, rows)
        log.info("theses: wrote %d theses to %s", len(rows), cache)
    else:
        raise missing_files_error(NAME, directory, CACHE_NAME, MANUAL_DOWNLOAD)

    records: dict[str, Record] = {}
    for row in rows:
        year = row.get("year")
        if before_year is not None and isinstance(year, int) and year >= before_year:
            continue
        if not cc_license([str(row.get("license") or "")]):
            continue
        repo = re.sub(r"[^a-z0-9]+", "-", str(row.get("repo") or "unknown").lower()) or "unknown"
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        for k, text in thesis_sections(row, seed, min_words, max_words):
            rid = f"{NAME}:{repo}:{slug}:s{k}"
            records[rid] = Record(id=rid, text=text, label="human", generator="human",
                                  domain="academic", subdomain="student_thesis", source=NAME)
    ids = sorted(records)
    if max_texts and len(ids) > max_texts:
        ids = sorted(random.Random(seed).sample(ids, max_texts))
    if not ids:
        log.warning("theses: %s holds no thesis with a usable %d-%d word section", cache,
                    min_words, max_words)
    for rid in ids:
        yield records[rid]
