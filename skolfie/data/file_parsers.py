"""
skolfie/data/file_parsers.py
-----------------------------
Utilities for extracting raw text from uploaded files.

Supports:
    - .txt   (plain text)
    - .docx  (Word documents, via python-docx)
    - .pdf   (PDF documents, via pypdf)

Every parser function returns a single string of extracted text.
"""

from __future__ import annotations

import codecs
import os

SUPPORTED_EXTENSIONS = (".txt", ".docx", ".pdf")


def read_txt(file_path: str) -> str:
    """
    Read a plain text file.

    A byte-order mark (UTF-8 / UTF-16) wins if present; otherwise UTF-8 is
    tried, then Windows-1252, then Latin-1 (which accepts any byte sequence,
    so decoding never fails outright).
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")

    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_docx(file_path: str) -> str:
    """Extract all paragraph (and table cell) text from a .docx file."""
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required to read .docx files. "
            "Install it with: pip install python-docx"
        ) from exc

    document = docx.Document(file_path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    # Reports and letters often store content in tables too.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)

    return "\n".join(paragraphs)


def read_pdf(file_path: str) -> str:
    """Extract text from a .pdf file, page by page."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to read .pdf files. "
            "Install it with: pip install pypdf"
        ) from exc

    text_chunks = []
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)

    return "\n".join(text_chunks)


def extract_text_from_file(file_path: str | os.PathLike[str]) -> str:
    """
    Dispatch to the right parser based on file extension.
    Raises a ValueError for unsupported extensions.
    """
    if not file_path:
        return ""

    file_path = os.fspath(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload a .txt, .docx, or .pdf file."
        )
