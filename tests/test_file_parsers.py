"""
tests/test_file_parsers.py
---------------------------
Text extraction from .txt / .docx / .pdf uploads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skolfie.data.file_parsers import extract_text_from_file, read_txt

SAMPLE = "Café déjà vu: naïve résumé."


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]
)
def test_read_txt_encodings(tmp_path: Path, encoding: str) -> None:
    path = tmp_path / f"sample_{encoding}.txt"
    path.write_bytes(SAMPLE.encode(encoding))
    assert read_txt(str(path)) == SAMPLE


def test_read_txt_cp1252_smart_quotes(tmp_path: Path) -> None:
    text = "“Quoted” — and an apostrophe’s"
    path = tmp_path / "smart.txt"
    path.write_bytes(text.encode("cp1252"))
    assert extract_text_from_file(str(path)) == text


def test_extract_txt_accepts_pathlike(tmp_path: Path) -> None:
    path = tmp_path / "plain.TXT"
    path.write_text("hello world", encoding="utf-8")
    assert extract_text_from_file(path) == "hello world"


def test_extract_docx(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("   ")
    document.add_paragraph("Second paragraph.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Cell A"
    table.rows[0].cells[1].text = "Cell B"
    path = tmp_path / "doc.docx"
    document.save(str(path))

    text = extract_text_from_file(str(path))
    assert text.splitlines() == ["First paragraph.", "Second paragraph.", "Cell A", "Cell B"]


def _minimal_pdf(text: str) -> bytes:
    """Build a one-page PDF with a single line of Helvetica text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_extract_pdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    path = tmp_path / "doc.pdf"
    path.write_bytes(_minimal_pdf("Hello from a PDF"))
    assert "Hello from a PDF" in extract_text_from_file(str(path))


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.rtf"
    path.write_text("whatever")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_file(str(path))


def test_empty_path_returns_empty_string() -> None:
    assert extract_text_from_file("") == ""
