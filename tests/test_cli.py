"""
tests/test_cli.py
------------------
Roadmap step 11: the `skolfie` command (check / bundle / info) with the fake
models of tests/test_service.py.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from skolfie.cli import main
from skolfie.service import DISCLAIMER
from tests.test_service import AI_ESSAY, HUMAN_ESSAY, MIXED_ESSAY, make_colab_dirs


@pytest.fixture(autouse=True)
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SKOLFIE_BUNDLE", "SKOLFIE_MODELS_DIR", "SKOLFIE_REPORTS_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return make_colab_dirs(tmp_path_factory.mktemp("colab"))


@pytest.fixture()
def essays(tmp_path: Path) -> Path:
    """A folder of essays: nested, a Markdown file, files to skip and a broken PDF."""
    root = tmp_path / "essays"
    (root / "class-b").mkdir(parents=True)
    (root / ".hidden").mkdir()
    (root / "anna.txt").write_text(HUMAN_ESSAY, encoding="utf-8")
    (root / "class-b" / "ben.md").write_text(AI_ESSAY, encoding="utf-8")
    (root / "class-b" / "cleo.TXT").write_text(MIXED_ESSAY, encoding="utf-8")
    (root / "short.txt").write_text("Too short to judge.", encoding="utf-8")
    (root / "notes.csv").write_text("a,b\n1,2\n")
    (root / ".hidden" / "x.txt").write_text(AI_ESSAY)
    (root / "~$draft.docx").write_bytes(b"lock file")
    return root


def run(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_check_folder_table(dirs, essays) -> None:
    code, out, err = run(["check", str(essays), "--models-dir", str(dirs["models"]),
                          "--reports-dir", str(dirs["reports"])])
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0].split() == ["file", "words", "label", "AI", "likelihood", "warnings"]
    body = lines[2:6]
    assert [line.split()[0] for line in body] == [
        str(essays / "anna.txt"), str(essays / "class-b" / "ben.md"),
        str(essays / "class-b" / "cleo.TXT"), str(essays / "short.txt")]
    assert [line.split()[2] for line in body] == ["human", "ai", "mixed", "too_short"]
    assert body[1].split()[3].endswith("%")
    assert DISCLAIMER in out


def test_check_csv_with_error_row(dirs, essays, tmp_path: Path) -> None:
    (essays / "broken.pdf").write_bytes(b"this is not a pdf at all")
    out_file = tmp_path / "out" / "results.csv"
    code, out, err = run(["check", str(essays), str(tmp_path / "missing.txt"),
                          "--models-dir", str(dirs["models"]), "--format", "csv",
                          "--out", str(out_file)])
    assert code == 1 and out == ""
    assert "2 of 6 input(s) could not be checked" in err
    rows = list(csv.DictReader(out_file.open(encoding="utf-8")))
    by_name = {Path(r["file"]).name: r for r in rows}
    assert list(by_name) == ["anna.txt", "broken.pdf", "ben.md", "cleo.TXT", "short.txt",
                             "missing.txt"]
    assert by_name["broken.pdf"]["label"] == "error"
    assert by_name["broken.pdf"]["error"] == "This is not a valid .pdf file."
    assert by_name["missing.txt"]["error"] == "File not found."
    assert by_name["ben.md"]["label"] == "ai" and float(by_name["ben.md"]["ai_likelihood_pct"]) > 70
    assert "fairness" in by_name["ben.md"]["warning_kinds"]
    assert by_name["anna.txt"]["warnings"] == "0" and by_name["anna.txt"]["error"] == ""
    assert by_name["short.txt"]["label"] == "too_short"
    assert by_name["short.txt"]["ai_likelihood_pct"] == ""


def test_check_stdin_json_from_env(dirs, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKOLFIE_MODELS_DIR", str(dirs["models"]))
    monkeypatch.setenv("SKOLFIE_REPORTS_DIR", str(dirs["reports"]))
    code, out, err = run(["check", "-", "--format", "json"], stdin=AI_ESSAY)
    assert code == 0, err
    data = json.loads(out)
    assert data["disclaimer"] == DISCLAIMER
    [row] = data["results"]
    assert row["file"] == "<stdin>" and row["error"] is None
    assert row["result"]["label"] == "ai" and row["result"]["reasons"]


def test_check_html_reports(dirs, tmp_path: Path) -> None:
    evil = tmp_path / "evil.txt"
    evil.write_text("<script>alert(1)</script> " + AI_ESSAY, encoding="utf-8")
    html_dir = tmp_path / "html"
    code, _, err = run(["check", str(evil), "-", "--models-dir", str(dirs["models"]),
                        "--html", str(html_dir)], stdin=HUMAN_ESSAY)
    assert code == 0, err
    assert sorted(p.name for p in html_dir.iterdir()) == ["evil.txt.html", "stdin.html"]
    page = (html_dir / "evil.txt.html").read_text(encoding="utf-8")
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "Likely AI-generated" in page and "sole basis" in page
    assert "Essays by non-native English writers" in page  # a fairness warning
    assert 'class="chunk"' in page and "<h2>Why</h2>" in page


def test_bundle_and_info(dirs, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    code, out, err = run(["bundle", "--models-dir", str(dirs["models"]),
                          "--reports-dir", str(dirs["reports"]), "--out", str(bundle)])
    assert code == 0, err
    assert (bundle / "bundle.json").is_file() and "Bundle written" in out
    code, out, err = run(["info", "--bundle", str(bundle)])
    assert code == 0, err
    info = json.loads(out)
    assert info["source"] == "bundle" and info["cutoffs"]["ai_threshold"] == 0.7
    code, out, _ = run(["check", "-", "--bundle", str(bundle), "--format", "csv"],
                       stdin=HUMAN_ESSAY)
    assert code == 0 and ",human," in out


def test_usage_errors(dirs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text(HUMAN_ESSAY)
    code, _, err = run(["check", str(txt)])
    assert code == 2 and "no models" in err
    code, _, err = run(["check", str(txt), "--bundle", str(tmp_path),
                        "--models-dir", str(dirs["models"])])
    assert code == 2 and "not both" in err
    code, _, err = run(["check", str(txt), "--models-dir", str(tmp_path / "nope")])
    assert code == 2 and "models folder not found" in err
    code, _, err = run(["check", str(txt), "--bundle", str(tmp_path)])
    assert code == 2 and "bundle.json" in err
    (tmp_path / "empty").mkdir()
    code, _, err = run(["check", str(tmp_path / "empty"), "--models-dir", str(dirs["models"])])
    assert code == 2 and "no files to check" in err
    code, _, err = run(["check", "-", "-", "--models-dir", str(dirs["models"])])
    assert code == 2
    assert run(["check", str(txt), "--format", "xml"])[0] == 2
    assert run([])[0] == 2
    assert run(["bundle", "--models-dir", str(tmp_path / "nope"),
                "--out", str(tmp_path / "b")])[0] == 2
    monkeypatch.setenv("SKOLFIE_BUNDLE", str(tmp_path / "nope"))
    assert run(["info"])[0] == 2


def test_too_long_text_is_an_error_row(dirs, tmp_path: Path) -> None:
    from skolfie.service import MAX_TEXT_CHARS

    big = tmp_path / "big.txt"
    big.write_text("word " * (MAX_TEXT_CHARS // 5 + 10))
    ok = tmp_path / "ok.txt"
    ok.write_text(HUMAN_ESSAY)
    code, out, _ = run(["check", str(big), str(ok), "--models-dir", str(dirs["models"]),
                        "--format", "json"])
    assert code == 1
    rows = json.loads(out)["results"]
    assert "too long" in rows[0]["error"] and rows[0]["result"] is None
    assert rows[1]["result"]["label"] == "human"
