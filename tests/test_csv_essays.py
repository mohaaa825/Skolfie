"""Tests for scripts/import_csv_essays.py (CSV of AI essays -> fresh JSONL)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from skolfie.eval import splits as splits_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = ["Essay Number", "Model Used", "Grade Level", "Essay Type", "Target Length", "Prompt",
          "Generated Essay"]
BODY = ("So, last week in biology we did a lab on enzymes and temperature, and honestly it was "
        "a lot. We tested catalase at four temperatures and watched it fizz. ") * 3


def _script():
    name = "import_csv_essays_under_test"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" /
                                                  "import_csv_essays.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row(n: int, model: str, topic: str, text: str, kind: str = "book review") -> list:
    return [n, model, "10th grader", kind, "around 450 words",
            f"Write a {kind} about {topic}. Write like a 10th grader. Make it sound less like AI.",
            text]


def _csv(tmp_path: Path, rows: list) -> Path:
    path = tmp_path / "essays.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def test_strip_markdown_keeps_words_and_prices() -> None:
    imp = _script()
    text = ("# Lab Report\n**Hypothesis:** it *will* fizz  \n* tube one\n"
            "We used $M_1V_1 = M_2V_2$ and $0.1\\text{ M}$ NaOH, so $x$ was found. "
            "A $10 shirt and a $20 hat cost less than $\\text{CO}_2$ damage.")
    out = imp.strip_markdown(text)
    assert out.startswith("Lab Report\nHypothesis: it will fizz\n- tube one\n")
    assert "M1V1 = M2V2" in out and "0.1 M NaOH" in out and "so x was found" in out
    assert "A $10 shirt and a $20 hat" in out and "CO2 damage" in out
    assert "*" not in out and "\\" not in out


def test_import_csv_writes_records_grouped_by_topic_and_grade(tmp_path: Path, capsys) -> None:
    imp = _script()
    rows = [_row(1, "models/gemini-3.1-flash-lite", "the water cycle", "**Title:** x\n\n" + BODY),
            _row(2, "models/gemini-3.1-flash-lite", "the water cycle", BODY,
                 kind="personal narrative"),
            _row(3, "models/gemini-3.5-flash", "Justice in To Kill a Mockingbird", BODY),
            _row(4, "models/gemini-3.5-flash", "the water cycle", "Too short to keep.")]
    out = tmp_path / "fresh"
    assert imp.main([str(_csv(tmp_path, rows)), "--out-dir", str(out)]) == 0
    assert "too_short" in capsys.readouterr().out
    lite = [json.loads(line) for line in
            (out / "gemini-3.1-flash-lite.jsonl").read_text(encoding="utf-8").splitlines()]
    flash = [json.loads(line) for line in
             (out / "gemini-3.5-flash.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lite) == 2 and len(flash) == 1
    assert all(r["label"] == "ai" and r["source"] == "fresh" for r in lite + flash)
    assert "**" not in lite[0]["text"]
    assert lite[0]["id"] == ("fresh:gemini-3.1-flash-lite:gemini-the-water-cycle.10th-grader."
                             "book-review.w450.n0001:0")
    assert (lite[1]["domain"], lite[1]["subdomain"]) == ("creative", "personal_narrative")
    assert (lite[0]["domain"], lite[0]["subdomain"]) == ("academic", "student_essay")
    groups = {splits_mod.group_key(r["id"], "fresh") for r in lite}
    assert groups == {"fresh:gemini-the-water-cycle.10th-grader"}
    assert (splits_mod.group_key(flash[0]["id"], "fresh")
            == "fresh:gemini-justice-in-to-kill-a-mockingbird.10th-grader")


def test_import_csv_rejects_a_bad_file(tmp_path: Path) -> None:
    imp = _script()
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    assert imp.main([str(bad), "--out-dir", str(tmp_path / "o")]) == 2
    no_topic = _csv(tmp_path, [[1, "models/gemini-x", "9th grader", "essay", "250", "Hi.", BODY]])
    assert imp.main([str(no_topic), "--out-dir", str(tmp_path / "o")]) == 2
