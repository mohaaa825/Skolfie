"""
tests/test_review_newer.py
---------------------------
Regression tests from the review of roadmap steps 7-9 (Mixed detection,
explanations, evasion attacks), the Colab notebooks and CI.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skolfie.robust import attacks as A
from tests.test_robust import _make_argv
from tests.test_robust import data as data  # noqa: F401  (module-scoped fixture)
from tests.test_stacker import _load_script

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


# --------------------------------------------------------------------------- #
# Notebooks and CI
# --------------------------------------------------------------------------- #
def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _git_function(src: str) -> str:
    """The notebook's `def git(...)` block (up to the next top-level statement)."""
    m = re.search(r"^def git\(.*?(?=^\S)", src, re.DOTALL | re.MULTILINE)
    assert m, "no git() helper in the setup cell"
    return m.group(0)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_github_token_never_in_url_or_command(path: Path, tmp_path: Path) -> None:
    cells = [c for c in _code_cells(path) if "GITHUB_TOKEN" in c]
    assert len(cells) == 1
    src = cells[0]
    # The old pattern put the token in the clone URL: printed by CalledProcessError and
    # kept in /content/Skolfie/.git/config.
    assert "@github.com" not in src and "{token}" not in src
    assert 'url = f"https://github.com/{REPO}.git"' in src
    # An existing clone gets its remote URL reset (older runs stored the token there).
    assert '"remote", "set-url", "origin", url' in src

    # Run the notebook's git() with a fake secret: the token must reach git only through
    # the environment, and the credential helper must hand it to git.
    calls = []
    ns = {"os": __import__("os"), "secret": lambda name: "tok-SECRET-123"
          if name == "GITHUB_TOKEN" else None,
          "subprocess": type("S", (), {"run": staticmethod(
              lambda argv, env, check: calls.append((argv, env)))})}
    exec(_git_function(src), ns)  # noqa: S102 - our own notebook code
    ns["git"]("clone", "https://github.com/x/y.git", "/content/Skolfie")
    argv, env = calls[0]
    assert all("tok-SECRET-123" not in a for a in argv)
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    auth = argv[1:argv.index("clone")]
    out = subprocess.run(["git", *auth, "credential", "fill"], env=env, cwd=tmp_path,
                         input="protocol=https\nhost=github.com\n\n", capture_output=True,
                         text=True, timeout=30, check=True).stdout
    assert "password=tok-SECRET-123" in out.splitlines()


def test_notebook_setup_cells_are_consistent() -> None:
    blocks = {_git_function(c) for p in NOTEBOOKS for c in _code_cells(p) if "GITHUB_TOKEN" in c}
    assert len(blocks) == 1


def test_ci_workflow_is_least_privilege() -> None:
    yaml = pytest.importorskip("yaml")
    ci = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert ci["permissions"] == {"contents": "read"}
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "${{ secrets" not in text
    for uses in re.findall(r"uses:\s*(\S+)", text):
        assert re.search(r"@v?\d", uses), f"unpinned action {uses}"
    assert "persist-credentials: false" in text


# --------------------------------------------------------------------------- #
# render.py
# --------------------------------------------------------------------------- #
def _explanation(text: str, feature: str = "sty_mattr50", direction: str = "ai") -> dict:
    return {
        "label": "ai", "verdict": "ai", "p_ai": 0.9,
        "chunks": [{"start": 0, "end": len(text), "p": 0.9, "label": "ai"}],
        "top_features": [{"feature": feature, "value": 0.4, "typical_human": 0.7,
                          "typical_ai": 0.5, "contribution": 1.0, "direction": direction,
                          "reason": "r"}],
        "feature_model": {"p_ai": 0.8},
    }


def test_render_html_attribute_and_class_injection() -> None:
    from skolfie.explain.render import render_html

    text = 'He said "hi" & <img src=x onerror=alert(1)> `code` | pipe'
    ex = _explanation(text, feature='f"><script>x</script>', direction='ai" onclick="x')
    page = render_html([ex], [text], captions=['" onmouseover="x'],
                       title="<script>t</script>")
    assert "<script>" not in page and "<img" not in page
    assert 'onclick="' not in page and 'onmouseover="' not in page
    assert "&quot;hi&quot;" in page


def _cells(row: str) -> int:
    """Number of cells in a Markdown table row (escaped pipes do not split)."""
    return len(re.split(r"(?<!\\)\|", row.strip().strip("|")))


def test_render_markdown_tables_and_links() -> None:
    from skolfie.explain.render import md_escape, render_markdown

    text = "Look ![pixel](http://evil.example/p.png) and [click](javascript:alert(1)) now."
    md = render_markdown([_explanation(text, feature="a|b`c")], [text],
                         cases=[{"id": "x](http://evil)", "true_label": "ai"}])
    rows = [ln for ln in md.splitlines() if ln.startswith("| 1 | ")]
    # The checklist row (8 cells), the feature row (7) and the chunk row (4).
    assert sorted(_cells(r) for r in rows) == [4, 7, 8]
    # No live link or image from user text.
    assert not re.search(r"(?<!\\)\]\(", md)
    assert md_escape("[a](b)") == "\\[a\\](b)"


# --------------------------------------------------------------------------- #
# attacks.py
# --------------------------------------------------------------------------- #
def test_clean_llm_output_keeps_inner_quotes() -> None:
    body = " ".join(["word"] * 25)
    dialogue = f'"Hello," she said. {body} "Goodbye."'
    assert A.clean_llm_output(dialogue) == dialogue
    assert A.clean_llm_output(f'"{body}"') == body
    assert A.clean_llm_output(f'"""\n{body}\n"""') == body


def test_run_prompts_redacts_api_key_in_logs() -> None:
    from skolfie.data.generation.providers import ProviderError

    class Leaky:
        model_name = "m"
        api_key = "sk-very-secret-key"

        def generate(self, prompt, *, temperature, max_tokens):
            raise ProviderError("bad request for key sk-very-secret-key", status=400)

    logs: list[str] = []
    assert A.run_prompts(Leaky(), ["a"], temperature=0.5, max_tokens=[10],
                         log=logs.append) == [None]
    assert logs and all("sk-very-secret-key" not in m for m in logs)


# --------------------------------------------------------------------------- #
# make_attacks.py
# --------------------------------------------------------------------------- #
def test_make_attacks_cache_not_reused_after_interrupted_write(data, monkeypatch) -> None:  # noqa: F811
    script = _load_script("make_attacks")
    out = data["root"] / "attacks-interrupt"
    argv = _make_argv(data, out, "--n", "6", "--provider", "none", "--attacks", "typo_10")
    assert script.main(argv) == 0
    first = pd.read_parquet(out / "attacks.parquet")

    # A run with another seed is interrupted right after the part file is written.
    original = pd.DataFrame.to_parquet

    def interrupted(self, path, *args, **kwargs):
        original(self, path, *args, **kwargs)
        if "parts" in Path(path).parts:
            raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(KeyboardInterrupt):
        script.main(argv[:-2] + ["--seed", "1", "--attacks", "typo_10"])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", original)

    # Back to the first command: the seed-1 part must not be reused as if it were seed 0.
    assert script.main(argv) == 0
    pd.testing.assert_frame_equal(pd.read_parquet(out / "attacks.parquet"), first)


def test_make_attacks_corrupt_signature_rebuilds(data, capsys) -> None:  # noqa: F811
    script = _load_script("make_attacks")
    out = data["root"] / "attacks-badsig"
    argv = _make_argv(data, out, "--n", "4", "--provider", "none", "--attacks", "typo_5")
    assert script.main(argv) == 0
    (out / "parts" / "typo_5.json").write_text("{not json")
    capsys.readouterr()
    assert script.main(argv) == 0
    assert "reused" not in capsys.readouterr().out


def test_make_attacks_share_too_small(data) -> None:  # noqa: F811
    script = _load_script("make_attacks")
    with pytest.raises(SystemExit) as exc:
        script.main(_make_argv(data, data["root"] / "attacks-tiny", "--split", "train",
                               "--share", "0.001", "--attacks", "typo_5"))
    assert exc.value.code == 2


def test_make_attacks_rejects_path_like_attack_names(data) -> None:  # noqa: F811
    script = _load_script("make_attacks")
    for bad in ("../typo_5", "typo_5/../../x", "typo_5\x00"):
        with pytest.raises(SystemExit):
            script.main(_make_argv(data, data["root"] / "attacks-bad", "--attacks", bad))


# --------------------------------------------------------------------------- #
# eval_robust.py
# --------------------------------------------------------------------------- #
def test_eval_robust_attack_without_ai_rows(data) -> None:  # noqa: F811
    make = _load_script("make_attacks")
    attacks_dir = data["root"] / "attacks-noai"
    assert make.main(_make_argv(data, attacks_dir, "--n", "10", "--provider", "none",
                                "--attacks", "typo_5")) == 0
    df = pd.read_parquet(attacks_dir / "attacks.parquet")
    df = df[~((df["attack"] == "typo_5") & (df["label"] == "ai"))]
    path = attacks_dir / "noai.parquet"
    df.to_parquet(path, index=False)
    script = _load_script("eval_robust")
    out = data["root"] / "robust-noai"
    assert script.main(["--attacks", str(path), "--model", str(data["tfidf"]),
                        "--out-dir", str(out)]) == 0
    report = json.loads((out / "robust_report.json").read_text())
    typo = report["models"]["tfidf"]["attacks"]["typo_5"]
    assert typo["n_ai"] == 0 and typo["auroc"] is None and typo["n_human"] == 10
    assert "| typo_5 | 0 / 10 |" in (out / "robust_report.md").read_text()


def test_threshold_at_fpr_ignores_nan() -> None:
    script = _load_script("eval_robust")
    p = np.array([np.nan, 0.1, 0.2, 0.3])
    t = script.threshold_at_fpr(p, 0.0)
    assert t > 0.3 and np.isfinite(t)
    assert np.isnan(script.threshold_at_fpr([np.nan], 0.1))


# --------------------------------------------------------------------------- #
# explain.py / eval_mixed.py
# --------------------------------------------------------------------------- #
def test_explain_corrupt_mixed_report_is_a_usage_error(tmp_path: Path) -> None:
    script = _load_script("explain")
    bad = tmp_path / "mixed_report.json"
    bad.write_text('{"rules": {"chunk"')  # half-written
    args = script.build_parser().parse_args(["--corpus", "c", "--splits", "s", "--model", "m",
                                             "--mixed-report", str(bad)])
    with pytest.raises(SystemExit):
        script.chunk_settings(args, script.build_parser())


def test_eval_mixed_warns_when_fit_and_eval_split_are_the_same(tmp_path: Path, capsys) -> None:
    import joblib

    from skolfie.models.baselines import TfidfLRBaseline
    from tests.test_eval_mixed import _argv
    from tests.test_evaluate import make_corpus, make_splits

    corpus = make_corpus(n_per_cell=60, seed=5)
    splits = make_splits(corpus, seed=5)
    paths = {"corpus": tmp_path / "corpus.parquet", "splits": tmp_path / "splits.parquet",
             "model": tmp_path / "tfidf.joblib"}
    corpus.to_parquet(paths["corpus"], index=False)
    splits.to_parquet(paths["splits"], index=False)
    joblib.dump(TfidfLRBaseline().fit(corpus["text"].tolist(), corpus["label"].tolist()),
                paths["model"])
    out = tmp_path / "out"
    assert _load_script("eval_mixed").main(
        _argv(paths, "--fit-split", "test", "--eval-split", "test", "--out-dir", str(out))) == 0
    assert "optimistic" in capsys.readouterr().err
    assert json.loads((out / "mixed_report.json").read_text())["fit_split"] == "test"
    assert not list(out.glob("*.tmp"))


def test_render_handles_missing_scores() -> None:
    """Explanations are JSON-safe, so a NaN score arrives as None."""
    from skolfie.explain.render import render_html, render_markdown

    text = "Some text here."
    ex = _explanation(text)
    ex["p_ai"] = None
    ex["chunks"][0]["p"] = None
    ex["top_features"][0]["contribution"] = None
    assert "n/a" in render_html([ex], [text])
    assert "n/a" in render_markdown([ex], [text])


def test_attack_names_match_whole_string() -> None:
    for bad in ("typo_5\n", "typo_5/..", " typo_5"):
        with pytest.raises(ValueError):
            A.get_attack(bad)


def test_make_attacks_never_writes_the_api_key(data, monkeypatch, capsys) -> None:  # noqa: F811
    from skolfie.data.generation import providers

    secret = "sk-ant-THIS-IS-SECRET-42"
    for _, key in _load_script("make_attacks").API_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    calls = []

    def fake_post(url, headers, payload, timeout):
        calls.append(headers)
        assert secret not in url
        if len(calls) == 1:  # an API error that echoes the key
            return providers.HTTPResponse(400, {}, json.dumps({"error": f"bad key {secret}"}))
        words = " ".join(["reworded"] * 30)
        return providers.HTTPResponse(200, {}, json.dumps({
            "content": [{"type": "text", "text": words}],
            "usage": {"input_tokens": 1, "output_tokens": 1}}))

    monkeypatch.setattr(providers, "_http_post", fake_post)
    out = data["root"] / "attacks-key"
    assert _load_script("make_attacks").main(_make_argv(
        data, out, "--n", "4", "--provider", "auto", "--workers", "1",
        "--attacks", "llm_paraphrase,temp_low")) == 0
    assert any(secret in str(h.values()) for h in calls)  # it was used...
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err  # ...but never printed
    assert "request 0 failed" in captured.out
    for path in out.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path
