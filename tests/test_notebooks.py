"""
tests/test_notebooks.py
------------------------
The Colab notebooks' shared setup cell (the GitHub token never reaches a URL,
a command line or .git/config) and the CI workflow's permissions.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


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

