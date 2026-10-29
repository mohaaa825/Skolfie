"""Tests for skolfie.deploy.hf_space and the `skolfie space` commands (no network)."""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest

from skolfie import cli
from skolfie.app.gradio_app import NO_AUTH_FLAG
from skolfie.deploy import hf_space
from skolfie.service import BUNDLE_FILE

VERSIONS = {"numpy": "1.26.4", "scikit-learn": "1.5.2", "lightgbm": "4.5.0",
            "torch": "2.5.1+cu121", "transformers": "4.46.2", "gradio": "5.6.0",
            "python": "3.12.7"}


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / BUNDLE_FILE).write_text(json.dumps({"files": {"verdict": "stacked.joblib"}}))
    (bundle / "stacked.joblib").write_bytes(b"model bytes")
    return bundle


def _build(tmp_path: Path, **kw) -> Path:
    return hf_space.build_space(_bundle(tmp_path), tmp_path / "space", git_ref="abc1234",
                                versions=VERSIONS, **kw)


# --------------------------------------------------------------------------- #
# build_space
# --------------------------------------------------------------------------- #
def test_build_space_writes_every_file(tmp_path: Path) -> None:
    out = _build(tmp_path)
    names = {p.name for p in out.iterdir()}
    assert {"README.md", "requirements.txt", "app.py", "packages.txt", ".gitattributes",
            "bundle"} <= names
    assert (out / "bundle" / "stacked.joblib").read_bytes() == b"model bytes"
    assert (out / "bundle" / BUNDLE_FILE).is_file()
    assert "libgomp1" in (out / "packages.txt").read_text()
    assert "*.joblib filter=lfs" in (out / ".gitattributes").read_text()


def test_readme_front_matter(tmp_path: Path) -> None:
    readme = (_build(tmp_path) / "README.md").read_text(encoding="utf-8")
    head = readme.split("---\n")[1]
    assert "sdk: gradio\n" in head and 'sdk_version: "5.6.0"' in head
    assert 'python_version: "3.12"' in head and "app_file: app.py" in head
    assert "never sole proof" in readme


def test_requirements_pin_versions_and_use_the_cpu_torch_index(tmp_path: Path) -> None:
    lines = (_build(tmp_path) / "requirements.txt").read_text().splitlines()
    assert lines[0] == f"--extra-index-url {hf_space.TORCH_CPU_INDEX}"
    assert lines[1] == ("skolfie[features,app] @ "
                        "git+https://github.com/mohaaa825/Skolfie@abc1234")
    assert "torch==2.5.1" in lines                      # +cu121 stripped
    assert "scikit-learn==1.5.2" in lines and "gradio==5.6.0" in lines
    assert not any(line.startswith("pandas") for line in lines)  # not installed: not pinned


def test_app_py_needs_a_login_unless_public(tmp_path: Path) -> None:
    private_app = (_build(tmp_path) / "app.py").read_text()
    assert NO_AUTH_FLAG not in private_app and "PUBLIC = False" in private_app
    compile(private_app, "app.py", "exec")
    public_app = hf_space.app_text(public=True)
    assert NO_AUTH_FLAG in public_app and "PUBLIC = True" in public_app
    compile(public_app, "app.py", "exec")


def test_build_space_rejects_bad_input(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ValueError, match="not a Skolfie bundle"):
        hf_space.build_space(tmp_path, tmp_path / "s", git_ref="x", versions=VERSIONS)
    with pytest.raises(ValueError, match="git_ref"):
        hf_space.build_space(bundle, tmp_path / "s", git_ref=" ", versions=VERSIONS)
    with pytest.raises(ValueError, match="gradio"):
        hf_space.build_space(bundle, tmp_path / "s", git_ref="x", versions={"numpy": "1"})
    with pytest.raises(ValueError, match="contain each other"):
        hf_space.build_space(bundle, bundle / "space", git_ref="x", versions=VERSIONS)


def test_build_space_only_clears_its_own_folders(tmp_path: Path) -> None:
    out = _build(tmp_path)
    (out / "stale.txt").write_text("old")
    hf_space.build_space(tmp_path / "bundle", out, git_ref="def5678", versions=VERSIONS)
    assert not (out / "stale.txt").exists()
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="not empty"):
        hf_space.build_space(tmp_path / "bundle", other, git_ref="x", versions=VERSIONS)
    assert (other / "notes.txt").read_text() == "mine"


def test_collect_versions_has_python() -> None:
    versions = hf_space.collect_versions()
    assert versions["python"].count(".") == 2


@pytest.mark.parametrize("repo_id,site", [("Master/skolfie", "https://master-skolfie.hf.space"),
                                          ("a_b/c.d", "https://a-b-c-d.hf.space")])
def test_urls(repo_id: str, site: str) -> None:
    assert hf_space.space_site_url(repo_id) == site
    assert hf_space.space_page_url(repo_id) == f"https://huggingface.co/spaces/{repo_id}"
    with pytest.raises(ValueError):
        hf_space.space_site_url("no-slash")


# --------------------------------------------------------------------------- #
# publish_space (huggingface_hub replaced by a fake)
# --------------------------------------------------------------------------- #
class FakeApi:
    calls: list = []

    def __init__(self, token: str) -> None:
        FakeApi.calls.append(("init", token))

    def create_repo(self, repo_id, **kw):
        FakeApi.calls.append(("create_repo", repo_id, kw))

    def add_space_secret(self, repo_id, key, value, **kw):
        FakeApi.calls.append(("secret", repo_id, key, value))

    def upload_folder(self, **kw):
        FakeApi.calls.append(("upload", kw))


@pytest.fixture
def fake_hub(monkeypatch):
    FakeApi.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    return FakeApi


def test_publish_space_creates_sets_secret_and_uploads(tmp_path: Path, fake_hub) -> None:
    out = _build(tmp_path)
    url = hf_space.publish_space(out, "me/skolfie", "tok", private=True, auth="teacher:pw123456")
    assert url == "https://huggingface.co/spaces/me/skolfie"
    kinds = [c[0] for c in fake_hub.calls]
    assert kinds == ["init", "create_repo", "secret", "upload"]
    create = fake_hub.calls[1][2]
    assert create["repo_type"] == "space" and create["space_sdk"] == "gradio"
    assert create["private"] is True
    assert fake_hub.calls[2][2:] == (hf_space.AUTH_SECRET, "teacher:pw123456")
    assert fake_hub.calls[3][1]["folder_path"] == str(out)


def test_publish_space_without_auth_leaves_the_secret(tmp_path: Path, fake_hub) -> None:
    hf_space.publish_space(_build(tmp_path), "me/skolfie", "tok", private=False, auth=None)
    assert "secret" not in [c[0] for c in fake_hub.calls]


def test_publish_space_rejects_bad_input(tmp_path: Path, fake_hub) -> None:
    out = _build(tmp_path)
    with pytest.raises(ValueError, match="token"):
        hf_space.publish_space(out, "me/skolfie", "", private=False, auth=None)
    with pytest.raises(ValueError, match="user:password"):
        hf_space.publish_space(out, "me/skolfie", "tok", private=False, auth="nopassword")
    with pytest.raises(ValueError, match="not a Space folder"):
        hf_space.publish_space(tmp_path, "me/skolfie", "tok", private=False, auth=None)
    assert fake_hub.calls == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(argv, **kw):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, stdout=out, stderr=err, **kw)
    return code, out.getvalue(), err.getvalue()


def test_cli_space_build_from_a_bundle(tmp_path: Path, monkeypatch) -> None:
    # Pins come from the installed libraries; CI may not have gradio installed.
    monkeypatch.setattr(hf_space, "collect_versions", lambda: dict(VERSIONS))
    bundle = _bundle(tmp_path)
    code, out, _ = _cli(["space", "build", "--bundle", str(bundle), "--out",
                         str(tmp_path / "space"), "--git-ref", "abc1234"])
    assert code == 0 and "Space folder written" in out
    assert (tmp_path / "space" / "app.py").is_file()


def test_cli_space_build_needs_models(tmp_path: Path) -> None:
    code, _, err = _cli(["space", "build", "--out", str(tmp_path / "s"), "--git-ref", "a"])
    assert code == 2 and "--bundle" in err


def test_cli_space_publish_needs_a_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code, _, err = _cli(["space", "publish", "--dir", str(tmp_path), "--repo-id", "me/s"])
    assert code == 2 and "HF_TOKEN" in err and "settings/tokens" in err


def test_cli_space_publish_uses_env(tmp_path: Path, monkeypatch, fake_hub) -> None:
    out = _build(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setenv("SKOLFIE_APP_AUTH", "teacher:pw123456")
    code, stdout, _ = _cli(["space", "publish", "--dir", str(out), "--repo-id", "me/skolfie"])
    assert code == 0 and "https://me-skolfie.hf.space" in stdout
    assert "secret-token" not in stdout
    assert ("init", "secret-token") in fake_hub.calls


def test_publish_space_explains_payment_required(tmp_path: Path, monkeypatch) -> None:
    class PaidApi(FakeApi):
        def create_repo(self, repo_id, **kw):
            err = RuntimeError("402")
            err.response = types.SimpleNamespace(status_code=402)
            raise err

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=PaidApi))
    with pytest.raises(ValueError, match="PRO"):
        hf_space.publish_space(_build(tmp_path), "me/skolfie", "tok", private=False, auth=None)
