"""Sanity checks for the static website in website/."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "website"


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.controls: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if a.get("aria-controls"):
            self.controls.append(a["aria-controls"])
        if tag in ("a", "link") and a.get("href"):
            self.hrefs.append(a["href"])
        if tag in ("script", "img") and a.get("src"):
            self.srcs.append(a["src"])


def _page() -> _Collector:
    c = _Collector()
    c.feed((SITE / "index.html").read_text(encoding="utf-8"))
    return c


def test_files_exist():
    for name in ("index.html", "styles.css", "main.js", "config.js", "assets/logo.svg"):
        assert (SITE / name).is_file(), name


def test_in_page_anchors_resolve():
    c = _page()
    for href in c.hrefs:
        if href.startswith("#") and len(href) > 1:
            assert href[1:] in c.ids, href
    for target in c.controls:
        assert target in c.ids, target


def test_local_assets_exist_and_no_third_party_scripts():
    c = _page()
    for ref in c.srcs + [h for h in c.hrefs if not h.startswith(("#", "http", "mailto:"))]:
        assert (SITE / ref).is_file(), ref
    for src in c.srcs:
        assert not src.startswith("http"), f"external script {src}"
    external = [h for h in c.hrefs if h.startswith("http")]
    assert all(h.startswith("https://fonts.g") for h in external), external


def test_config_ships_without_links_or_secrets():
    cfg = (SITE / "config.js").read_text(encoding="utf-8")
    assert re.search(r'appUrl:\s*""', cfg)
    assert re.search(r'contactEmail:\s*""', cfg)
    assert "gradio.live" not in cfg


def test_palette_has_blue_and_pink():
    css = (SITE / "styles.css").read_text(encoding="utf-8").lower()
    assert "#7cc4f2" in css  # baby blue, same as the app
    assert "--pink-300" in css


def test_background_photos_referenced_in_css_and_html_exist():
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    html_text = (SITE / "index.html").read_text(encoding="utf-8")
    refs = set(re.findall(r"url\((assets/[^)]+)\)", css + html_text))
    assert refs, "expected background photos"
    for ref in refs:
        path = SITE / ref
        assert path.is_file(), ref
        assert path.stat().st_size < 400_000, f"{ref} is too heavy for a background"


def test_motion_respects_reduced_motion():
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    html_text = (SITE / "index.html").read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in css
    assert "prefers-reduced-motion: reduce" in html_text
