"""
skolfie/explain/render.py
--------------------------
Human-readable views of `Explainer` output (roadmap step 8).

render_html      one self-contained page (inline CSS, no scripts, no external
                 assets): per text the verdict and P(ai), the text as a
                 per-chunk heatmap (red = scored AI, blue = scored human, the
                 stronger the tint the more confident the chunk score), and
                 the top features with their direction and reason.
render_markdown  the review sheet: a checklist table to fill in by hand
                 (are the reasons and the highlighted chunks plausible?),
                 then the details of every case, with each chunk's text.

All text that is not a number is escaped (`html.escape` for HTML;
`md_escape` for Markdown: table-breaking characters, raw HTML and link /
image syntax), so a page made from corpus or user text is safe to open.
"""

from __future__ import annotations

import html
import math
from typing import Any, Optional, Sequence

from skolfie.explain.reasons import lookup

_AI_RGB = (214, 39, 40)
_HUMAN_RGB = (31, 119, 180)
_LABEL_COLORS = {"ai": "#d62728", "human": "#1f77b4", "uncertain": "#7f7f7f",
                 "mixed": "#9467bd"}

_CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0 auto;
       max-width: 960px; padding: 16px; color: #1a1a1a; background: #fff; line-height: 1.5; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin: 0 0 8px; }
.case { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 20px 0; }
.meta { color: #555; font-size: 0.9rem; margin-bottom: 8px; }
.note { color: #8a5a00; font-size: 0.9rem; margin: 0 0 8px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff;
         font-weight: 600; font-size: 0.85rem; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; border-left: 3px solid #eee;
        padding: 4px 12px; margin: 12px 0; }
.chunk { border-radius: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #eee;
         vertical-align: top; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ai { color: #d62728; font-weight: 600; } .human { color: #1f77b4; font-weight: 600; }
.legend span { padding: 1px 6px; border-radius: 3px; margin-right: 6px; }
.note { color: #555; font-size: 0.85rem; }
.scroll { overflow-x: auto; }
"""


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    return str(x)


def _num(x: Any, spec: str) -> str:
    """A score formatted with `spec`; "n/a" for None / NaN (JSON-safe explanations hold None)."""
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    return format(float(x), spec)


def _tint(p: Optional[float], threshold: float = 0.5) -> str:
    """Background colour for a chunk: red above the threshold, blue below, by confidence."""
    if p is None or not math.isfinite(float(p)):
        return "rgba(127, 127, 127, 0.15)"
    rgb = _AI_RGB if p >= threshold else _HUMAN_RGB
    strength = min(1.0, abs(p - threshold) / max(threshold, 1 - threshold))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {0.08 + 0.42 * strength:.2f})"


def _heatmap_html(text: str, chunks: Sequence[dict]) -> str:
    parts, pos = [], 0
    for c in sorted(chunks, key=lambda c: (c["start"], c["end"])):
        s, e = max(pos, int(c["start"])), min(len(text), int(c["end"]))
        if e <= s:
            continue
        parts.append(html.escape(text[pos:s]))
        tip = f"P(ai) {_num(c['p'], '.2f')} ({'AI' if c['label'] == 'ai' else 'human'})"
        parts.append(f'<span class="chunk" style="background: {_tint(c["p"])}" '
                     f'title="{html.escape(tip)}">{html.escape(text[s:e])}</span>')
        pos = e
    parts.append(html.escape(text[pos:]))
    return "".join(parts)


def _badge(label: str) -> str:
    color = _LABEL_COLORS.get(label, "#555")
    return f'<span class="badge" style="background: {color}">{html.escape(label.upper())}</span>'


def _features_html(rows: Sequence[dict]) -> str:
    out = ['<div class="scroll"><table><thead><tr><th>#</th><th>feature</th>'
           '<th>value</th><th>typical human / AI</th><th>SHAP (log-odds)</th>'
           '<th>pushes towards</th><th>reason</th></tr></thead><tbody>']
    for k, r in enumerate(rows, 1):
        typical = (f"{_fmt(r.get('typical_human'))} / {_fmt(r.get('typical_ai'))}"
                   if r.get("typical_human") is not None or r.get("typical_ai") is not None
                   else "n/a")
        direction = "ai" if r["direction"] == "ai" else "human"
        out.append(
            f"<tr><td>{k}</td><td>{html.escape(lookup(r['feature']).title)}<br>"
            f'<code class="note">{html.escape(r["feature"])}</code></td>'
            f'<td class="num">{html.escape(_fmt(r["value"]))}</td>'
            f'<td class="num">{html.escape(typical)}</td>'
            f'<td class="num">{_num(r["contribution"], "+.3f")}</td>'
            f'<td class="{direction}">{"AI" if direction == "ai" else "human"}</td>'
            f"<td>{html.escape(r['reason'])}</td></tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def disagreement_note(ex: dict) -> str:
    """A plain-text warning when the feature model alone points the other way, else ""."""
    fm = ex.get("feature_model") or {}
    if fm.get("agrees") is not False:
        return ""
    side = "AI" if (fm.get("p_ai") or 0) >= 0.5 else "human"
    return (f"The style features alone point to {side} (P(ai) {_fmt(fm.get('p_ai'))}), so the "
            "verdict rests mainly on the other model(s) and these features do not explain it.")


def render_html(explanations: Sequence[dict], texts: Sequence[str],
                captions: Optional[Sequence[str]] = None,
                title: str = "Skolfie explanations") -> str:
    """A self-contained HTML page for the explanations of `texts` (same order)."""
    if len(explanations) != len(texts):
        raise ValueError("explanations and texts must have the same length")
    if captions is not None and len(captions) != len(texts):
        raise ValueError("captions and texts must have the same length")
    body = [
        f"<h1>{html.escape(title)}</h1>",
        '<p class="legend">Chunks: <span style="background: '
        f'{_tint(0.95)}">scored AI</span><span style="background: {_tint(0.05)}">scored '
        "human</span>(stronger colour = more confident; hover a chunk for its score). "
        "Features: the feature model's top SHAP contributions in log-odds; positive pushes "
        "towards AI.</p>",
    ]
    for i, (ex, text) in enumerate(zip(explanations, texts)):
        caption = captions[i] if captions is not None else f"Text {i + 1}"
        fm = ex.get("feature_model") or {}
        verdict_note = ""
        if ex.get("verdict") and ex["verdict"] != ex["label"]:
            verdict_note = f" (verdict without the chunk rule: {html.escape(ex['verdict'])})"
        note = disagreement_note(ex)
        note = f'<p class="note">{html.escape(note)}</p>' if note else ""
        body.append(
            '<section class="case">'
            f"<h2>{html.escape(caption)}</h2>"
            f'<div class="meta">{_badge(ex["label"])} P(ai) {_num(ex["p_ai"], ".3f")}{verdict_note}'
            f" &middot; feature model P(ai) {_fmt(fm.get('p_ai'))}"
            f" &middot; {len(ex.get('chunks') or [])} chunk(s)</div>"
            f"{note}"
            f'<div class="text">{_heatmap_html(text, ex.get("chunks") or [])}</div>'
            f"{_features_html(ex.get('top_features') or [])}"
            "</section>")
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")


# --------------------------------------------------------------------------- #
# Markdown review sheet
# --------------------------------------------------------------------------- #
_MD_SPECIAL = str.maketrans({c: "\\" + c for c in "\\|`[]"})


def md_escape(text: Any) -> str:
    """Make text safe inside a Markdown table cell: one line, no HTML, no cell breaks, no
    code spans, links or images."""
    s = str(text).translate(_MD_SPECIAL)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return " ".join(s.split())


def _feature_line(r: dict) -> str:
    arrow = "AI" if r["direction"] == "ai" else "human"
    return f"{md_escape(r['feature'])} → {arrow}: {md_escape(r['reason'])}"


def render_markdown(explanations: Sequence[dict], texts: Sequence[str],
                    cases: Optional[Sequence[dict]] = None,
                    title: str = "Explanation review sheet") -> str:
    """
    The review sheet. `cases` (same order as texts) may give each text an
    "id" and its "true_label"; both show as "?" when missing.
    """
    if len(explanations) != len(texts):
        raise ValueError("explanations and texts must have the same length")
    cases = list(cases) if cases is not None else [{} for _ in texts]
    if len(cases) != len(texts):
        raise ValueError("cases and texts must have the same length")
    lines = [
        f"# {md_escape(title)}",
        "",
        "For each case, read the text (`review.html` shows the chunk heatmap) and fill in the "
        "last two columns with y or n:",
        "",
        "- **reasons plausible?** Do the top features and their reasons describe this text "
        "(e.g. is the vocabulary really this varied, are the sentences really this uniform)?",
        "- **highlight plausible?** Are the chunks marked AI the ones that read as AI?",
        "",
        "| # | id | true label | verdict | P(ai) | top features | reasons plausible? (y/n) "
        "| highlight plausible? (y/n) |",
        "|---:|---|---|---|---:|---|---|---|",
    ]
    for i, (ex, case) in enumerate(zip(explanations, cases), 1):
        feats = "<br>".join(f"{k}. {_feature_line(r)}"
                            for k, r in enumerate(ex.get("top_features") or [], 1))
        lines.append(f"| {i} | {md_escape(case.get('id', '?'))} | "
                     f"{md_escape(case.get('true_label', '?'))} | {md_escape(ex['label'])} | "
                     f"{_num(ex['p_ai'], '.3f')} | {feats} |  |  |")

    lines += ["", "## Cases", ""]
    for i, (ex, text, case) in enumerate(zip(explanations, texts, cases), 1):
        fm = ex.get("feature_model") or {}
        note = disagreement_note(ex)
        lines += [
            f"### {i}. {md_escape(case.get('id', f'text {i}'))}",
            "",
            f"True label **{md_escape(case.get('true_label', '?'))}**, verdict "
            f"**{md_escape(ex['label'])}** (P(ai) {_num(ex['p_ai'], '.3f')}"
            + (f", without the chunk rule: {md_escape(ex['verdict'])}"
               if ex.get("verdict") and ex["verdict"] != ex["label"] else "")
            + f"); feature model P(ai) {_fmt(fm.get('p_ai'))}.",
            "",
            *([f"> {md_escape(note)}", ""] if note else []),
            "| # | feature | value | typical human / AI | SHAP (log-odds) | towards | reason |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
        for k, r in enumerate(ex.get("top_features") or [], 1):
            lines.append(
                f"| {k} | {md_escape(r['feature'])} | {_fmt(r['value'])} | "
                f"{_fmt(r.get('typical_human'))} / {_fmt(r.get('typical_ai'))} | "
                f"{_num(r['contribution'], '+.3f')} | "
                f"{'AI' if r['direction'] == 'ai' else 'human'} | "
                f"{md_escape(r['reason'])} |")
        lines += ["", "| chunk | P(ai) | scored | text |", "|---:|---:|---|---|"]
        for k, c in enumerate(ex.get("chunks") or [], 1):
            scored = "**AI**" if c["label"] == "ai" else "human"
            lines.append(f"| {k} | {_num(c['p'], '.3f')} | {scored}"
                         f" | {md_escape(text[c['start']:c['end']])} |")
        lines.append("")
    return "\n".join(lines) + "\n"
