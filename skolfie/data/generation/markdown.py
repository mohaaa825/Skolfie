"""
skolfie/data/generation/markdown.py
-----------------------------------
`strip_markdown`: AI chat answers come with Markdown (headings, **bold**, inline
LaTeX); a student who pastes the answer into a document hands in the words
without those marks, so the generated essays are stored as plain text too.
"""

from __future__ import annotations

import re


def _plain_math(m: re.Match) -> str:
    math = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", m.group(1))
    math = math.replace("\\times", "x").replace("\\cdot", "x").replace("\\rightarrow", "->")
    return re.sub(r"[_^{}\\]", "", math)


def strip_markdown(text: str) -> str:
    """Remove Markdown emphasis and heading marks, keep the words; inline LaTeX
    (`$M_1V_1$`, `$\\text{CO}_2$`) becomes plain text (`M1V1`, `CO2`)."""
    # Math only, not prices ("$10 ... $20"): a backslash command, a single letter, or no
    # leading digit/space plus a _ ^ or = inside.
    text = re.sub(r"\$((?=[^$\n]*\\)[^$\n]{1,80}?|[A-Za-z]"
                  r"|(?=[^$\d\s][^$\n]*[_^=])[^$\n]{1,80}?)\$", _plain_math, text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^(\s*)[*+]\s+", r"\1- ", text)  # "* item" bullets -> "- item"
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"[ \t]+\n", "\n", text)  # Markdown hard breaks ("  \n")
    return text.strip()
