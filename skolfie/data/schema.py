"""
skolfie/data/schema.py
-----------------------
The single record schema every corpus source is normalised into.

Each source loader yields `Record`s; the corpus builder concatenates,
cleans and deduplicates them and writes one Parquet file with exactly
`COLUMNS`. Downstream code (training, evaluation splits) relies only on
these columns, never on source-specific fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LABELS = ("human", "ai")

# Coarse domains used for held-out-domain evaluation. Sources map their own
# finer-grained domains onto these; the original value is kept in `subdomain`.
DOMAINS = ("academic", "news", "creative", "technical", "casual", "qa", "legal", "review", "other")

COLUMNS = [
    "id",          # stable unique id: "<source>:<source-local id>"
    "text",        # the text itself, whitespace-normalised
    "label",       # "human" or "ai"
    "generator",   # model name for AI text (e.g. "gpt-3.5-turbo"), "human" for human text
    "domain",      # one of DOMAINS
    "subdomain",   # the source's own domain/genre name
    "source",      # dataset name, e.g. "hc3", "m4gt", "autextification", "fresh"
    "split_hint",  # the source's own split ("train"/"test"/...) or "" if none
    "n_words",     # whitespace word count of `text`
]


@dataclass
class Record:
    id: str
    text: str
    label: str
    generator: str
    domain: str
    subdomain: str
    source: str
    split_hint: str = ""
    n_words: int = 0

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        if self.label == "human" and self.generator != "human":
            raise ValueError("human records must have generator='human'")
        if not self.n_words:
            self.n_words = len(self.text.split())

    def to_dict(self) -> dict:
        return asdict(self)
