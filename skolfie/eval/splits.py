"""
skolfie/eval/splits.py
----------------------
Deterministic, versioned train/val/test splits of the corpus that test
generalisation (to unseen generators and unseen domains), not memorisation.

Contract with the evaluator (`write_splits`):

    out_dir/splits.parquet        columns `id` (str), `split` (str); one row per
                                  (id, split) membership, sorted by split (in
                                  `SPLITS` order) then id
    out_dir/splits_manifest.json  splits_version, corpus_version, config + its
                                  hash, split sizes, warnings, and
                                  split x label x generator x domain counts
    out_dir/splits.md             human-readable summary

Every corpus row has exactly one *primary* split (`PRIMARY_SPLITS`), which
`assign_splits` computes in this order:

1. `test_ood_domain`: any row (human or AI) whose domain is held out.
2. `test_ood_generator`: AI rows whose generator *family* is held out
   (`generator_families` maps generators to families; holding out one
   generator holds out its whole family; an unmapped generator is its own
   family).
3. `train` / `val` / `test`: everything else, by a stable hash of
   (seed, group_key). The hash is blake2b of the key, so a row's split never
   depends on the other rows, their order, pandas/numpy versions or a random
   state: adding rows to the corpus never moves existing rows.
   Exception, for leak safety: a group that contains any held-out row (rule
   1 or 2) is sent to `test` as a whole, so that e.g. the human answer to a
   question whose held-out-model answer is in `test_ood_generator` is never
   trained on. Only a new row that brings a held-out generator/domain into
   an existing group can therefore move that group's other rows (to test).

Two splits are *extra* memberships on top of the primary one, which is why the
file is a long (id, split) table rather than one split per id:

- `test_ood_generator` also gets human texts, needed to measure false
  positives next to the unseen-generator AI texts: the human rows of the
  in-distribution `test` split whose domain is one the held-out AI rows
  cover. They stay in `test` as well (one id, two memberships). They come
  from `test`, never `train`/`val`, so they are just as unseen; and taking
  them only from covered domains keeps the split's domain mix comparable
  between its two classes.
- `test_short`: rows of `test` with `n_words < short_text_max_words`.

Groups (`group_key`): texts that must stay on the same side of the split.

- hc3 `hc3:<subset>:<line>:<human|chatgpt>:<i>` -> `hc3:<subset>:<line>`:
  every answer (human and ChatGPT) to one question.
- fresh `fresh:<generator>:<template>.<slots>.<style>.w<N>.t<T>:<i>` ->
  `fresh:<template>.<slots>`: one filled topic, across generators, styles,
  lengths, temperatures and samples.
- m4gt, autextification: the id itself. Their loaders expose no pairing
  between a human text and the generations made from it (M4GT's
  `parallel` field is not loaded; AuTexTification subtask 1 releases none).

Guarantees, checked by `check_guarantees` on every `build_memberships` call
(it raises `SplitLeakError`, an `AssertionError`, also under `python -O`):

- no id is both in train and in val or any test split;
- no group is in more than one of train / val / test-side (test-side is
  `test` plus every `test_*` split);
- no held-out-family AI text is in train or val;
- no held-out-domain text is in train or val.

Everything is vectorised; the only Python loop is one blake2b call per unique
group key (about 0.2 s for 260k keys).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from skolfie.data.schema import LABELS

#: Version of the file contract and of the assignment rules. Bump it when a
#: change to this module can move rows between splits.
SPLITS_FORMAT_VERSION = 1

FRACTION_SPLITS = ("train", "val", "test")
PRIMARY_SPLITS = ("train", "val", "test", "test_ood_generator", "test_ood_domain")
SPLITS = (*PRIMARY_SPLITS, "test_short")
TEST_SPLITS = ("test", "test_ood_generator", "test_ood_domain", "test_short")
_SPLIT_CODE = {name: i for i, name in enumerate(SPLITS)}
_SPLIT_INDEX = pd.Index(SPLITS)

#: splits.md warns when a split has fewer texts than this of a label.
MIN_TEXTS_PER_LABEL = 500

REQUIRED_COLUMNS = ("id", "label", "generator", "domain", "source", "n_words")

PARQUET_SCHEMA = pa.schema(
    [pa.field("id", pa.string(), nullable=False), pa.field("split", pa.string(), nullable=False)]
)


class SplitLeakError(AssertionError):
    """A split guarantee is violated (train/test leakage)."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class SplitConfig:
    seed: int = 0
    fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    held_out_generators: list[str] = field(default_factory=list)
    held_out_domains: list[str] = field(default_factory=list)
    short_text_max_words: int = 100
    generator_families: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        self.fractions = {k: float(v) for k, v in dict(self.fractions).items()}
        if set(self.fractions) != set(FRACTION_SPLITS):
            raise ValueError(
                f"fractions must have exactly the keys {FRACTION_SPLITS}, "
                f"got {sorted(self.fractions)}"
            )
        if any(v < 0 for v in self.fractions.values()):
            raise ValueError(f"fractions must be >= 0, got {self.fractions}")
        if abs(sum(self.fractions.values()) - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {self.fractions}")
        for name in ("held_out_generators", "held_out_domains"):
            value = getattr(self, name)
            if isinstance(value, str) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{name} must be a list of strings, got {value!r}")
            setattr(self, name, list(value))
        if not isinstance(self.short_text_max_words, int) or self.short_text_max_words < 1:
            raise ValueError(
                f"short_text_max_words must be a positive int, got {self.short_text_max_words!r}"
            )
        self.generator_families = {str(k): str(v) for k, v in dict(self.generator_families).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitConfig:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown split config key(s) {unknown}; known: {sorted(known)}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fractions": {k: self.fractions[k] for k in FRACTION_SPLITS},
            "held_out_generators": list(self.held_out_generators),
            "held_out_domains": list(self.held_out_domains),
            "short_text_max_words": self.short_text_max_words,
            "generator_families": dict(sorted(self.generator_families.items())),
        }

    def config_hash(self) -> str:
        """sha256 of the canonical JSON form (key order and list order of
        held-out names do not matter)."""
        d = self.to_dict()
        d["held_out_generators"] = sorted(set(d["held_out_generators"]))
        d["held_out_domains"] = sorted(set(d["held_out_domains"]))
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def family(self, generator: str) -> str:
        return self.generator_families.get(generator, generator)

    @property
    def held_out_families(self) -> frozenset[str]:
        return frozenset(self.family(g) for g in self.held_out_generators)


def load_config(path: str | Path) -> SplitConfig:
    with open(path, encoding="utf-8") as fh:
        return SplitConfig.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #

# source -> (pattern on the id, replacement giving the group key). An id that
# does not match keeps itself as its group. The same (pattern, replacement)
# pair is used by the scalar and the vectorised function.
_GROUP_RULES: dict[str, tuple[re.Pattern[str], str]] = {
    # hc3:<subset>:<line>:<human|chatgpt>:<i>
    "hc3": (re.compile(r"^(hc3:[^:]+:[^:]+):(?:human|chatgpt):\d+$"), r"\1"),
    # fresh:<generator, may contain ':'>:<template>.<slots>.<style>.w<N>.t<T>:<i>
    "fresh": (re.compile(r"^fresh:.+:([^:.]+\.[^:.]+)\.[^:]+:\d+$"), r"fresh:\1"),
}


def group_key(id: str, source: str) -> str:
    """The key of the group `id` belongs to (see the module docstring)."""
    rule = _GROUP_RULES.get(source)
    if rule is None:
        return id
    pattern, repl = rule
    return pattern.sub(repl, id)


def group_keys(ids: pd.Series, sources: pd.Series) -> pd.Series:
    """Vectorised `group_key` over aligned id and source columns."""
    out = ids.astype(str).copy()
    src = sources.astype(str).to_numpy()
    for source, (pattern, repl) in _GROUP_RULES.items():
        mask = src == source
        if mask.any():
            out[mask] = out[mask].str.replace(pattern, repl, regex=True)
    return out


def _unit_hash(keys: Sequence[str], seed: int) -> np.ndarray:
    """A float in [0, 1) per key from blake2b(seed, key): stable everywhere."""
    prefix = f"{seed}\x1f".encode()
    digests = b"".join(
        hashlib.blake2b(prefix + k.encode("utf-8"), digest_size=8).digest() for k in keys
    )
    h = np.frombuffer(digests, dtype=">u8").astype(np.uint64)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def _check_frame(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"corpus frame is missing column(s) {missing}")
    n_null = int(df["id"].isna().sum())
    if n_null:
        raise ValueError(f"{n_null} corpus row(s) have no id")
    dup = df["id"].duplicated()
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} duplicate id(s) in the corpus, e.g. {df['id'][dup].iloc[0]!r}"
        )


def _families(df: pd.DataFrame, config: SplitConfig) -> pd.Series:
    gen = df["generator"].astype(str)
    return gen.map(config.generator_families).fillna(gen)


def _held_out_masks(df: pd.DataFrame, config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """(ood_domain, ood_generator) boolean arrays; they never overlap."""
    ood_domain = df["domain"].isin(config.held_out_domains).to_numpy()
    is_ai = (df["label"] == "ai").to_numpy()
    held_family = _families(df, config).isin(config.held_out_families).to_numpy()
    return ood_domain, is_ai & held_family & ~ood_domain


def assign_splits(df: pd.DataFrame, config: SplitConfig) -> pd.Series:
    """The primary split of each row (a Series named "split", aligned with df).

    Values are in `PRIMARY_SPLITS`. Depends only on each row's own fields and
    group (not on row order or the rest of the corpus), see the module docstring.
    """
    _check_frame(df)
    groups = group_keys(df["id"], df["source"])
    codes, uniques = pd.factorize(groups, sort=False)
    ood_domain, ood_gen = _held_out_masks(df, config)

    tainted_group = np.zeros(len(uniques), dtype=bool)
    tainted_group[codes[ood_domain | ood_gen]] = True

    u = _unit_hash(uniques, config.seed)[codes]
    f_train = config.fractions["train"]
    f_val = config.fractions["val"]
    split = np.where(u < f_train, "train", np.where(u < f_train + f_val, "val", "test"))
    split = split.astype(object)
    split[tainted_group[codes]] = "test"
    split[ood_gen] = "test_ood_generator"
    split[ood_domain] = "test_ood_domain"
    return pd.Series(split, index=df.index, name="split", dtype=object)


def _covered_domains(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> list[str]:
    return sorted(pd.unique(df["domain"].to_numpy()[np.asarray(mask, dtype=bool)]).tolist())


def ood_generator_human_domains(df: pd.DataFrame, primary: pd.Series) -> list[str]:
    """Domains covered by the held-out-generator AI rows (whose human `test`
    rows also join `test_ood_generator`)."""
    return _covered_domains(df, primary == "test_ood_generator")


def build_memberships(
    df: pd.DataFrame, primary: pd.Series, config: SplitConfig, *, check: bool = True
) -> pd.DataFrame:
    """The long (id, split) table: primary splits plus the extra memberships
    (human texts in `test_ood_generator`, `test_short`). Sorted by split in
    `SPLITS` order, then id. Runs `check_guarantees` unless check=False."""
    _check_frame(df)
    codes = _SPLIT_INDEX.get_indexer(primary)
    if (codes < 0).any() or len(codes) != len(df):
        raise ValueError("primary must hold one split name from SPLITS per corpus row")
    in_test = codes == _SPLIT_CODE["test"]
    covered = ood_generator_human_domains(df, primary)
    is_human = (df["label"] == "human").to_numpy()
    human_ood = np.flatnonzero(in_test & is_human & df["domain"].isin(covered).to_numpy())
    short = np.flatnonzero(in_test & (df["n_words"].to_numpy() < config.short_text_max_words))

    pos = np.concatenate([np.arange(len(df)), human_ood, short])
    split_codes = np.concatenate(
        [
            codes,
            np.full(len(human_ood), _SPLIT_CODE["test_ood_generator"], dtype=codes.dtype),
            np.full(len(short), _SPLIT_CODE["test_short"], dtype=codes.dtype),
        ]
    )
    id_rank = np.empty(len(df), dtype=np.int64)
    id_rank[np.argsort(df["id"].to_numpy(dtype=object), kind="stable")] = np.arange(len(df))
    order = np.lexsort((id_rank[pos], split_codes))
    pos, split_codes = pos[order], split_codes[order]
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy(dtype=object)[pos],
            "split": np.asarray(SPLITS, dtype=object)[split_codes],
        }
    )
    if check:
        check_guarantees(df, out, config)
    return out


def make_splits(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """`assign_splits` + `build_memberships` (checked): the (id, split) table."""
    return build_memberships(df, assign_splits(df, config), config)


def _positions(df: pd.DataFrame, memberships: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(row position in df, index in SPLITS) per membership row."""
    pos = pd.Index(df["id"]).get_indexer(memberships["id"])
    if (pos < 0).any():
        bad = memberships["id"].to_numpy(dtype=object)[pos < 0][0]
        raise SplitLeakError(f"splits contain ids that are not in the corpus, e.g. {bad!r}")
    codes = _SPLIT_INDEX.get_indexer(memberships["split"])
    if (codes < 0).any():
        unknown = sorted(set(memberships["split"]) - set(SPLITS))
        raise SplitLeakError(f"unknown split name(s) {unknown}")
    return pos.astype(np.int64), codes.astype(np.int64)


def _membership_matrix(df: pd.DataFrame, memberships: pd.DataFrame) -> np.ndarray:
    """Boolean (len(df), len(SPLITS)) matrix: row i is a member of split j."""
    pos, codes = _positions(df, memberships)
    if len(np.unique(pos * len(SPLITS) + codes)) != len(pos):
        raise SplitLeakError("an (id, split) membership is listed twice")
    member = np.zeros((len(df), len(SPLITS)), dtype=bool)
    member[pos, codes] = True
    return member


def check_guarantees(df: pd.DataFrame, memberships: pd.DataFrame, config: SplitConfig) -> None:
    """Raise `SplitLeakError` if any split guarantee is violated."""
    _check_frame(df)
    member = _membership_matrix(df, memberships)
    col = _SPLIT_CODE
    in_train = member[:, col["train"]]
    in_val = member[:, col["val"]]
    in_test_side = member[:, [col[s] for s in TEST_SPLITS]].any(axis=1)

    # Every corpus row has exactly one primary split: one "core" split, or
    # (held-out-generator AI rows only) test_ood_generator and no core split.
    core = member[:, [col[s] for s in ("train", "val", "test", "test_ood_domain")]].sum(axis=1)
    primary_ok = (core == 1) | ((core == 0) & member[:, col["test_ood_generator"]])
    if not primary_ok.all():
        bad = df["id"].to_numpy(dtype=object)[~primary_ok][0]
        raise SplitLeakError(f"a corpus row has no or several primary splits, e.g. {bad!r}")

    # 1. no id in train and in any other split
    leak = in_train & (in_val | in_test_side)
    if leak.any():
        raise SplitLeakError(
            f"{int(leak.sum())} id(s) in train and another split, "
            f"e.g. {df['id'].to_numpy(dtype=object)[leak][0]!r}"
        )

    # 2. no group on more than one side (train / val / test-side)
    gcodes, guniq = pd.factorize(group_keys(df["id"], df["source"]))
    n_sides = np.zeros(len(guniq), dtype=np.int64)
    for side in (in_train, in_val, in_test_side):
        hit = np.zeros(len(guniq), dtype=bool)
        hit[gcodes[side]] = True
        n_sides += hit
    bad = np.flatnonzero(n_sides > 1)
    if len(bad):
        raise SplitLeakError(
            f"{len(bad)} group(s) straddle train/val/test, e.g. {guniq[bad[0]]!r}"
        )

    fit = in_train | in_val
    ids = df["id"].to_numpy(dtype=object)
    # 3. no held-out-family AI text in train/val
    ood_domain, _ = _held_out_masks(df, config)
    held_ai = (df["label"] == "ai").to_numpy() & _families(df, config).isin(
        config.held_out_families
    ).to_numpy()
    leak = fit & held_ai
    if leak.any():
        raise SplitLeakError(
            f"{int(leak.sum())} held-out-family AI text(s) in train/val, e.g. {ids[leak][0]!r}"
        )
    # 4. no held-out-domain text in train/val
    leak = fit & ood_domain
    if leak.any():
        raise SplitLeakError(
            f"{int(leak.sum())} held-out-domain text(s) in train/val, e.g. {ids[leak][0]!r}"
        )


# --------------------------------------------------------------------------- #
# Summary and output
# --------------------------------------------------------------------------- #


def split_summary(df: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    """Counts per split x label x generator x domain (columns split, label,
    generator, domain, count), splits in `SPLITS` order."""
    pos, codes = _positions(df, memberships)
    m = pd.DataFrame(
        {
            "split": codes,
            "label": df["label"].to_numpy(dtype=object)[pos],
            "generator": df["generator"].to_numpy(dtype=object)[pos],
            "domain": df["domain"].to_numpy(dtype=object)[pos],
        }
    )
    counts = m.groupby(["split", "label", "generator", "domain"], sort=True).size()
    counts = counts.rename("count").reset_index()
    counts["split"] = np.asarray(SPLITS, dtype=object)[counts["split"].to_numpy()]
    counts["count"] = counts["count"].astype(int)
    return counts


def _split_sizes(summary: pd.DataFrame) -> dict[str, dict[str, int]]:
    sizes: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        part = summary[summary["split"] == split]
        by_label = part.groupby("label")["count"].sum()
        row = {lab: int(by_label.get(lab, 0)) for lab in LABELS}
        row["total"] = int(part["count"].sum())
        sizes[split] = row
    return sizes


def _warnings(
    df: pd.DataFrame, sizes: Mapping[str, Mapping[str, int]], config: SplitConfig
) -> list[str]:
    out = []
    for split in SPLITS:
        for lab in LABELS:
            n = sizes[split][lab]
            if n < MIN_TEXTS_PER_LABEL:
                out.append(f"{split}: only {n} {lab} text(s) (< {MIN_TEXTS_PER_LABEL})")
    gens = set(df["generator"].astype(str))
    present_families = {config.family(g) for g in gens}
    for g in config.held_out_generators:
        if config.family(g) not in present_families:
            out.append(
                f"held-out generator {g!r} (family {config.family(g)!r}) is not in the corpus"
            )
    domains = set(df["domain"].astype(str))
    for d in config.held_out_domains:
        if d not in domains:
            out.append(f"held-out domain {d!r} is not in the corpus")
    return out


def _corpus_version(corpus_path: str | Path | None) -> str | None:
    if corpus_path is None:
        return None
    manifest = Path(corpus_path).with_name("manifest.json")
    try:
        with open(manifest, encoding="utf-8") as fh:
            return json.load(fh).get("corpus_version")
    except (OSError, ValueError, AttributeError):
        return None


def _memberships_sha256(memberships: pd.DataFrame) -> str:
    blob = (memberships["id"].astype(str) + "\t" + memberships["split"].astype(str)).str.cat(
        sep="\n"
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


@dataclass
class SplitsResult:
    out_dir: Path
    parquet_path: Path
    manifest_path: Path
    summary_path: Path
    splits_version: str
    manifest: dict[str, Any]
    summary_markdown: str


def write_splits(
    df: pd.DataFrame,
    memberships: pd.DataFrame,
    config: SplitConfig,
    out_dir: str | Path,
    *,
    corpus_path: str | Path | None = None,
) -> SplitsResult:
    """Write splits.parquet, splits_manifest.json and splits.md to out_dir.

    `corpus_version` is read from manifest.json next to `corpus_path` when it
    exists, else it is null. `splits_version` is "s-" + 16 hex digits of a
    sha256 over the config hash and the (id, split) rows, so identical splits
    get the same version.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = split_summary(df, memberships)
    sizes = _split_sizes(summary)
    warnings = _warnings(df, sizes, config)
    cfg_hash = config.config_hash()
    content_sha = _memberships_sha256(memberships)
    splits_version = "s-" + hashlib.sha256(f"{cfg_hash}\n{content_sha}".encode()).hexdigest()[:16]
    families = _families(df, config)
    held_gens = sorted(
        set(df["generator"][(df["label"] == "ai") & families.isin(config.held_out_families)])
    )

    manifest: dict[str, Any] = {
        "splits_version": splits_version,
        "format_version": SPLITS_FORMAT_VERSION,
        "corpus_version": _corpus_version(corpus_path),
        "corpus_file": str(corpus_path) if corpus_path is not None else None,
        "n_corpus_rows": int(len(df)),
        "n_memberships": int(len(memberships)),
        "memberships_sha256": content_sha,
        "config_hash": cfg_hash,
        "config": config.to_dict(),
        "held_out_families": sorted(config.held_out_families),
        "held_out_generators_in_corpus": held_gens,
        "ood_generator_human_domains": _covered_domains(
            df,
            _membership_matrix(df, memberships)[:, _SPLIT_CODE["test_ood_generator"]]
            & (df["label"] == "ai").to_numpy(),
        ),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": list(SPLITS),
        "split_sizes": sizes,
        "warnings": warnings,
        "counts": summary.to_dict(orient="records"),
    }
    summary_md = render_summary(manifest)

    parquet_path = out_dir / "splits.parquet"
    manifest_path = out_dir / "splits_manifest.json"
    summary_path = out_dir / "splits.md"
    table = pa.Table.from_pandas(
        memberships[["id", "split"]].astype(str), schema=PARQUET_SCHEMA, preserve_index=False
    )
    # All three files go to *.tmp first and are renamed only once all are complete,
    # so an interrupted run never pairs a new splits.parquet with an old manifest.
    finals = (parquet_path, manifest_path, summary_path)
    tmps = [_tmp_path(p) for p in finals]
    try:
        pq.write_table(table, tmps[0], compression="zstd")
        tmps[1].write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=int) + "\n",
            encoding="utf-8",
        )
        tmps[2].write_text(summary_md, encoding="utf-8")
        for tmp, final in zip(tmps, finals):
            os.replace(tmp, final)
    except BaseException:
        for tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise
    return SplitsResult(
        out_dir=out_dir,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        splits_version=splits_version,
        manifest=manifest,
        summary_markdown=summary_md,
    )


def load_splits(path: str | Path) -> pd.DataFrame:
    """Read splits.parquet (or the directory holding it) as an (id, split) frame."""
    path = Path(path)
    if path.is_dir():
        path = path / "splits.parquet"
    return pd.read_parquet(path, columns=["id", "split"])


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_summary(manifest: Mapping[str, Any]) -> str:
    """Markdown summary of a splits manifest."""
    cfg = manifest["config"]
    sizes = manifest["split_sizes"]
    fr = cfg["fractions"]
    out = [
        f"# Splits {manifest['splits_version']}",
        "",
        f"- corpus: {manifest.get('corpus_version') or 'unknown version'} "
        f"({manifest['n_corpus_rows']} rows, {manifest['n_memberships']} memberships)",
        f"- config hash: {manifest['config_hash'][:16]}",
        f"- seed {cfg['seed']}; train/val/test = {fr['train']}/{fr['val']}/{fr['test']}; "
        f"test_short: n_words < {cfg['short_text_max_words']}",
        f"- held-out generators: {', '.join(cfg['held_out_generators']) or '(none)'} "
        f"(families {', '.join(manifest['held_out_families']) or '(none)'}; "
        f"in corpus: {', '.join(manifest['held_out_generators_in_corpus']) or '(none)'})",
        f"- held-out domains: {', '.join(cfg['held_out_domains']) or '(none)'}",
        "- test_ood_generator human texts: the `test` human rows from "
        f"{', '.join(manifest['ood_generator_human_domains']) or '(no domain)'} "
        "(also members of `test`)",
        "",
        "## Split sizes",
        "",
        _md_table(
            ["split", *LABELS, "total"],
            [[s, *(sizes[s][lab] for lab in LABELS), sizes[s]["total"]] for s in SPLITS],
        ),
    ]
    if manifest["warnings"]:
        out += ["", "## Warnings", ""]
        out += [f"- WARNING: {w}" for w in manifest["warnings"]]

    counts = pd.DataFrame(manifest["counts"], columns=["split", "label", "generator", "domain",
                                                       "count"])
    for dim in ("generator", "domain"):
        out += ["", f"## {dim.capitalize()} x split", ""]
        if counts.empty:
            out.append("(empty)")
            continue
        pivot = counts.pivot_table(
            index=dim, columns="split", values="count", aggfunc="sum", fill_value=0
        ).reindex(columns=list(SPLITS), fill_value=0)
        rows = [[name, *(int(v) for v in pivot.loc[name])] for name in pivot.index]
        out.append(_md_table([dim, *SPLITS], rows))
    return "\n".join(out) + "\n"
