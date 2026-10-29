"""
skolfie/service.py
-------------------
Roadmap step 11: the one entry point the web app, the CLI and (later) the
REST API use to check a text.

`Detector` loads the trained models and reports once and checks texts:

    Detector.from_dirs(models_dir, reports_dir)   the folders as Colab wrote
                                                   them (My Drive/skolfie/models
                                                   and .../reports)
    Detector.from_bundle(bundle_dir)               a portable folder made by
                                                   `make_bundle` (bundle.json)
    detector.check(text) / check_many(texts)       one result dict per text
    detector.info()                                what was loaded or missing

Files (names without .joblib can be changed with verdict= / features= / chunk=):

    models/stacked.joblib             verdict model (StackedModel), required
    models/features.joblib            LightGBM feature model (SHAP reasons),
                                      required (else the stacker's feature member)
    models/transformer-chunks.joblib  chunk scorer for Mixed + highlights;
                                      fallback transformer.joblib, else the
                                      verdict model (then no Mixed)
    reports/mixed-<chunk>/mixed_report.json     chunk rule + chunk size;
                                      fallback reports/mixed-transformer/
    reports/explain/reference.json    typical human / AI feature values
    reports/fairness/stacked/known_limitations.json   user-facing caveats

Missing optional files are listed in `info()["missing"]` and skipped (no
Mixed without a chunk rule, no "typical of AI writing" without reference
values, no fairness warnings without known limitations). A missing required
model raises FileNotFoundError naming the file and the notebook that makes it.

The StackedModel records its members' paths as they were at training time
(absolute Colab paths). A member path that does not exist is looked up by
its file name in models_dir; a bundle records the members' new names in
bundle.json. A model file that is a member is loaded only once
(`skolfie.explain.explainer.reuse_or_load`).

`check()` returns plain JSON-safe Python:

    label        "human" | "uncertain" | "ai" | "mixed" | "too_short"
    headline     the user-facing sentence for the label
    p_ai         the verdict model's P(ai), None for too_short
    n_words      whitespace-separated words
    verdict, chunks, highlights, top_features, feature_model, obfuscation
                 as `Explainer.explain` returns them (None / [] for
                 too_short, except obfuscation)
    reasons      the top features' plain-language reasons, in order
    notes        e.g. the feature model disagreeing with the verdict
    warnings     [{"kind", "message"}], kinds "short_text", "obfuscation",
                 "fairness" (only for ai / mixed), "language"
    disclaimer   DISCLAIMER

Trust boundary: model paths come from the operator only (CLI arguments,
environment variables), never from user input, because loading a joblib
file runs code. Texts are untrusted: longer than MAX_TEXT_CHARS is a
ValueError, before any model runs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

MAX_TEXT_CHARS = 100_000
SHORT_TEXT_WORDS = 150
MIN_TEXT_WORDS = 30
DISCLAIMER = (
    "AI-text detection is probabilistic and can be wrong, for example on short texts, "
    "heavily edited texts or writing by non-native English speakers. Never use this result "
    "as the sole basis for an academic misconduct decision; talk to the student and look at "
    "other evidence first."
)

LABELS = ("human", "uncertain", "ai", "mixed", "too_short")
HEADLINES = {
    "human": "Likely written by a human",
    "ai": "Likely AI-generated",
    "mixed": "Mixed: parts look AI-generated",
    "uncertain": "Uncertain: no clear signal either way",
    "too_short": f"Too short to judge (needs at least {MIN_TEXT_WORDS} words)",
}
WARNING_KINDS = ("short_text", "obfuscation", "fairness", "language")

BUNDLE_FILE = "bundle.json"
BUNDLE_FORMAT = "skolfie-bundle"
BUNDLE_VERSION = 1

DEFAULT_NAMES = {"verdict": "stacked", "features": "features", "chunk": "transformer-chunks"}
FALLBACK_CHUNK = "transformer"
MIXED_REPORT = Path("mixed_report.json")
REFERENCE = Path("explain") / "reference.json"
KNOWN_LIMITATIONS = Path("fairness") / "stacked" / "known_limitations.json"

# Which notebook produces a model, for the "file not found" messages.
NOTEBOOKS = {
    "stacked": "notebooks/stack.ipynb (scripts/stack.py --save-model)",
    "features": "notebooks/feature_baseline.ipynb (scripts/evaluate.py --model features)",
    "transformer": "notebooks/transformer.ipynb (scripts/evaluate.py --model transformer)",
    "transformer-chunks": "notebooks/chunks.ipynb (scripts/evaluate.py --model "
                          "transformer-chunks)",
}

# The language check: the share of these common English function words among
# the word tokens. English prose is well above LANGUAGE_MIN_SHARE (about 0.3 to
# 0.5), other languages well below. "i" is left out (Swedish "i" = "in").
ENGLISH_FUNCTION_WORDS = frozenset("""
the of and to a in is that it for was on are as with be by this have from or at not but
they which we you he she an their can has more there were been would will all what so if
about also when one these how our its than other some them my do because into only most
many should may very such those who his her no could any each then just than had does did
being where while why your us me him own same both few through over under between after
before again further once here up down out off above below during until against
""".split())
LANGUAGE_MIN_SHARE = 0.10
LANGUAGE_MIN_TOKENS = 20
LANGUAGE_MESSAGE = ("This text does not look like English. Skolfie is trained on English text "
                    "only, so this result is not reliable.")

_WORD = re.compile(r"\S+")
_LETTERS = re.compile(r"[^\W\d_]+")


def count_words(text: str) -> int:
    """Whitespace-separated words, as the word limits count them."""
    return sum(1 for _ in _WORD.finditer(text))


def english_share(text: str) -> tuple[float, int]:
    """(share of common English function words among the letter tokens, number of tokens)."""
    from skolfie.robust.normalize import normalize_for_scoring

    tokens = _LETTERS.findall(normalize_for_scoring(text).lower())
    if not tokens:
        return 0.0, 0
    return sum(t in ENGLISH_FUNCTION_WORDS for t in tokens) / len(tokens), len(tokens)


def looks_non_english(text: str) -> bool:
    """A cheap heuristic: too few common English function words (None for too few tokens)."""
    share, n = english_share(text)
    return n >= LANGUAGE_MIN_TOKENS and share < LANGUAGE_MIN_SHARE


def _warning(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": message}


def _basename(path: str) -> str:
    """The file name of a path recorded on any OS (Colab / Windows / Linux)."""
    return re.split(r"[\\/]", str(path))[-1]


def _hint(name: str) -> str:
    return NOTEBOOKS.get(name, "the notebook that trains it")


# --------------------------------------------------------------------------- #
# Which files to load
# --------------------------------------------------------------------------- #
@dataclass
class _Files:
    """The resolved files of a Detector (None: not available)."""

    source: str                    # "dirs" | "bundle"
    location: dict[str, str]       # models_dir / reports_dir, or bundle_dir
    models_dir: Path               # where member models are looked up by file name
    verdict: Path
    features: Optional[Path]
    chunk: Optional[Path]
    mixed_report: Optional[Path]
    reference: Optional[Path]
    known_limitations: Optional[Path]
    member_overrides: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Optional[str]]:
        return {k: None if getattr(self, k) is None else str(getattr(self, k))
                for k in ("verdict", "features", "chunk", "mixed_report", "reference",
                          "known_limitations")}


def _model_file(models_dir: Path, name: str) -> Path:
    return models_dir / (name if name.endswith(".joblib") else f"{name}.joblib")


def _resolve_dirs(models_dir: str | os.PathLike[str],
                  reports_dir: str | os.PathLike[str] | None,
                  names: dict[str, str]) -> _Files:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"models folder not found: {models_dir}")
    if reports_dir is None:
        reports_dir = models_dir.parent / "reports"
    reports_dir = Path(reports_dir)
    names = {**DEFAULT_NAMES, **{k: v for k, v in names.items() if v}}
    unknown = set(names) - set(DEFAULT_NAMES)
    if unknown:
        raise TypeError(f"unknown model name argument(s): {sorted(unknown)}")
    missing: list[str] = []
    notes: list[str] = []

    verdict = _model_file(models_dir, names["verdict"])
    if not verdict.is_file():
        raise FileNotFoundError(
            f"verdict model not found: {verdict}. Produce it with "
            f"{_hint(names['verdict'])} and copy it into {models_dir}.")

    features: Optional[Path] = _model_file(models_dir, names["features"])
    if not features.is_file():
        missing.append(str(features))
        notes.append(f"{features.name} not found; using the verdict model's feature member")
        features = None

    chunk_names = [names["chunk"]]
    if names["chunk"] == DEFAULT_NAMES["chunk"]:
        chunk_names.append(FALLBACK_CHUNK)
    chunk: Optional[Path] = None
    chunk_name = None
    for name in chunk_names:
        path = _model_file(models_dir, name)
        if path.is_file():
            chunk, chunk_name = path, name
            break
        missing.append(str(path))
    if chunk is None:
        notes.append("no chunk model: chunks are scored with the verdict model and no text "
                      "is called Mixed")
    elif chunk_name != names["chunk"]:
        notes.append(f"chunk model {names['chunk']} not found; using {chunk_name}")

    mixed_report = None
    if chunk_name is not None:
        candidates = [reports_dir / f"mixed-{chunk_name}" / MIXED_REPORT]
        if chunk_name != FALLBACK_CHUNK:
            candidates.append(reports_dir / f"mixed-{FALLBACK_CHUNK}" / MIXED_REPORT)
        for path in candidates:
            if path.is_file():
                mixed_report = path
                break
            missing.append(str(path))
        if mixed_report is None:
            notes.append("no mixed report: no text is called Mixed")
        elif mixed_report != candidates[0]:
            notes.append(f"chunk rule from {mixed_report} (tuned for {FALLBACK_CHUNK}, not "
                         f"{chunk_name})")

    optional = {}
    for key, rel in (("reference", REFERENCE), ("known_limitations", KNOWN_LIMITATIONS)):
        path = reports_dir / rel
        if path.is_file():
            optional[key] = path
        else:
            optional[key] = None
            missing.append(str(path))
    return _Files(source="dirs",
                  location={"models_dir": str(models_dir), "reports_dir": str(reports_dir)},
                  models_dir=models_dir, verdict=verdict, features=features, chunk=chunk,
                  mixed_report=mixed_report, missing=missing, notes=notes, **optional)


def _bundle_path(bundle_dir: Path, value: Any, what: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value or _basename(value) != value or value in (
            ".", ".."):
        raise ValueError(f"{bundle_dir / BUNDLE_FILE}: {what} must be a plain file name in "
                         f"the bundle folder, got {value!r}")
    return bundle_dir / value


def _resolve_bundle(bundle_dir: str | os.PathLike[str]) -> _Files:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / BUNDLE_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found: not a Skolfie bundle. Make one with "
            "`skolfie bundle --models-dir ... --reports-dir ... --out DIR`.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON ({exc})") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{manifest_path} is not a Skolfie bundle manifest")
    if int(manifest.get("version") or 0) > BUNDLE_VERSION:
        raise ValueError(f"{manifest_path} has bundle version {manifest.get('version')}; this "
                         f"Skolfie reads up to {BUNDLE_VERSION}")
    files = manifest.get("files") or {}
    paths = {k: _bundle_path(bundle_dir, files.get(k), f"files.{k}")
             for k in ("verdict", "features", "chunk", "mixed_report", "reference",
                       "known_limitations")}
    if paths["verdict"] is None:
        raise ValueError(f"{manifest_path} names no verdict model")
    missing = [str(p) for p in paths.values() if p is not None and not p.is_file()]
    if not paths["verdict"].is_file():
        raise FileNotFoundError(f"verdict model not found: {paths['verdict']} (named in "
                                f"{manifest_path}); remake the bundle")
    for key in ("features", "chunk", "mixed_report", "reference", "known_limitations"):
        if paths[key] is not None and not paths[key].is_file():
            paths[key] = None
    missing += [str(m) for m in manifest.get("missing") or [] if isinstance(m, str)]
    overrides = {str(name): _bundle_path(bundle_dir, value, f"members.{name}")
                 for name, value in (manifest.get("members") or {}).items()
                 if value is not None}
    return _Files(source="bundle", location={"bundle_dir": str(bundle_dir)},
                  models_dir=bundle_dir, member_overrides=overrides,
                  missing=missing, notes=[str(n) for n in manifest.get("notes") or []],
                  **paths)


def _member_paths(verdict_model: Any) -> Optional[dict[str, str]]:
    members = getattr(verdict_model, "members", None)
    if isinstance(members, dict) and callable(getattr(verdict_model, "_member", None)):
        return members
    return None


def _remap_members(verdict_model: Any, files: _Files) -> dict[str, str]:
    """Point the StackedModel's members at files that exist; FileNotFoundError if one is gone.

    Returns {name: path} as now set on the model.
    """
    members = _member_paths(verdict_model)
    if members is None:
        return {}
    for name, recorded in list(members.items()):
        if name in files.member_overrides:
            path = files.member_overrides[name]
            if not path.is_file():
                raise FileNotFoundError(
                    f"stacked model member {name!r} not found: {path}; remake the bundle")
            members[name] = str(path)
            continue
        if os.path.isfile(recorded):
            continue
        candidate = files.models_dir / _basename(recorded)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"the stacked model needs its member {name!r}, recorded at {recorded}, which "
                f"does not exist, and {candidate} does not exist either. Copy "
                f"{_basename(recorded)} into {files.models_dir} (it is made by "
                f"{_hint(Path(_basename(recorded)).stem)}).")
        members[name] = str(candidate)
    return dict(members)


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what} {path} could not be read ({exc})") from exc


def _limitations(data: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("known_limitations", data.get("items"))
    if not isinstance(data, list):
        raise ValueError(f"known limitations {path} must hold a list of items")
    return [item for item in data
            if isinstance(item, dict) and isinstance(item.get("message"), str)
            and item["message"].strip()]


def _versions() -> dict[str, Optional[str]]:
    from importlib import metadata

    out: dict[str, Optional[str]] = {}
    for dist in ("skolfie", "numpy", "scikit-learn", "lightgbm", "torch", "transformers"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class Detector:
    """Checks texts with the trained models (see the module docstring)."""

    def __init__(self, explainer: Any, known_limitations: Sequence[dict[str, Any]] = (),
                 info: Optional[dict[str, Any]] = None) -> None:
        self.explainer = explainer
        self.known_limitations = [dict(item) for item in known_limitations]
        self._info = dict(info or {})

    # -- loading ----------------------------------------------------------- #
    @classmethod
    def from_dirs(cls, models_dir: str | os.PathLike[str],
                  reports_dir: str | os.PathLike[str] | None = None, *,
                  verdict: str = "stacked", features: str = "features",
                  chunk: str = "transformer-chunks") -> "Detector":
        """Load from the models and reports folders (reports_dir: default ../reports)."""
        return cls._load(_resolve_dirs(models_dir, reports_dir,
                                       {"verdict": verdict, "features": features,
                                        "chunk": chunk}))

    @classmethod
    def from_bundle(cls, bundle_dir: str | os.PathLike[str]) -> "Detector":
        """Load a folder made by `make_bundle` (reads bundle_dir/bundle.json)."""
        return cls._load(_resolve_bundle(bundle_dir))

    @classmethod
    def _load(cls, files: _Files) -> "Detector":
        from functools import partial

        from skolfie.explain.explainer import (
            Explainer,
            find_feature_model,
            load_model,
            reuse_or_load,
        )
        from skolfie.mixed.chunking import chunk_spans
        from skolfie.mixed.detector import (
            DEFAULT_MIN_WORDS,
            DEFAULT_TARGET_WORDS,
            chunk_rule_from_params,
            read_mixed_report,
        )

        verdict_model = load_model(files.verdict)
        members = _remap_members(verdict_model, files)

        if files.features is not None:
            feature_model = reuse_or_load(verdict_model, files.verdict, files.features)
        else:
            feature_model = find_feature_model(verdict_model)
            if feature_model is None:
                raise FileNotFoundError(
                    f"feature model not found: {_model_file(files.models_dir, 'features')}, "
                    f"and the verdict model has no feature member. Produce it with "
                    f"{_hint('features')}.")

        chunk_model = (None if files.chunk is None
                       else reuse_or_load(verdict_model, files.verdict, files.chunk))
        rule, target_words, min_words = None, DEFAULT_TARGET_WORDS, DEFAULT_MIN_WORDS
        if chunk_model is not None and files.mixed_report is not None:
            report = read_mixed_report(files.mixed_report)
            rule = chunk_rule_from_params(report["rule"])
            target_words = report["target_words"] or target_words
            min_words = report["min_words"] or min_words

        reference = None
        if files.reference is not None:
            reference = _read_json(files.reference, "reference values")
            if not isinstance(reference, dict):
                raise ValueError(f"reference values {files.reference} must be a JSON object")
        limitations: list[dict[str, Any]] = []
        if files.known_limitations is not None:
            limitations = _limitations(_read_json(files.known_limitations,
                                                  "known limitations"),
                                       files.known_limitations)

        explainer = Explainer(verdict_model, feature_model, chunk_scorer=chunk_model,
                              chunk_rule=rule, reference=reference,
                              chunker=partial(chunk_spans, target_words=target_words,
                                              min_words=min_words))
        info = {
            "source": files.source,
            **files.location,
            "files": files.as_dict(),
            "members": members,
            "missing": list(dict.fromkeys(files.missing)),
            "notes": list(files.notes),
            "mixed_enabled": rule is not None,
            "chunking": {"target_words": int(target_words), "min_words": int(min_words)},
            "known_limitations": len(limitations),
        }
        return cls(explainer, limitations, info)

    def info(self) -> dict[str, Any]:
        """Which files were loaded or are missing, the cut-offs, the chunk rule, versions."""
        from skolfie.eval.report import _json_safe

        ex = self.explainer.info()
        return _json_safe({
            **self._info,
            "cutoffs": ex.get("cutoffs"),
            "chunk_rule": ex.get("chunk_rule"),
            "explainer": ex,
            "limits": {"max_text_chars": MAX_TEXT_CHARS, "min_text_words": MIN_TEXT_WORDS,
                       "short_text_words": SHORT_TEXT_WORDS},
            "versions": _versions(),
        })

    # -- checking ---------------------------------------------------------- #
    @staticmethod
    def _validate(text: Any) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is too long ({len(text):,} characters); the limit is "
                             f"{MAX_TEXT_CHARS:,}")
        return text

    def _too_short(self, text: str, n_words: int) -> dict[str, Any]:
        from skolfie.robust.normalize import obfuscation_note, obfuscation_stats

        stats = obfuscation_stats(text)
        note = obfuscation_note(stats)
        return {
            "label": "too_short", "headline": HEADLINES["too_short"], "p_ai": None,
            "n_words": n_words, "verdict": None, "chunks": [], "highlights": [],
            "top_features": [], "feature_model": None, "obfuscation": stats,
            "reasons": [], "notes": [],
            "warnings": [_warning("obfuscation", note)] if note else [],
            "disclaimer": DISCLAIMER,
        }

    def _result(self, text: str, n_words: int, ex: dict[str, Any]) -> dict[str, Any]:
        from skolfie.explain.render import disagreement_note
        from skolfie.robust.normalize import obfuscation_note

        label = str(ex["label"])
        warnings = []
        if n_words < SHORT_TEXT_WORDS:
            warnings.append(_warning(
                "short_text", f"Short text ({n_words} words): results are less reliable "
                              f"under {SHORT_TEXT_WORDS} words."))
        note = obfuscation_note(ex.get("obfuscation"))
        if note:
            warnings.append(_warning("obfuscation", note))
        if label in ("ai", "mixed"):
            warnings += [_warning("fairness", item["message"])
                         for item in self.known_limitations]
        if looks_non_english(text):
            warnings.append(_warning("language", LANGUAGE_MESSAGE))
        disagreement = disagreement_note(ex)
        top = list(ex.get("top_features") or [])
        return {
            "label": label,
            "headline": HEADLINES.get(label, label),
            "p_ai": ex.get("p_ai"),
            "n_words": n_words,
            "verdict": ex.get("verdict"),
            "chunks": ex.get("chunks") or [],
            "highlights": ex.get("highlights") or [],
            "top_features": top,
            "feature_model": ex.get("feature_model"),
            "obfuscation": ex.get("obfuscation"),
            "reasons": [str(f["reason"]) for f in top if f.get("reason")],
            "notes": [disagreement] if disagreement else [],
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    def check_many(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """One result per text, in order (ValueError if any text is over MAX_TEXT_CHARS)."""
        texts = [self._validate(t) for t in texts]
        n_words = [count_words(t) for t in texts]
        todo = [i for i, n in enumerate(n_words) if n >= MIN_TEXT_WORDS]
        explained = dict(zip(todo, self.explainer.explain_many([texts[i] for i in todo])))
        return [self._result(t, n, explained[i]) if i in explained else self._too_short(t, n)
                for i, (t, n) in enumerate(zip(texts, n_words))]

    def check(self, text: str) -> dict[str, Any]:
        """The result for one text (see the module docstring)."""
        return self.check_many([text])[0]


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
def make_bundle(models_dir: str | os.PathLike[str], reports_dir: str | os.PathLike[str] | None,
                out_dir: str | os.PathLike[str], **names: str) -> Path:
    """
    Copy the files a Detector needs into `out_dir` and write bundle.json; returns its path.

    `names` are verdict= / features= / chunk= as for `Detector.from_dirs`. The
    stacked model's member models are copied too and recorded in bundle.json
    by file name, so the folder can be moved anywhere and loaded with
    `Detector.from_bundle`. Missing optional files are listed under "missing".
    """
    from skolfie.explain.explainer import load_model, same_file

    files = _resolve_dirs(models_dir, reports_dir, names)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}  # bundle file name -> source

    def copy(src: Path, preferred: str) -> str:
        for name, prev in copied.items():
            if same_file(prev, src):
                return name
        name, stem, suffix, k = preferred, Path(preferred).stem, Path(preferred).suffix, 2
        while name in copied or name == BUNDLE_FILE:
            name, k = f"{stem}-{k}{suffix}", k + 1
        dest = out / name
        if not same_file(src, dest):
            shutil.copy2(src, dest)
        copied[name] = src
        return name

    entries: dict[str, Optional[str]] = {"verdict": copy(files.verdict, files.verdict.name)}
    verdict_model = load_model(files.verdict)  # a StackedModel loads its members lazily
    members = _remap_members(verdict_model, files)
    member_names = {name: copy(Path(path), _basename(path)) for name, path in members.items()}
    for key in ("features", "chunk"):
        path = getattr(files, key)
        entries[key] = None if path is None else copy(path, path.name)
    for key, preferred in (("mixed_report", "mixed_report.json"),
                           ("reference", "reference.json"),
                           ("known_limitations", "known_limitations.json")):
        path = getattr(files, key)
        entries[key] = None if path is None else copy(path, preferred)

    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "members": member_names,
        "sources": {name: str(src) for name, src in copied.items()},
        "missing": list(dict.fromkeys(files.missing)),
        "notes": files.notes,
    }
    path = out / BUNDLE_FILE
    tmp = out / (BUNDLE_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
