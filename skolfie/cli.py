"""
skolfie/cli.py
---------------
Roadmap step 11: the `skolfie` command (also `python -m skolfie`).

    skolfie check PATH... [-]   check files (.txt, .md, .docx, .pdf), folders
                                (searched recursively for those) and, with
                                `-`, stdin; prints a table (or --format csv /
                                json, to --out FILE) and with --html DIR
                                writes one HTML report per input
    skolfie bundle              copy the models and reports into one portable
                                folder (`skolfie.service.make_bundle`)
    skolfie info                what the detector loaded, as JSON
    skolfie space build         write a Hugging Face Space folder for a bundle
                                (`skolfie.deploy.hf_space.build_space`)
    skolfie space publish       upload it as a Space; the token comes from
                                $HF_TOKEN, the app login from $SKOLFIE_APP_AUTH

The models come from `--bundle DIR`, or `--models-dir DIR [--reports-dir
DIR]`, else from the environment variables SKOLFIE_BUNDLE, or
SKOLFIE_MODELS_DIR [+ SKOLFIE_REPORTS_DIR]. Only the operator sets these:
loading a model file runs code.

Files are read with `skolfie.data.file_parsers` (size and content checks);
a file that cannot be read, or a text over the length limit, becomes an
error row and the rest are still checked. Exit code: 0 when every input was
checked, 1 when any failed, 2 for usage errors (bad arguments, no models).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TextIO

EXTENSIONS = (".txt", ".md", ".docx", ".pdf")
STDIN = "-"
ENV_BUNDLE = "SKOLFIE_BUNDLE"
ENV_MODELS_DIR = "SKOLFIE_MODELS_DIR"
ENV_REPORTS_DIR = "SKOLFIE_REPORTS_DIR"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_APP_AUTH = "SKOLFIE_APP_AUTH"
CSV_COLUMNS = ("file", "words", "label", "ai_likelihood_pct", "warnings", "warning_kinds",
               "headline", "error")
BATCH_SIZE = 16


class UsageError(Exception):
    """Bad arguments or configuration: exit code 2."""


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "models", f"default: ${ENV_BUNDLE}, else ${ENV_MODELS_DIR} (+ ${ENV_REPORTS_DIR})")
    group.add_argument("--bundle", type=Path, metavar="DIR",
                       help="A folder made by `skolfie bundle`.")
    group.add_argument("--models-dir", type=Path, metavar="DIR",
                       help="Folder with stacked.joblib, features.joblib, "
                            "transformer-chunks.joblib.")
    group.add_argument("--reports-dir", type=Path, metavar="DIR",
                       help="Folder with the reports (default: reports/ next to --models-dir).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skolfie",
        description="Skolfie: check English texts (e.g. student essays) for AI-generated "
                    "writing. Results are probabilistic; never use them as the sole basis "
                    "for a misconduct decision.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    check = sub.add_parser("check", help="Check files, folders or stdin.",
                           description="Check files (.txt, .md, .docx, .pdf), folders "
                                       "(recursively) or stdin (-).")
    check.add_argument("paths", nargs="+", metavar="PATH",
                       help="A file, a folder, or - for stdin.")
    _add_model_args(check)
    check.add_argument("--format", choices=("table", "csv", "json"), default="table",
                       help="Output format (default: table).")
    check.add_argument("--out", type=Path, metavar="FILE",
                       help="Write the output here instead of stdout.")
    check.add_argument("--html", type=Path, metavar="DIR",
                       help="Also write one HTML report per checked input into DIR.")

    bundle = sub.add_parser("bundle", help="Copy models and reports into a portable folder.")
    bundle.add_argument("--models-dir", type=Path, required=True, metavar="DIR")
    bundle.add_argument("--reports-dir", type=Path, metavar="DIR",
                        help="Default: reports/ next to --models-dir.")
    bundle.add_argument("--out", type=Path, required=True, metavar="DIR",
                        help="The bundle folder to write.")
    bundle.add_argument("--verdict", default="stacked", help="Verdict model name.")
    bundle.add_argument("--features", default="features", help="Feature model name.")
    bundle.add_argument("--chunk", default="transformer-chunks", help="Chunk model name.")

    info = sub.add_parser("info", help="Show what the detector loads, as JSON.")
    _add_model_args(info)

    space = sub.add_parser(
        "space", help="Put the web app on Hugging Face Spaces.",
        description="Build a Hugging Face Space folder for the models, then publish it.")
    space_sub = space.add_subparsers(dest="space_command", metavar="ACTION")
    space_sub.required = True
    build = space_sub.add_parser(
        "build", help="Write a ready-to-upload Space folder.",
        description="Write a Space folder (README.md, requirements.txt, app.py, bundle/) "
                    "pinned to the library versions installed here. Run it where the models "
                    "were trained, so the pickled models load on the Space.")
    _add_model_args(build)
    build.add_argument("--out", type=Path, required=True, metavar="DIR",
                       help="The Space folder to write (new, or made by an earlier build).")
    build.add_argument("--git-ref", required=True, metavar="REF",
                       help="Skolfie commit (or tag) the Space installs; it must be on GitHub.")
    build.add_argument("--repo-url", default=None, metavar="URL",
                       help="Skolfie's Git repository (default: the GitHub repository).")
    build.add_argument("--title", default="Skolfie", help="The Space's title.")
    build.add_argument("--public", action="store_true",
                       help="serve the site WITHOUT a login (anyone with the address can use it)")
    publish = space_sub.add_parser(
        "publish", help="Upload a Space folder to Hugging Face.",
        description=f"Create the Space if needed, set its {ENV_APP_AUTH} secret from "
                    f"${ENV_APP_AUTH} (user:password) when set, and upload the folder. The "
                    f"token (Write role) is read from ${ENV_HF_TOKEN}.")
    publish.add_argument("--dir", type=Path, required=True, metavar="DIR",
                         help="A folder made by `skolfie space build`.")
    publish.add_argument("--repo-id", required=True, metavar="USER/NAME",
                         help="The Space, e.g. yourname/skolfie.")
    publish.add_argument("--private", action="store_true",
                         help="create the Space as private (only your Hugging Face account "
                              "can open it)")
    return parser


def load_detector(args: argparse.Namespace, environ: Optional[dict[str, str]] = None) -> Any:
    """The Detector named by the arguments, else by the environment (UsageError if none)."""
    from skolfie.service import Detector

    env = os.environ if environ is None else environ
    if args.bundle is not None and args.models_dir is not None:
        raise UsageError("give --bundle or --models-dir, not both")
    if args.bundle is None and args.models_dir is None:
        if env.get(ENV_BUNDLE):
            args.bundle = Path(env[ENV_BUNDLE])
        elif env.get(ENV_MODELS_DIR):
            args.models_dir = Path(env[ENV_MODELS_DIR])
        else:
            raise UsageError(f"no models: give --bundle DIR or --models-dir DIR (or set "
                             f"{ENV_BUNDLE} or {ENV_MODELS_DIR})")
    if args.models_dir is not None and args.reports_dir is None and env.get(ENV_REPORTS_DIR):
        args.reports_dir = Path(env[ENV_REPORTS_DIR])
    try:
        if args.bundle is not None:
            return Detector.from_bundle(args.bundle)
        return Detector.from_dirs(args.models_dir, args.reports_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise UsageError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
Input = tuple[str, Callable[[], str]]


def _skip(path: Path) -> bool:
    """Hidden files / folders and Word lock files (~$name.docx)."""
    return any(part.startswith(".") for part in path.parts) or path.name.startswith("~$")


def read_file(path: Path) -> str:
    """The text of a .txt / .md / .docx / .pdf file (FileParseError if it cannot be read)."""
    from skolfie.data.file_parsers import (
        MAX_FILE_BYTES,
        FileParseError,
        extract_text_from_file,
        read_txt,
    )

    if not path.is_file():
        raise FileParseError("File not found.")
    if path.suffix.lower() != ".md":
        return extract_text_from_file(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise FileParseError("The file could not be opened.") from exc
    if size > MAX_FILE_BYTES:
        raise FileParseError(f"The file is too large ({size / (1024 * 1024):.1f} MB); the "
                             f"limit is {MAX_FILE_BYTES / (1024 * 1024):.0f} MB.")
    return read_txt(str(path))


def collect_inputs(paths: Sequence[str], stdin: TextIO) -> list[Input]:
    """(display name, reader) per input, in order; folders expand to their files, sorted."""
    inputs: list[Input] = []
    used_stdin = False
    for raw in paths:
        if raw == STDIN:
            if used_stdin:
                raise UsageError("- (stdin) can be given only once")
            used_stdin = True
            inputs.append(("<stdin>", stdin.read))
            continue
        path = Path(raw)
        if path.is_dir():
            found = sorted(p for p in path.rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSIONS
                           and not _skip(p.relative_to(path)))
            inputs += [(str(p), (lambda p=p: read_file(p))) for p in found]
        else:
            inputs.append((raw, (lambda p=path: read_file(p))))
    return inputs


# --------------------------------------------------------------------------- #
# Checking and output
# --------------------------------------------------------------------------- #
def _row(name: str, result: Optional[dict[str, Any]], error: Optional[str],
         text: Optional[str] = None) -> dict[str, Any]:
    return {"file": name, "error": error, "result": result, "text": text}


def check_inputs(detector: Any, inputs: Sequence[Input],
                 progress: Optional[Callable[[str], None]] = None) -> list[dict[str, Any]]:
    """One row per input: {"file", "error", "result", "text"} (error or result is None)."""
    from skolfie.service import MAX_TEXT_CHARS

    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, str]] = []

    def flush() -> None:
        if not pending:
            return
        results = detector.check_many([text for _, text in pending])
        for (i, text), result in zip(pending, results):
            rows[i] = _row(rows[i]["file"], result, None, text)
        pending.clear()
        if progress is not None:
            progress(f"checked {sum(r['result'] is not None for r in rows)} of {len(inputs)}")

    for name, read in inputs:
        try:
            text = read()
        except (OSError, ValueError) as exc:  # FileParseError is a ValueError
            message = str(exc) if not isinstance(exc, OSError) else (
                f"could not read the file ({exc.strerror or type(exc).__name__})")
            rows.append(_row(name, None, message))
            continue
        if len(text) > MAX_TEXT_CHARS:
            rows.append(_row(name, None, f"text is too long ({len(text):,} characters); the "
                                         f"limit is {MAX_TEXT_CHARS:,}"))
            continue
        rows.append(_row(name, None, None))
        pending.append((len(rows) - 1, text))
        if len(pending) >= BATCH_SIZE:
            flush()
    flush()
    return rows


def _pct(result: Optional[dict[str, Any]]) -> Optional[float]:
    if result is None or result.get("p_ai") is None:
        return None
    return round(100 * float(result["p_ai"]), 1)


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    from skolfie.service import DISCLAIMER

    header = ("file", "words", "label", "AI likelihood", "warnings")
    lines, errors = [], []
    for row in rows:
        res = row["result"]
        if res is None:
            lines.append((row["file"], "", "error", "", ""))
            errors.append(f"{row['file']}: {row['error']}")
            continue
        pct = _pct(res)
        lines.append((row["file"], str(res["n_words"]), res["label"],
                      "" if pct is None else f"{pct:.0f}%", str(len(res["warnings"]))))
    widths = [max([len(header[k])] + [len(r[k]) for r in lines]) for k in range(len(header))]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(cell.rjust(w) if k in (1, 3, 4) else cell.ljust(w)
                         for k, (cell, w) in enumerate(zip(cells, widths))).rstrip()

    text = [fmt(header), "  ".join("-" * w for w in widths)] + [fmt(r) for r in lines]
    if errors:
        text += ["", "Errors:"] + [f"  {e}" for e in errors]
    return "\n".join(text) + f"\n\n{DISCLAIMER}\n"


def format_csv(rows: Sequence[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        res = row["result"] or {}
        pct = _pct(row["result"])
        writer.writerow({
            "file": row["file"], "words": res.get("n_words", ""),
            "label": res.get("label", "error"),
            "ai_likelihood_pct": "" if pct is None else pct,
            "warnings": len(res.get("warnings") or []) if row["result"] else "",
            "warning_kinds": ";".join(w["kind"] for w in res.get("warnings") or []),
            "headline": res.get("headline", ""), "error": row["error"] or ""})
    return buf.getvalue()


def format_json(rows: Sequence[dict[str, Any]]) -> str:
    from skolfie.service import DISCLAIMER

    return json.dumps({"disclaimer": DISCLAIMER,
                       "results": [{"file": r["file"], "error": r["error"],
                                    "result": r["result"]} for r in rows]},
                      indent=2, allow_nan=False) + "\n"


FORMATTERS = {"table": format_table, "csv": format_csv, "json": format_json}


def _html_name(name: str, used: set[str]) -> str:
    stem = Path(name).name if name != "<stdin>" else "stdin"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "text"
    out, k = f"{stem}.html", 2
    while out in used:
        out, k = f"{stem}-{k}.html", k + 1
    used.add(out)
    return out


def write_html_reports(rows: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    from skolfie.explain.render import render_result_html

    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    written = []
    for row in rows:
        if row["result"] is None:
            continue
        path = out_dir / _html_name(row["file"], used)
        path.write_text(render_result_html(row["result"], row["text"] or "",
                                           title=f"Skolfie: {row['file']}"), encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> int:
    inputs = collect_inputs(args.paths, stdin)
    if not inputs:
        raise UsageError(f"no files to check (looked for {', '.join(EXTENSIONS)})")
    detector = load_detector(args)

    def progress(message: str) -> None:
        print(f"[skolfie] {message}", file=stderr)

    show = len(inputs) > BATCH_SIZE and getattr(stderr, "isatty", lambda: False)()
    rows = check_inputs(detector, inputs, progress if show else None)
    output = FORMATTERS[args.format](rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        stdout.write(output)
    if args.html is not None:
        written = write_html_reports(rows, args.html)
        print(f"[skolfie] {len(written)} HTML report(s) in {args.html}", file=stderr)
    failed = [r for r in rows if r["result"] is None]
    if failed:
        print(f"[skolfie] {len(failed)} of {len(rows)} input(s) could not be checked",
              file=stderr)
    return 1 if failed else 0


def cmd_bundle(args: argparse.Namespace, stdout: TextIO) -> int:
    from skolfie.service import make_bundle

    try:
        path = make_bundle(args.models_dir, args.reports_dir, args.out, verdict=args.verdict,
                           features=args.features, chunk=args.chunk)
    except (FileNotFoundError, ValueError) as exc:
        raise UsageError(str(exc)) from exc
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"Bundle written to {path.parent} ({len(manifest['sources'])} files).", file=stdout)
    for missing in manifest["missing"]:
        print(f"  missing (skipped): {missing}", file=stdout)
    return 0


def cmd_info(args: argparse.Namespace, stdout: TextIO) -> int:
    detector = load_detector(args)
    stdout.write(json.dumps(detector.info(), indent=2, allow_nan=False) + "\n")
    return 0


def cmd_space_build(args: argparse.Namespace, stdout: TextIO) -> int:
    import tempfile

    from skolfie.deploy.hf_space import DEFAULT_REPO_URL, build_space, collect_versions
    from skolfie.service import make_bundle

    if (args.bundle is None) == (args.models_dir is None):
        raise UsageError("give --bundle DIR or --models-dir DIR (not both)")
    if args.bundle is not None and args.reports_dir is not None:
        raise UsageError("--reports-dir goes with --models-dir, not --bundle")
    versions = collect_versions()
    options = dict(repo_url=args.repo_url or DEFAULT_REPO_URL, git_ref=args.git_ref,
                   versions=versions, title=args.title, public=args.public)
    try:
        if args.bundle is not None:
            out = build_space(args.bundle, args.out, **options)
        else:
            with tempfile.TemporaryDirectory(prefix="skolfie-bundle-") as tmp:
                make_bundle(args.models_dir, args.reports_dir, tmp)
                out = build_space(tmp, args.out, **options)
    except (FileNotFoundError, ValueError) as exc:
        raise UsageError(str(exc)) from exc
    pins = ", ".join(f"{k} {v}" for k, v in versions.items())
    print(f"Space folder written to {out}\n  pinned: {pins}", file=stdout)
    if args.public:
        print("  --public: the site will NOT ask for a login.", file=stdout)
    return 0


def cmd_space_publish(args: argparse.Namespace, stdout: TextIO, stderr: TextIO,
                      environ: Optional[dict[str, str]] = None) -> int:
    from skolfie.deploy.hf_space import publish_space, space_site_url

    env = os.environ if environ is None else environ
    token = (env.get(ENV_HF_TOKEN) or "").strip()
    if not token:
        raise UsageError(f"${ENV_HF_TOKEN} is not set: create a token with the Write role at "
                         f"https://huggingface.co/settings/tokens and run "
                         f"`export {ENV_HF_TOKEN}=...` (it is never printed or saved)")
    auth = env.get(ENV_APP_AUTH) or None
    try:
        page = publish_space(args.dir, args.repo_id, token, private=args.private, auth=auth)
    except (ValueError, ImportError) as exc:
        raise UsageError(str(exc)) from exc
    if auth is None:
        print(f"[skolfie] ${ENV_APP_AUTH} is not set: the Space's login secret was left as "
              "it was.", file=stderr)
    print(f"Published. Build logs: {page}\nWebsite: {space_site_url(args.repo_id)} "
          "(ready once the build finishes, usually 5-10 minutes the first time)", file=stdout)
    return 0


def main(argv: Optional[Sequence[str]] = None, *, stdin: Optional[TextIO] = None,
         stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None) -> int:
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) or a usage error (2)
        return int(exc.code or 0)
    try:
        if args.command == "check":
            return cmd_check(args, stdin, stdout, stderr)
        if args.command == "bundle":
            return cmd_bundle(args, stdout)
        if args.command == "space":
            if args.space_command == "build":
                return cmd_space_build(args, stdout)
            return cmd_space_publish(args, stdout, stderr)
        return cmd_info(args, stdout)
    except UsageError as exc:
        command = args.command + (f" {args.space_command}" if args.command == "space" else "")
        print(f"skolfie {command}: error: {exc}", file=stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
