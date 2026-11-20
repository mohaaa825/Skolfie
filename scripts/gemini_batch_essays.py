"""
scripts/gemini_batch_essays.py
------------------------------
Write `--n` long student essays (800-1,500 words) with Gemini 3.5 Flash-Lite
through the Gemini Batch API (half the normal token price), minimal thinking
and, when the API accepts it, a context cache for the shared instruction.

Assignments come from `skolfie.data.generation.essay_plan.essay_plan` (seeded:
genre x topic x grade x student voice, no genre+topic pair twice). See
`skolfie.data.generation.gemini_batch.BatchRunner` for the rounds, the length
check, the top-ups and the cost cap.

    GEMINI_API_KEY=... python scripts/gemini_batch_essays.py \\
        --out-dir "/content/drive/MyDrive/skolfie/data/raw/fresh_v3" \\
        --work-dir "/content/drive/MyDrive/skolfie/data/raw/gemini_batch_work" \\
        --n 1000 --max-usd 10

Resumable: run it again with the same --work-dir and it waits for the open job
instead of submitting a new one. `--dry-run` prints the plan summary and the
cost estimate without calling the API.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skolfie.data.generation import gemini_batch as gb  # noqa: E402
from skolfie.data.generation.essay_plan import (  # noqa: E402
    SYSTEM_INSTRUCTION,
    capacity,
    essay_plan,
)

OUT_FILE = "gemini-3.5-flash-lite-batch.jsonl"  # not the repo's gemini-3.5-flash-lite.jsonl


def _generate_module():
    name = "skolfie_generate_ai_texts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=gb.MODEL)
    p.add_argument("--max-usd", type=float, default=10.0)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--thinking-level", default="minimal",
                   help='"minimal" (default), "low", "medium", "high", or "" to leave it out')
    p.add_argument("--no-cache", action="store_true", help="do not try context caching")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-wait-hours", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    return p


def summarize_plan(plan) -> str:
    genres = Counter(s.template_id.split(":")[0] for s in plan)
    grades = Counter(s.style.split("+")[0] for s in plan)
    persuade = sum(s.prompt_id.startswith("persuade-") for s in plan)
    return ("genres: " + ", ".join(f"{g} {n}" for g, n in genres.most_common())
            + "\ngrades: " + ", ".join(f"{g} {n}" for g, n in sorted(grades.items()))
            + f"\nPERSUADE topics (shared with the human essays): {persuade}"
            + f"\ndistinct genre+topic pairs: {len({s.template_id for s in plan})}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan = essay_plan(capacity(), seed=args.seed)   # whole plan; rounds take it in order
    first = plan[: args.n]
    sys_tokens = len(SYSTEM_INSTRUCTION) // 4
    est = gb.estimate_cost(first, sys_tokens)
    print(summarize_plan(first))
    print(f"Plan has {len(plan)} assignments; the first {args.n} go out in round 1 "
          f"(+10% spare), the rest top up.")
    print(f"Estimated cost of {args.n} essays at batch prices: ${est:.2f} "
          f"(cap ${args.max_usd:.2f}).")
    if args.dry_run:
        print("\nExample assignment:\n" + first[0].text)
        return 0
    if est > args.max_usd:
        print("error: the estimate is above --max-usd", file=sys.stderr)
        return 2
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("error: set GEMINI_API_KEY", file=sys.stderr)
        return 2
    from google import genai  # pip install google-genai

    gen = _generate_module()
    runner = gb.BatchRunner(
        genai.Client(api_key=key), plan, work_dir=args.work_dir,
        out_path=args.out_dir / OUT_FILE, clean=gen.clean_output, make_record=gen.make_record,
        n_target=args.n, model=args.model, max_usd=args.max_usd, max_rounds=args.max_rounds,
        thinking_level=args.thinking_level or None, use_cache=not args.no_cache,
        poll_seconds=args.poll_seconds, max_wait_hours=args.max_wait_hours,
        log=lambda m: print(m, flush=True))
    try:
        state = runner.run()
    except TimeoutError as exc:
        print(exc)
        return 3
    print(f"Kept {len(state['kept'])}/{args.n}; spent ${state['spent_usd']:.2f}.")
    return 0 if len(state["kept"]) >= args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
