"""
skolfie/data/generation/gemini_batch.py
---------------------------------------
Gemini Batch API run that writes long student essays (added 2026-11-20).

`BatchRunner` takes a `google.genai.Client` (or a test double with the same
calls), the seeded plan from `skolfie.data.generation.essay_plan`, and works in
rounds until `n_target` essays between `min_words` and `max_words` are kept:

1. Write one JSONL request per assignment (`request_line`): the shared system
   instruction (or a context cache that holds it), the assignment, minimal
   thinking, an output-token cap; no temperature (Gemini 3.x ignores it).
2. Upload the file, create a batch job, save its name in `state.json` before
   waiting, so a Colab disconnect never pays for the same round twice: a
   rerun picks the job up again.
3. Poll until the job ends, download the results, clean each essay (Markdown
   marks removed, wrappers and refusals dropped), count words. Essays in range
   are kept (`<out-dir>/<file>.jsonl`, fresh-source records). Too short, too
   long or cut off: the same assignment is asked once more with a note about
   the length; after that it is dropped and the next unused plan entry tops
   up.
4. Stop at `n_target`, at `max_rounds`, or before a round whose estimated
   cost would pass `max_usd` (real costs come from each result's token
   counts).

Prices are batch prices: `PRICES` (standard, USD per 1M tokens) times
`BATCH_DISCOUNT`. Check them on https://ai.google.dev/gemini-api/docs/pricing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from skolfie.data.generation.essay_plan import MAX_WORDS, MIN_WORDS, SYSTEM_INSTRUCTION
from skolfie.data.generation.markdown import strip_markdown
from skolfie.data.generation.prompts import PromptSpec
from skolfie.data.schema import count_words

MODEL = "gemini-3.5-flash-lite"
GENERATOR = "gemini-3.5-flash-lite"
# Standard prices, USD per 1M tokens (Gemini 3.5 Flash-Lite, ai.google.dev, Nov 2026). Output
# includes thinking tokens. The cached-input price is an assumption (10% of input, as for the
# other Gemini 3 models).
PRICES = {"input": 0.30, "output": 2.50, "cached_input": 0.03}
BATCH_DISCOUNT = 0.5
THINKING_TOKENS_EST = 150     # per essay at thinking level "minimal" (estimate only)
TOKENS_PER_WORD = 1.35
TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_EXPIRED"}


def request_line(key: str, text: str, *, system_instruction: str = SYSTEM_INSTRUCTION,
                 cache_name: Optional[str] = None, thinking_level: Optional[str] = "minimal",
                 max_output_tokens: int = 4096) -> dict:
    """One Batch API JSONL line: {"key": ..., "request": GenerateContentRequest}."""
    config: dict[str, Any] = {"max_output_tokens": max_output_tokens}
    if thinking_level:
        config["thinking_config"] = {"thinking_level": thinking_level}
    req: dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": text}]}],
                           "generation_config": config}
    if cache_name:
        req["cached_content"] = cache_name
    else:
        req["system_instruction"] = {"parts": [{"text": system_instruction}]}
    return {"key": key, "request": req}


def _get(d: Any, *names: str, default: Any = None) -> Any:
    """First present key among camelCase / snake_case spellings."""
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


@dataclass
class Result:
    key: str
    text: Optional[str]
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None


def parse_result_line(obj: dict) -> Result:
    """A batch output line -> Result. Thought parts are left out of the text."""
    key = str(obj.get("key", ""))
    if obj.get("error"):
        return Result(key, None, error=json.dumps(obj["error"])[:300])
    resp = obj.get("response") or {}
    cands = resp.get("candidates") or []
    usage_raw = _get(resp, "usageMetadata", "usage_metadata", default={}) or {}
    usage = {
        "prompt": int(_get(usage_raw, "promptTokenCount", "prompt_token_count", default=0)),
        "cached": int(_get(usage_raw, "cachedContentTokenCount", "cached_content_token_count",
                           default=0)),
        "output": int(_get(usage_raw, "candidatesTokenCount", "candidates_token_count",
                           default=0)),
        "thoughts": int(_get(usage_raw, "thoughtsTokenCount", "thoughts_token_count",
                             default=0)),
    }
    if not cands:
        block = _get(resp, "promptFeedback", "prompt_feedback", default={})
        return Result(key, None, "NO_CANDIDATE", usage, error=json.dumps(block)[:300] or None)
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    finish = str(_get(cand, "finishReason", "finish_reason", default=""))
    return Result(key, text, finish, usage)


def usage_cost(usage: dict, prices: dict = PRICES, discount: float = BATCH_DISCOUNT) -> float:
    cached = usage.get("cached", 0)
    fresh_in = max(usage.get("prompt", 0) - cached, 0)
    out = usage.get("output", 0) + usage.get("thoughts", 0)
    return discount * (fresh_in * prices["input"] + cached * prices["cached_input"]
                       + out * prices["output"]) / 1e6


def estimate_cost(specs: list[PromptSpec], system_tokens: int, cached: bool = False,
                  prices: dict = PRICES, discount: float = BATCH_DISCOUNT) -> float:
    """Upper-end estimate in USD for one batch of `specs`."""
    total = 0.0
    for s in specs:
        usage = {"prompt": system_tokens + len(s.text) // 4 + 8,
                 "cached": system_tokens if cached else 0,
                 "output": int(min(s.target_words * 1.1, MAX_WORDS) * TOKENS_PER_WORD),
                 "thoughts": THINKING_TOKENS_EST}
        total += usage_cost(usage, prices, discount)
    return total


def length_retry_text(spec: PromptSpec, words: int) -> str:
    side = "short" if words < MIN_WORDS else "long"
    return (f"{spec.text} Your last draft was {words} words, which is too {side}: this time it "
            f"must be between {MIN_WORDS} and {MAX_WORDS} words, about {spec.target_words}.")


class BatchRunner:
    """Rounds of batch jobs until `n_target` essays are kept. All state lives in `work_dir`
    (state.json, requests/results per round, rejected.jsonl); kept essays go to `out_path`."""

    def __init__(self, client: Any, plan: list[PromptSpec], *, work_dir: Path, out_path: Path,
                 clean: Callable[[str], tuple[Optional[str], str]],
                 make_record: Callable[[str, PromptSpec, str], dict],
                 n_target: int = 1000, model: str = MODEL, generator: str = GENERATOR,
                 min_words: int = MIN_WORDS, max_words: int = MAX_WORDS,
                 max_usd: float = 10.0, max_rounds: int = 4, max_attempts: int = 2,
                 spare_first: float = 0.10, spare_later: float = 0.30,
                 thinking_level: Optional[str] = "minimal", max_output_tokens: int = 4096,
                 use_cache: bool = True, cache_ttl_hours: int = 48,
                 poll_seconds: int = 60, max_wait_hours: float = 30.0,
                 log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep):
        self.client, self.plan = client, plan
        self.work_dir, self.out_path = Path(work_dir), Path(out_path)
        self.clean, self.make_record = clean, make_record
        self.n_target, self.model, self.generator = n_target, model, generator
        self.min_words, self.max_words = min_words, max_words
        self.max_usd, self.max_rounds, self.max_attempts = max_usd, max_rounds, max_attempts
        self.spare_first, self.spare_later = spare_first, spare_later
        self.thinking_level, self.max_output_tokens = thinking_level, max_output_tokens
        self.use_cache, self.cache_ttl_hours = use_cache, cache_ttl_hours
        self.poll_seconds, self.max_wait_hours = poll_seconds, max_wait_hours
        self.log, self.sleep = log, sleep
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "state.json"
        self.state = self._load_state()

    # ---- state ---------------------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("model") != self.model:
                raise ValueError(f"{self.state_path} belongs to model {state.get('model')!r}; "
                                 "use another work dir")
            return state
        return {"model": self.model, "n_target": self.n_target, "rounds": [], "spent_usd": 0.0,
                "next_plan_index": 0, "attempts": {}, "last_words": {}, "kept": [],
                "dropped": [], "requeue": [], "cache_name": None, "system_tokens": None}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    @property
    def kept(self) -> int:
        return len(self.state["kept"])

    # ---- setup ---------------------------------------------------------------------------
    def system_tokens(self) -> int:
        if self.state["system_tokens"] is None:
            try:
                resp = self.client.models.count_tokens(model=self.model,
                                                       contents=SYSTEM_INSTRUCTION)
                self.state["system_tokens"] = int(resp.total_tokens)
            except Exception as exc:  # noqa: BLE001 - an estimate is enough
                self.log(f"count_tokens failed ({exc}); estimating from characters")
                self.state["system_tokens"] = len(SYSTEM_INSTRUCTION) // 4
            self._save()
        return self.state["system_tokens"]

    def setup_cache(self) -> Optional[str]:
        """Create a context cache for the system instruction, or explain why not."""
        if not self.use_cache:
            return None
        if self.state.get("cache_name"):
            return self.state["cache_name"]
        n = self.system_tokens()
        try:
            cache = self.client.caches.create(
                model=self.model,
                config={"system_instruction": SYSTEM_INSTRUCTION,
                        "ttl": f"{self.cache_ttl_hours * 3600}s",
                        "display_name": "skolfie-essay-system"})
        except Exception as exc:  # noqa: BLE001
            self.log(f"Context cache NOT used: the shared instruction is {n} tokens and the API "
                     f"refused to cache it ({str(exc)[:200]}). Most likely it is below the "
                     "model's minimum cache size. Every request carries the instruction "
                     "instead; it is a small part of the cost (see the estimate).")
            self.state["cache_name"] = None
            self.use_cache = False
            return None
        self.state["cache_name"] = cache.name
        self._save()
        self.log(f"Context cache {cache.name} holds the {n}-token shared instruction.")
        return cache.name

    # ---- rounds --------------------------------------------------------------------------
    def _spec_for(self, idx: int) -> tuple[str, str]:
        """(key, prompt text) for the next attempt at plan entry `idx`."""
        attempt = self.state["attempts"].get(str(idx), 0)
        spec = self.plan[idx]
        text = spec.text
        if attempt and str(idx) in self.state["last_words"]:
            text = length_retry_text(spec, self.state["last_words"][str(idx)])
        return f"{idx}.a{attempt}", text

    def _retry_candidates(self) -> list[int]:
        busy = set(self.state["kept"]) | set(self.state["dropped"])
        return sorted(int(i) for i, a in self.state["attempts"].items()
                      if int(i) not in busy and 0 < a < self.max_attempts)

    def _next_round(self) -> Optional[dict]:
        need = self.n_target - self.kept
        if need <= 0:
            return None
        spare = self.spare_first if not self.state["rounds"] else self.spare_later
        want = math.ceil(need * (1 + spare))
        requeue = self.state.setdefault("requeue", [])
        idxs = (requeue + [i for i in self._retry_candidates() if i not in requeue])[:want]
        while len(idxs) < want and self.state["next_plan_index"] < len(self.plan):
            idxs.append(self.state["next_plan_index"])
            self.state["next_plan_index"] += 1
        if not idxs:
            self.log("The plan is used up; make it longer to top up further.")
            return None
        specs = [self.plan[i] for i in idxs]
        cached = bool(self.state.get("cache_name"))
        est = estimate_cost(specs, self.system_tokens(), cached=cached)
        left = self.max_usd - self.state["spent_usd"]
        if est > left:
            per = est / len(idxs)
            fit = int(left // per) if per > 0 else 0
            if fit < max(1, min(need, 10)):
                self.log(f"Stopping: next round would cost about ${est:.2f}, but only "
                         f"${left:.2f} of MAX_USD={self.max_usd:.2f} is left.")
                self._unreserve(idxs)
                return None
            self._unreserve(idxs[fit:])
            idxs = idxs[:fit]
            est = per * fit
            self.log(f"Round shrunk to {fit} essays to stay under the cost cap.")
        n_round = len(self.state["rounds"]) + 1
        lines = []
        for i in idxs:
            key, text = self._spec_for(i)
            lines.append(request_line(key, text, cache_name=self.state.get("cache_name"),
                                      thinking_level=self.thinking_level,
                                      max_output_tokens=self.max_output_tokens))
            self.state["attempts"][str(i)] = self.state["attempts"].get(str(i), 0) + 1
        req_path = self.work_dir / f"round{n_round}_requests.jsonl"
        with req_path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.log(f"Round {n_round}: {len(lines)} requests "
                 f"({sum(1 for i in idxs if self.state['attempts'][str(i)] > 1)} length "
                 f"retries), estimated ${est:.2f}; spent so far ${self.state['spent_usd']:.2f}.")
        uploaded = self.client.files.upload(
            file=str(req_path),
            config={"display_name": f"skolfie-essays-round{n_round}", "mime_type": "jsonl"})
        job = self.client.batches.create(model=self.model, src=uploaded.name,
                                         config={"display_name":
                                                 f"skolfie-essays-round{n_round}"})
        self.state["requeue"] = [i for i in requeue if i not in idxs]
        rnd = {"n": n_round, "job": job.name, "idxs": idxs, "requests": len(lines),
               "estimate_usd": est,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "done": False}
        self.state["rounds"].append(rnd)
        self._save()
        self.log(f"Submitted batch job {job.name}.")
        return rnd

    def _unreserve(self, idxs: list[int]) -> None:
        """Give back plan entries that were taken for a round that will not run."""
        fresh = [i for i in idxs if self.state["attempts"].get(str(i), 0) == 0]
        if fresh:
            self.state["next_plan_index"] = min(min(fresh), self.state["next_plan_index"])

    def _wait(self, rnd: dict) -> Any:
        deadline = time.monotonic() + self.max_wait_hours * 3600
        last = None
        while True:
            job = self.client.batches.get(name=rnd["job"])
            state = getattr(job.state, "name", str(job.state))
            if state != last:
                self.log(f"{time.strftime('%H:%M:%S')}  {rnd['job']}: {state}")
                last = state
            if state in TERMINAL_STATES:
                return job, state
            if time.monotonic() > deadline:
                raise TimeoutError(f"{rnd['job']} still {state}; run again later to keep "
                                   "waiting (nothing is resubmitted).")
            self.sleep(self.poll_seconds)

    def _collect(self, rnd: dict, job: Any) -> None:
        raw = self.client.files.download(file=job.dest.file_name)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        res_path = self.work_dir / f"round{rnd['n']}_results.jsonl"
        res_path.write_text(raw, encoding="utf-8")
        counts = {"kept": 0, "short": 0, "long": 0, "truncated": 0, "failed": 0, "extra": 0}
        spent = 0.0
        with self.out_path.open("a", encoding="utf-8") as out, \
                (self.work_dir / "rejected.jsonl").open("a", encoding="utf-8") as rej, \
                (self.work_dir / "extra.jsonl").open("a", encoding="utf-8") as extra:
            for line in raw.splitlines():
                if not line.strip():
                    continue
                res = parse_result_line(json.loads(line))
                spent += usage_cost(res.usage)
                idx = int(res.key.split(".a")[0])
                spec = self.plan[idx]
                reason, words, text = self._judge(res)
                if reason == "ok":
                    rec = self.make_record(self.generator, spec, text)
                    if self.kept < self.n_target:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        self.state["kept"].append(idx)
                        counts["kept"] += 1
                    else:
                        extra.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["extra"] += 1
                    continue
                counts["short" if reason == "too_short" else
                       "long" if reason == "too_long" else
                       "truncated" if reason == "truncated" else "failed"] += 1
                rej.write(json.dumps({"key": res.key, "reason": reason, "words": words,
                                      "finish": res.finish_reason,
                                      "error": res.error}) + "\n")
                if words:
                    self.state["last_words"][str(idx)] = words
                if self.state["attempts"].get(str(idx), 0) >= self.max_attempts:
                    self.state["dropped"].append(idx)
        self.state["spent_usd"] += spent
        rnd.update(done=True, cost_usd=round(spent, 4), counts=counts)
        self._save()
        self.log(f"Round {rnd['n']}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
                 + f"; cost ${spent:.2f}; kept {self.kept}/{self.n_target}.")

    def _judge(self, res: Result) -> tuple[str, int, Optional[str]]:
        if res.error or res.text is None:
            return "failed", 0, None
        text = strip_markdown(res.text)
        words = count_words(text)
        if res.finish_reason.upper() in ("MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"):
            return "truncated", words, None
        cleaned, why = self.clean(text)
        if cleaned is None:
            return why, words, None
        words = count_words(cleaned)
        if words < self.min_words:
            return "too_short", words, None
        if words > self.max_words:
            return "too_long", words, None
        return "ok", words, cleaned

    def run(self) -> dict:
        """Finish any open round, then submit rounds until done. Returns the state."""
        self.setup_cache()
        while True:
            open_rounds = [r for r in self.state["rounds"] if not r["done"]]
            for rnd in open_rounds:
                job, state = self._wait(rnd)
                if state != "JOB_STATE_SUCCEEDED":
                    rnd.update(done=True, failed_state=state, cost_usd=0.0)
                    self._save()
                    self.log(f"Round {rnd['n']} ended as {state}; its essays are asked again "
                             "in the next round.")
                    for i in rnd.get("idxs", []):
                        a = self.state["attempts"].get(str(i), 0)
                        self.state["attempts"][str(i)] = max(a - 1, 0)
                        if i not in self.state["requeue"]:
                            self.state["requeue"].append(i)
                    self._save()
                    continue
                self._collect(rnd, job)
            if self.kept >= self.n_target:
                self.log(f"Done: {self.kept} essays in {self.out_path}.")
                break
            if len(self.state["rounds"]) >= self.max_rounds:
                self.log(f"Stopped after {self.max_rounds} rounds with {self.kept} essays; "
                         "raise max_rounds and run again to top up.")
                break
            if self._next_round() is None:
                break
        self._save()
        return self.state
