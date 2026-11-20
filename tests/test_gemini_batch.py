"""Tests for the Gemini Batch essay plan and runner (no network: a fake client)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from skolfie.data.generation import gemini_batch as gb
from skolfie.data.generation.essay_plan import (
    GENRES,
    MAX_WORDS,
    MIN_WORDS,
    SYSTEM_INSTRUCTION,
    capacity,
    essay_plan,
)
from skolfie.eval.splits import group_key

REPO_ROOT = Path(__file__).resolve().parent.parent


def _gen():
    name = "skolfie_generate_ai_texts"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" /
                                                  "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---- plan --------------------------------------------------------------------------------

def test_plan_unique_pairs_balanced_and_prefix_stable() -> None:
    assert capacity() >= 1300
    plan = essay_plan(1000, seed=0)
    assert len(plan) == 1000
    assert len({s.template_id for s in plan}) == 1000          # no genre+topic twice
    genres = {s.template_id.split(":")[0] for s in plan}
    assert genres == {g.id for g in GENRES}
    assert essay_plan(1300, seed=0)[:1000] == plan
    assert essay_plan(1000, seed=1) != plan
    assert sum(s.prompt_id.startswith("persuade-") for s in plan) >= 60
    with pytest.raises(ValueError):
        essay_plan(capacity() + 1)


def test_plan_prompts_ask_for_the_length_and_fit_grades() -> None:
    for s in essay_plan(300):
        assert MIN_WORDS < s.target_words < MAX_WORDS
        assert f"between {MIN_WORDS} and {MAX_WORDS} words" in s.text
        genre = s.template_id.split(":")[0]
        if genre in ("lab-report", "research-paper", "literary-analysis", "critique"):
            assert "8th-grade" not in s.text
    assert "Markdown" in SYSTEM_INSTRUCTION


def test_plan_ids_group_by_topic_like_the_corpus() -> None:
    plan = essay_plan(400)
    gen = _gen()
    for s in plan[:200]:
        rec = gen.make_record("gemini-3.5-flash-lite", s, "word " * 900)
        key = group_key(rec["id"], "fresh")
        if s.prompt_id.startswith("persuade-"):
            name, grade = s.prompt_id.split(".")[:2]
            assert key == f"fresh:{name}.{grade}"
        else:
            topic = s.template_id.split(":")[1]
            assert key == f"fresh:gb.{topic}"


# ---- request / result lines ----------------------------------------------------------------

def test_request_line_shapes() -> None:
    line = gb.request_line("3.a0", "Write it.")
    req = line["request"]
    assert line["key"] == "3.a0"
    assert req["generation_config"]["thinking_config"] == {"thinking_level": "minimal"}
    assert "temperature" not in req["generation_config"]
    assert req["system_instruction"]["parts"][0]["text"] == SYSTEM_INSTRUCTION
    cached = gb.request_line("3.a0", "Write it.", cache_name="cachedContents/abc",
                             thinking_level=None)["request"]
    assert cached["cached_content"] == "cachedContents/abc"
    assert "system_instruction" not in cached
    assert "thinking_config" not in cached["generation_config"]


def test_parse_result_line_skips_thoughts_and_reads_usage() -> None:
    obj = {"key": "1.a0", "response": {
        "candidates": [{"content": {"parts": [{"text": "hmm", "thought": True},
                                              {"text": "Essay body."}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 500, "candidatesTokenCount": 1500,
                          "thoughtsTokenCount": 100}}}
    res = gb.parse_result_line(obj)
    assert res.text == "Essay body." and res.finish_reason == "STOP"
    assert res.usage == {"prompt": 500, "cached": 0, "output": 1500, "thoughts": 100}
    cost = gb.usage_cost(res.usage)
    assert cost == pytest.approx(0.5 * (500 * 0.30 + 1600 * 2.50) / 1e6)
    err = gb.parse_result_line({"key": "2.a0", "error": {"code": 500}})
    assert err.text is None and err.error


# ---- runner with a fake client -------------------------------------------------------------

class FakeClient:
    """Answers each request with an essay whose length depends on the attempt."""

    def __init__(self, words_for, cache_ok=False, fail_rounds=()):
        self.words_for, self.cache_ok, self.fail_rounds = words_for, cache_ok, set(fail_rounds)
        self.uploads, self.jobs, self.created = {}, {}, 0
        self.files = SimpleNamespace(upload=self._upload, download=self._download)
        self.batches = SimpleNamespace(create=self._create, get=self._get)
        self.caches = SimpleNamespace(create=self._cache)
        self.models = SimpleNamespace(count_tokens=lambda model, contents:
                                      SimpleNamespace(total_tokens=420))

    def _cache(self, model, config):
        if not self.cache_ok:
            raise RuntimeError("400 Cached content is too small")
        return SimpleNamespace(name="cachedContents/x")

    def _upload(self, file, config):
        name = f"files/{len(self.uploads)}"
        self.uploads[name] = Path(file).read_text(encoding="utf-8")
        return SimpleNamespace(name=name)

    def _create(self, model, src, config):
        self.created += 1
        name = f"batches/{self.created}"
        out = []
        for line in self.uploads[src].splitlines():
            req = json.loads(line)
            idx, attempt = req["key"].split(".a")
            n = self.words_for(int(idx), int(attempt))
            text = ("**Intro.** " + "word " * (n - 1)).strip()
            out.append(json.dumps({"key": req["key"], "response": {
                "candidates": [{"content": {"parts": [{"text": text}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 600, "candidatesTokenCount": int(n * 1.3),
                                  "thoughtsTokenCount": 50}}}))
        state = ("JOB_STATE_FAILED" if self.created in self.fail_rounds
                 else "JOB_STATE_SUCCEEDED")
        self.jobs[name] = SimpleNamespace(
            state=SimpleNamespace(name=state),
            dest=SimpleNamespace(file_name=f"out/{name}"), _out="\n".join(out))
        return SimpleNamespace(name=name)

    def _get(self, name):
        return self.jobs[name]

    def _download(self, file):
        return self.jobs[file.removeprefix("out/")]._out.encode()


def _runner(tmp_path, client, n=40, **kw):
    gen = _gen()
    plan = essay_plan(200)
    return gb.BatchRunner(client, plan, work_dir=tmp_path / "work",
                          out_path=tmp_path / "out" / "g.jsonl", clean=gen.clean_output,
                          make_record=gen.make_record, n_target=n, poll_seconds=0,
                          log=lambda m: None, sleep=lambda s: None, **kw)


def _kept(tmp_path):
    lines = (tmp_path / "out" / "g.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_runner_keeps_in_range_retries_and_tops_up(tmp_path) -> None:
    # every 5th entry is too short on the first try and fine on the retry; every 7th is
    # always too long (dropped after 2 attempts) -> needs top-ups.
    def words(idx, attempt):
        if idx % 7 == 0:
            return 1700
        if idx % 5 == 0 and attempt == 0:
            return 500
        return 1000
    client = FakeClient(words)
    state = _runner(tmp_path, client).run()
    recs = _kept(tmp_path)
    assert len(recs) == 40 == len(state["kept"])
    assert all(MIN_WORDS <= len(r["text"].split()) <= MAX_WORDS for r in recs)
    assert all("*" not in r["text"] for r in recs)
    assert len({r["id"] for r in recs}) == 40
    assert client.created >= 2
    assert any(i % 5 == 0 for i in state["kept"])            # retried and kept
    assert not any(i % 7 == 0 for i in state["kept"])
    assert all(i % 7 == 0 for i in state["dropped"])
    assert state["spent_usd"] > 0
    assert "too short" in (tmp_path / "work" / "round2_requests.jsonl").read_text()
    assert state["cache_name"] is None                       # refused -> inline instruction
    first = json.loads((tmp_path / "work" / "round1_requests.jsonl").read_text().splitlines()[0])
    assert "system_instruction" in first["request"]


def test_runner_uses_cache_when_accepted_and_resumes(tmp_path) -> None:
    client = FakeClient(lambda i, a: 900, cache_ok=True)
    r = _runner(tmp_path, client, n=10)
    state = r.run()
    assert state["cache_name"] == "cachedContents/x"
    first = json.loads((tmp_path / "work" / "round1_requests.jsonl").read_text().splitlines()[0])
    assert first["request"]["cached_content"] == "cachedContents/x"
    assert len(_kept(tmp_path)) == 10
    created = client.created
    _runner(tmp_path, client, n=10).run()        # rerun: nothing new is submitted
    assert client.created == created and len(_kept(tmp_path)) == 10


def test_runner_requeues_a_failed_round(tmp_path) -> None:
    client = FakeClient(lambda i, a: 900, fail_rounds={1})
    state = _runner(tmp_path, client, n=10).run()
    assert len(state["kept"]) == 10
    assert state["rounds"][0]["failed_state"] == "JOB_STATE_FAILED"
    assert sorted(state["kept"]) == list(range(10))           # same assignments, asked again


def test_runner_stops_at_cost_cap(tmp_path) -> None:
    client = FakeClient(lambda i, a: 900)
    state = _runner(tmp_path, client, n=40, max_usd=0.0001).run()
    assert client.created == 0 and state["kept"] == []


def test_script_dry_run(capsys) -> None:
    spec = importlib.util.spec_from_file_location("gemini_batch_script",
                                                  REPO_ROOT / "scripts" / "gemini_batch_essays.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out-dir", "x", "--work-dir", "y", "--dry-run", "--n", "100"]) == 0
    out = capsys.readouterr().out
    assert "Estimated cost of 100 essays" in out and "lab-report" in out
