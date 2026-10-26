"""
tests/test_generation.py
-------------------------
Tests for fresh AI-text generation: the prompt bank, the provider layer
(with a fake HTTP layer, no network) and scripts/generate_ai_texts.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from skolfie.data.generation import providers
from skolfie.data.generation.prompts import (
    PROMPT_TEMPLATES,
    STYLES,
    TARGET_DOMAINS,
    count_combinations,
    sample_prompts,
    validate_bank,
)
from skolfie.data.generation.providers import (
    AnthropicProvider,
    FakeProvider,
    GeminiProvider,
    HTTPResponse,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
    ProviderConfigError,
    ProviderError,
    TransientHTTPError,
    make_provider,
)
from skolfie.data.schema import COLUMNS, DOMAINS, Record

REPO_ROOT = Path(__file__).resolve().parent.parent

BODY = (
    "The lighthouse had been dark for eleven years when the letter arrived. Mara turned it over "
    "twice before opening it, because the name on the envelope belonged to a man who had died "
    "long before she was born, and the postmark was from last Tuesday."
)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location(
        "skolfie_generate_script", REPO_ROOT / "scripts" / "generate_ai_texts.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY",
                "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("tests must not make HTTP requests")

    monkeypatch.setattr(providers, "_http_post", refuse)


# --------------------------------------------------------------------------- #
# Prompt bank
# --------------------------------------------------------------------------- #


def test_bank_is_consistent_and_large() -> None:
    validate_bank()
    assert {t["domain"] for t in PROMPT_TEMPLATES} >= set(TARGET_DOMAINS)
    assert {t["domain"] for t in PROMPT_TEMPLATES} <= set(DOMAINS)
    assert {s["kind"] for s in STYLES} >= {"plain", "persona", "humanlike"}
    assert count_combinations() >= 500


def test_sampling_is_deterministic() -> None:
    a = sample_prompts(200, seed=7)
    b = sample_prompts(200, seed=7)
    assert a == b
    assert sample_prompts(50, seed=7) == a[:50]
    assert sample_prompts(200, seed=8) != a


def test_sampling_covers_domains_styles_lengths() -> None:
    specs = sample_prompts(500, seed=0)
    assert len({s.prompt_id for s in specs}) == 500
    assert len({s.text for s in specs}) >= 400
    domains = [s.domain for s in specs]
    for d in TARGET_DOMAINS:
        assert domains.count(d) == 100
    assert set(domains) <= set(DOMAINS)
    assert {"review", "news"} <= set(domains)
    assert {s.style for s in specs} == {s["id"] for s in STYLES}
    assert min(s.target_words for s in specs) >= 100
    assert max(s.target_words for s in specs) <= 600
    assert len({s.temperature for s in specs}) > 1
    # every domain is present even in a tiny plan
    assert set(TARGET_DOMAINS) <= {s.domain for s in sample_prompts(10, seed=3)}
    for s in specs:
        assert ":" not in s.prompt_id
        assert f"about {s.target_words} words" in s.text
        assert s.max_tokens > s.target_words


def test_sampling_domain_filter_and_repeats() -> None:
    specs = sample_prompts(30, seed=1, domains=["technical"])
    assert {s.domain for s in specs} == {"technical"}
    with pytest.raises(ValueError):
        sample_prompts(5, domains=["legal"])
    # exhausting a tiny space repeats prompts with increasing sample_idx
    many = sample_prompts(3000, seed=0, domains=["casual"], max_redraws=0)
    keys = {(s.prompt_id, s.sample_idx) for s in many}
    assert len(keys) == len(many)


# --------------------------------------------------------------------------- #
# Post-filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [
    "Sure! Here's a short story about a lighthouse keeper:\n\n" + BODY,
    "Certainly! Here is your 300-word story.\n\n" + BODY,
    "Here is a short story about a lighthouse keeper and a letter:\n\n" + BODY,
    "Of course.\n\n" + BODY,
    "# The Last Letter\n\n" + BODY,
    "**The Last Letter**\n\n" + BODY,
    "Sure! Here's the story:\n\n---\n\n## The Last Letter\n\n" + BODY,
    "```\n" + BODY + "\n```",
    "<think>The user wants a story. Keep it short.</think>\n" + BODY,
    BODY + "\n\nWould you like me to make it longer or change the ending?",
    BODY + "\n\n---\n\nLet me know if you'd like me to adjust the tone of this story.",
])
def test_clean_output_strips_clear_wrappers(gen, raw: str) -> None:
    text, reason = gen.clean_output(raw)
    assert reason == "ok"
    assert text == BODY


@pytest.mark.parametrize("raw", [
    "Here's the thing about my grandmother: she never once cried in front of us. " + BODY,
    "Sure, I guess my landlord means well, but the heating has been off for a week. " + BODY,
    "I'm sorry, but I can't make it to Friday's meeting after all. " + BODY,
    "I can't help but think my roommate is doing this on purpose. " + BODY,
    BODY + "\n\nLet me know if Friday works for you.",
])
def test_clean_output_keeps_human_like_openings(gen, raw: str) -> None:
    text, reason = gen.clean_output(raw)
    assert reason == "ok"
    assert text == raw


def test_clean_output_keeps_inner_formatting(gen) -> None:
    raw = ("Here's a quick guide to backups:\n\n## Step 1\n\nPlug in the drive. " + BODY
           + "\n\n**Step 2**\n\nOpen the settings and choose the drive.")
    text, reason = gen.clean_output(raw)
    assert reason == "ok"
    assert text.startswith("## Step 1")
    assert "\nStep 2\n" in text and "**" not in text


@pytest.mark.parametrize("raw, reason", [
    ("", "empty"),
    ("   \n ", "empty"),
    (None, "empty"),
    ("I'm sorry, but I can't help with writing that.", "refusal"),
    ("I can't assist with that request. " + BODY, "refusal"),
    ("As an AI language model, I don't have personal experiences, but " + BODY, "refusal"),
    ("Sure! Here's a post:\n\nJust finished my first marathon. Legs are gone. Worth it.",
     "too_short"),
])
def test_clean_output_drops(gen, raw, reason) -> None:
    assert gen.clean_output(raw) == (None, reason)


# --------------------------------------------------------------------------- #
# End-to-end with FakeProvider
# --------------------------------------------------------------------------- #


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_fake_provider_end_to_end(gen, tmp_path: Path, no_network) -> None:
    provider = FakeProvider("Fake-Model-1")
    assert isinstance(provider, Provider)
    specs = sample_prompts(20, seed=0)
    out = gen.output_path(tmp_path, "fake-model-1")
    summary = gen.run_generation(provider, specs, out, concurrency=3, log=lambda m: None)
    assert summary.written == 20 and summary.failed == 0 and not summary.aborted
    assert len(provider.calls) == 20
    assert {c["temperature"] for c in provider.calls} <= {0.6, 0.8, 1.0}

    rows = _read(out)
    assert len(rows) == 20
    by_id = {s.prompt_id: s for s in specs}
    for row in rows:
        assert list(row) == COLUMNS
        rec = Record(**row)
        assert rec.label == "ai" and rec.source == "fresh"
        assert rec.generator == "fake-model-1"
        assert rec.n_words == len(rec.text.split()) >= 30
        prefix, generator, prompt_id, idx = row["id"].split(":")
        assert (prefix, generator, idx) == ("fresh", "fake-model-1", "0")
        assert row["domain"] == by_id[prompt_id].domain
        assert row["subdomain"] == by_id[prompt_id].subdomain
    assert len({r["id"] for r in rows}) == 20


def test_resume_skips_existing_and_drops_bad_outputs(gen, tmp_path: Path, no_network) -> None:
    specs = sample_prompts(12, seed=4)
    out = tmp_path / "fresh" / "fake-model.jsonl"
    first = gen.run_generation(FakeProvider(), specs[:5], out, log=lambda m: None)
    assert first.written == 5
    # simulate a crash in the middle of a write
    with out.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "fresh:fake-model:trunc')

    responses = ["Sure! Here's the post:\n\n" + BODY, "I'm sorry, but I can't help with that.",
                 "Too short."]
    provider = FakeProvider(responses=responses)
    second = gen.run_generation(provider, specs, out, concurrency=1, log=lambda m: None)
    assert second.skipped_existing == 5
    assert len(provider.calls) == 7
    assert second.written + sum(second.dropped.values()) == 7
    assert second.dropped["refusal"] >= 2 and second.dropped["too_short"] >= 2

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[5].startswith('{"id": "fresh:fake-model:trunc')  # kept, but on its own line
    rows = [json.loads(line) for i, line in enumerate(lines) if i != 5]
    assert len(rows) == 5 + second.written
    assert all(r["text"] == BODY for r in rows[5:])

    third_provider = FakeProvider()
    third = gen.run_generation(third_provider, specs, out, log=lambda m: None)
    assert third.skipped_existing == 5 + second.written
    assert len(third_provider.calls) == sum(second.dropped.values())


def test_cli_fake_run_and_resume(gen, tmp_path: Path, capsys, no_network) -> None:
    args = ["--provider", "fake", "--n", "15", "--seed", "2", "--out-dir", str(tmp_path),
            "--concurrency", "2"]
    assert gen.main(args) == 0
    out = tmp_path / "fake-model.jsonl"
    assert len(_read(out)) == 15
    assert "requests:   15 sent" in capsys.readouterr().out
    assert gen.main(args) == 0
    assert len(_read(out)) == 15
    assert "already done 15" in capsys.readouterr().out


def test_dry_run_calls_nothing(gen, tmp_path: Path, capsys, monkeypatch, no_network) -> None:
    def boom(*a, **k):
        raise AssertionError("dry run must not build a provider")

    monkeypatch.setattr(gen, "make_provider", boom)
    out_dir = tmp_path / "fresh"
    rc = gen.main(["--provider", "anthropic", "--n", "40", "--seed", "0",
                   "--out-dir", str(out_dir), "--dry-run", "--show", "3"])
    assert rc == 0
    assert not out_dir.exists()
    printed = capsys.readouterr().out
    assert "claude-opus-5-5" in printed
    assert "40 to request" in printed
    assert "requests:   40" in printed
    assert printed.count("[fresh:claude-opus-5-5:") == 3
    for d in TARGET_DOMAINS:
        assert f"{d}=" in printed


def test_dry_run_counts_existing(gen, tmp_path: Path, capsys) -> None:
    specs = sample_prompts(10, seed=0)
    out = gen.output_path(tmp_path, "fake-model")
    gen.run_generation(FakeProvider(), specs[:4], out, log=lambda m: None)
    gen.main(["--provider", "fake", "--n", "10", "--out-dir", str(tmp_path), "--dry-run"])
    assert "4 already in the output, 6 to request" in capsys.readouterr().out


def test_output_path_and_generator_naming(gen, tmp_path: Path) -> None:
    assert providers.generator_id("Claude-Opus-5-5") == "claude-opus-5-5"
    assert gen.output_path(tmp_path, "meta-llama/llama-3.3-70b-instruct-turbo").name == \
        "meta-llama_llama-3.3-70b-instruct-turbo.jsonl"
    assert gen.output_path(tmp_path, "llama3.3:70b").name == "llama3.3_70b.jsonl"
    assert providers.resolve_model("gemini", "models/gemini-3.8-flash") == "gemini-3.8-flash"


def test_cli_missing_key_and_model(gen, tmp_path: Path, capsys, no_network) -> None:
    assert gen.main(["--provider", "openai", "--n", "3", "--out-dir", str(tmp_path)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gen.main(["--provider", "openai-compatible", "--n", "3", "--dry-run"])


# --------------------------------------------------------------------------- #
# Providers with a fake HTTP layer
# --------------------------------------------------------------------------- #


class FakeHTTP:
    """Replays queued responses (HTTPResponse or exceptions) and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "payload": payload,
                              "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(body: dict) -> HTTPResponse:
    return HTTPResponse(200, {}, json.dumps(body))


def err(status: int, text: str = "error", headers: dict | None = None) -> HTTPResponse:
    return HTTPResponse(status, headers or {}, text)


ANTHROPIC_OK = {"content": [{"type": "thinking", "thinking": ""},
                            {"type": "text", "text": "  Hello there.  "}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 30}}


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", calls.append)
    return calls


def test_retries_with_backoff_then_succeeds(monkeypatch, sleeps) -> None:
    http = FakeHTTP(err(429), err(503), TransientHTTPError("reset"), ok(ANTHROPIC_OK))
    monkeypatch.setattr(providers, "_http_post", http)
    p = AnthropicProvider(api_key="k", max_retries=5, backoff_base=1.0, timeout=7)
    assert p.generate("hi", temperature=0.8, max_tokens=100) == "Hello there."
    assert len(http.requests) == 4
    assert len(sleeps) == 3
    for attempt, delay in enumerate(sleeps):
        assert 2**attempt <= delay <= 2**attempt * 1.25
    assert (p.stats.requests, p.stats.retries, p.stats.failures) == (4, 3, 0)
    assert (p.stats.input_tokens, p.stats.output_tokens) == (12, 30)
    assert all(r["timeout"] == 7 for r in http.requests)


def test_retry_after_header_is_honoured(monkeypatch, sleeps) -> None:
    http = FakeHTTP(err(429, headers={"retry-after": "9"}), ok(ANTHROPIC_OK))
    monkeypatch.setattr(providers, "_http_post", http)
    AnthropicProvider(api_key="k", backoff_base=0.5).generate("x", temperature=1, max_tokens=5)
    assert sleeps and sleeps[0] >= 9


def test_retries_give_up(monkeypatch, sleeps) -> None:
    http = FakeHTTP(*[err(500)] * 3)
    monkeypatch.setattr(providers, "_http_post", http)
    p = AnthropicProvider(api_key="k", max_retries=2, backoff_base=0.1, backoff_max=0.15)
    with pytest.raises(ProviderError) as info:
        p.generate("x", temperature=1, max_tokens=5)
    assert info.value.status == 500
    assert len(http.requests) == 3
    assert len(sleeps) == 2 and max(sleeps) <= 0.15 * 1.25
    assert p.stats.failures == 1


def test_client_errors_are_not_retried(monkeypatch, sleeps) -> None:
    http = FakeHTTP(err(401, "invalid x-api-key"))
    monkeypatch.setattr(providers, "_http_post", http)
    with pytest.raises(ProviderError) as info:
        AnthropicProvider(api_key="bad").generate("x", temperature=1, max_tokens=5)
    assert info.value.status == 401
    assert len(http.requests) == 1 and not sleeps


def test_anthropic_request_shape(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    http = FakeHTTP(ok(ANTHROPIC_OK), ok(ANTHROPIC_OK))
    monkeypatch.setattr(providers, "_http_post", http)

    p = make_provider("anthropic", effort="low")
    assert p.model_name == "claude-opus-5-5"
    p.generate("Write a poem.", temperature=0.6, max_tokens=300)
    req = http.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-test"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    body = req["payload"]
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": "Write a poem."}]
    assert body["max_tokens"] == 300 + AnthropicProvider.extra_max_tokens
    assert "temperature" not in body  # Claude 4.7+ rejects sampling parameters
    assert body["output_config"] == {"effort": "low"}

    old = make_provider("anthropic", "claude-sonnet-4-6")
    old.generate("x", temperature=0.6, max_tokens=10)
    assert http.requests[1]["payload"]["temperature"] == 0.6


def test_temperature_rejection_is_learned(monkeypatch, sleeps) -> None:
    http = FakeHTTP(err(400, '{"error": {"message": "temperature is not supported"}}'),
                    ok(ANTHROPIC_OK), ok(ANTHROPIC_OK))
    monkeypatch.setattr(providers, "_http_post", http)
    p = AnthropicProvider("claude-sonnet-4-6", api_key="k")
    assert p.generate("x", temperature=0.6, max_tokens=10) == "Hello there."
    p.generate("y", temperature=0.6, max_tokens=10)
    assert "temperature" in http.requests[0]["payload"]
    assert "temperature" not in http.requests[1]["payload"]
    assert "temperature" not in http.requests[2]["payload"]
    assert not sleeps


@pytest.mark.parametrize("data", [
    {"content": [{"type": "text", "text": ""}], "stop_reason": "refusal"},
    {"content": [{"type": "text", "text": "cut off mid"}], "stop_reason": "max_tokens"},
])
def test_anthropic_unusable_outputs_are_empty(monkeypatch, data) -> None:
    monkeypatch.setattr(providers, "_http_post", FakeHTTP(ok(data)))
    assert AnthropicProvider(api_key="k").generate("x", temperature=1, max_tokens=5) == ""


def test_openai_request_shape(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oa")
    reply = {"choices": [{"message": {"content": "Hi."}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    http = FakeHTTP(ok(reply), ok(reply))
    monkeypatch.setattr(providers, "_http_post", http)
    p = make_provider("openai")
    assert p.model_name == "gpt-6-astra"
    assert p.generate("x", temperature=0.8, max_tokens=100) == "Hi."
    req = http.requests[0]
    assert req["url"] == "https://api.openai.com/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer sk-oa"
    assert req["payload"]["max_completion_tokens"] == 100 + OpenAIProvider.extra_max_tokens
    assert req["payload"]["temperature"] == 0.8
    make_provider("openai", "gpt-5").generate("x", temperature=0.8, max_tokens=10)
    assert "temperature" not in http.requests[1]["payload"]
    assert p.stats.output_tokens == 2


def test_openai_compatible_presets(monkeypatch) -> None:
    reply = {"choices": [{"message": {"content": "Yo."}, "finish_reason": "stop"}]}
    http = FakeHTTP(ok(reply), ok(reply))
    monkeypatch.setattr(providers, "_http_post", http)

    with pytest.raises(ProviderConfigError):
        make_provider("openai-compatible", "llama3.3:70b")  # no base URL
    with pytest.raises(ProviderConfigError):
        make_provider("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo")  # no key
    with pytest.raises(ProviderConfigError):
        make_provider("ollama")  # no model

    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://gpu-box:8000/v1/")
    p = make_provider("openai-compatible", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    assert isinstance(p, OpenAICompatibleProvider)
    p.generate("x", temperature=1.0, max_tokens=50)
    req = http.requests[0]
    assert req["url"] == "http://gpu-box:8000/v1/chat/completions"
    assert "authorization" not in req["headers"]
    assert req["payload"]["max_tokens"] == 50 and req["payload"]["temperature"] == 1.0

    make_provider("ollama", "llama3.3:70b").generate("x", temperature=0.6, max_tokens=5)
    assert http.requests[1]["url"] == "http://localhost:11434/v1/chat/completions"


def test_gemini_request_shape(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    reply = {"candidates": [{"content": {"parts": [{"text": "plan", "thought": True},
                                                   {"text": "Answer."}]},
                             "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3,
                               "thoughtsTokenCount": 10}}
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}
    http = FakeHTTP(ok(reply), ok(blocked))
    monkeypatch.setattr(providers, "_http_post", http)
    p = make_provider("gemini")
    assert isinstance(p, GeminiProvider) and p.model_name == "gemini-4-argon"
    assert p.generate("x", temperature=0.6, max_tokens=100) == "Answer."
    req = http.requests[0]
    assert req["url"].endswith("/v1beta/models/gemini-4-argon:generateContent")
    assert req["headers"]["x-goog-api-key"] == "g-key"
    assert req["payload"]["generationConfig"]["temperature"] == 0.6
    assert req["payload"]["contents"][0]["parts"][0]["text"] == "x"
    assert (p.stats.input_tokens, p.stats.output_tokens) == (4, 13)
    assert p.generate("x", temperature=0.6, max_tokens=100) == ""


def test_fatal_status_aborts_run(gen, tmp_path: Path, monkeypatch, sleeps) -> None:
    http = FakeHTTP(*[err(401, "invalid x-api-key")] * 10)
    monkeypatch.setattr(providers, "_http_post", http)
    p = AnthropicProvider(api_key="bad")
    summary = gen.run_generation(p, sample_prompts(10, seed=0), tmp_path / "x.jsonl",
                                 concurrency=1, log=lambda m: None)
    assert summary.aborted and summary.written == 0
    assert len(http.requests) < 10
