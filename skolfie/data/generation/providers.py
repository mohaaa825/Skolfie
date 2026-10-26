"""
skolfie/data/generation/providers.py
-------------------------------------
A small, pluggable layer over the text-generation APIs used to make fresh
AI-written text.

Every provider satisfies the `Provider` protocol: a `model_name` attribute
and `generate(prompt, *, temperature, max_tokens) -> str`. The HTTP
providers call the vendors' REST APIs directly with `requests` (no vendor
SDKs), retry 408/429/5xx responses and connection errors with exponential
backoff, and keep request/token counts in `provider.stats`.

    provider          env vars                                      default model
    anthropic         ANTHROPIC_API_KEY (ANTHROPIC_BASE_URL)         claude-opus-5-5
    openai            OPENAI_API_KEY (OPENAI_BASE_URL)               gpt-6-astra
    gemini            GEMINI_API_KEY (GEMINI_BASE_URL)               gemini-4-argon
    openai-compatible OPENAI_COMPATIBLE_BASE_URL,                    (--model required)
                      OPENAI_COMPATIBLE_API_KEY
    together / groq / ollama / vllm: openai-compatible with the base URL preset
    fake              none (deterministic, for tests)                fake-model

`generate` returns "" when the API answers but gives no usable text (a
refusal stop reason, a safety block, or output cut off at the token cap),
so the caller can count and drop it like any other rejected output.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-6-astra",
    "gemini": "gemini-4-argon",
    "fake": "fake-model",
}

# Base URLs for OpenAI-compatible servers. None means OPENAI_COMPATIBLE_BASE_URL.
OPENAI_COMPATIBLE_PRESETS: dict[str, str | None] = {
    "openai-compatible": None,
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}

PROVIDER_NAMES = ("anthropic", "openai", "gemini", *OPENAI_COMPATIBLE_PRESETS, "fake")

# Auth / permission / unknown-model errors: retrying or moving on cannot help.
FATAL_STATUSES = frozenset({401, 403, 404})


class ProviderError(RuntimeError):
    """An API call failed (after retries, for retryable errors)."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class ProviderConfigError(ValueError):
    """A provider cannot be built: missing API key, base URL or model."""


class TransientHTTPError(RuntimeError):
    """A connection error or timeout; always retried."""


@runtime_checkable
class Provider(Protocol):
    model_name: str

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str: ...


# --------------------------------------------------------------------------- #
# HTTP layer (monkeypatch `_http_post` in tests)
# --------------------------------------------------------------------------- #


@dataclass
class HTTPResponse:
    status: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}


def _http_post(url: str, headers: Mapping[str, str], payload: dict, timeout: float) -> HTTPResponse:
    """POST JSON and return the raw response; connection problems raise TransientHTTPError."""
    import requests  # imported lazily: only real API calls need it

    try:
        resp = requests.post(url, headers=dict(headers), json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise TransientHTTPError(f"{type(exc).__name__}: {exc}") from exc
    return HTTPResponse(resp.status_code, {k.lower(): v for k, v in resp.headers.items()},
                        resp.text)


def is_retryable(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP date; fall back to our own backoff


def backoff_delay(attempt: int, base: float, cap: float, retry_after: float | None = None) -> float:
    """Exponential backoff with up to 25% jitter; honours Retry-After (capped at 5 min)."""
    delay = min(cap, base * (2**attempt))
    delay += random.uniform(0, delay * 0.25)
    if retry_after is not None:
        delay = max(delay, min(retry_after, 300.0))
    return delay


def approx_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for cost estimates."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class UsageStats:
    """Thread-safe counters. Tokens are API-reported where the API reports them."""

    requests: int = 0  # HTTP requests sent, retries included
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, **counts: int) -> None:
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + int(value or 0))


# --------------------------------------------------------------------------- #
# HTTP providers
# --------------------------------------------------------------------------- #


class HTTPProvider:
    """Shared request/retry logic. Subclasses build the payload and parse the reply."""

    provider_name = "http"
    # Added to the requested max_tokens. For models that think before answering
    # (thinking/reasoning tokens count against the cap) this keeps the visible
    # answer from being cut off. It is a cap only; unused tokens are not billed.
    extra_max_tokens = 0

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        send_temperature: bool | None = None,
    ) -> None:
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.send_temperature = (
            self.accepts_temperature(model) if send_temperature is None else send_temperature
        )
        self.stats = UsageStats()

    # -- subclass hooks ---------------------------------------------------- #

    @staticmethod
    def accepts_temperature(model: str) -> bool:
        return True

    def endpoint(self) -> tuple[str, dict[str, str]]:
        raise NotImplementedError

    def payload(self, prompt: str, temperature: float | None, max_tokens: int) -> dict:
        raise NotImplementedError

    def extract_text(self, data: dict) -> str:
        raise NotImplementedError

    def usage(self, data: dict) -> tuple[int, int]:
        return 0, 0

    # -- main entry point -------------------------------------------------- #

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        cap = max_tokens + self.extra_max_tokens
        temp = temperature if self.send_temperature else None
        try:
            data = self.post(self.payload(prompt, temp, cap))
        except ProviderError as exc:
            # Some current models reject sampling parameters outright. Learn that
            # once and resend without them, instead of failing every request.
            if temp is None or exc.status != 400 or "temperature" not in exc.body.lower():
                raise
            self.send_temperature = False
            data = self.post(self.payload(prompt, None, cap))
        n_in, n_out = self.usage(data)
        self.stats.add(input_tokens=n_in, output_tokens=n_out)
        return self.extract_text(data).strip()

    def post(self, payload: dict) -> dict:
        url, headers = self.endpoint()
        error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            self.stats.add(requests=1)
            retry_after: float | None = None
            try:
                resp = _http_post(url, headers, payload, self.timeout)
            except TransientHTTPError as exc:
                error = ProviderError(f"{self.provider_name}: {exc}")
            else:
                if 200 <= resp.status < 300:
                    return resp.json()
                error = ProviderError(
                    f"{self.provider_name} API returned HTTP {resp.status}: {resp.text[:300]}",
                    status=resp.status,
                    body=resp.text,
                )
                if not is_retryable(resp.status):
                    self.stats.add(failures=1)
                    raise error
                retry_after = parse_retry_after(resp.headers.get("retry-after"))
            if attempt < self.max_retries:
                self.stats.add(retries=1)
                time.sleep(backoff_delay(attempt, self.backoff_base, self.backoff_max,
                                         retry_after))
        self.stats.add(failures=1)
        assert error is not None
        raise error


def _anthropic_version_tuple(model: str) -> tuple[int, int] | None:
    m = re.match(r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:-|$)", model)
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    m = re.match(r"claude-(\d+)(?:-(\d{1,2}))?-", model)  # legacy "claude-3-5-sonnet-..."
    if m:
        return int(m.group(1)), int(m.group(2) or 0)
    return None


class AnthropicProvider(HTTPProvider):
    """Claude via POST /v1/messages."""

    provider_name = "anthropic"
    extra_max_tokens = 4096

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        effort: str | None = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigError("set ANTHROPIC_API_KEY to use the anthropic provider")
        super().__init__(
            model or DEFAULT_MODELS["anthropic"],
            api_key=api_key,
            base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com",
            **kwargs,
        )
        self.effort = effort

    @staticmethod
    def accepts_temperature(model: str) -> bool:
        # Claude 4.7 and later reject temperature/top_p/top_k with a 400.
        version = _anthropic_version_tuple(model)
        return version is not None and version < (4, 7)

    def endpoint(self) -> tuple[str, dict[str, str]]:
        return f"{self.base_url}/v1/messages", {
            "x-api-key": self.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def payload(self, prompt: str, temperature: float | None, max_tokens: int) -> dict:
        body: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        return body

    def extract_text(self, data: dict) -> str:
        if data.get("stop_reason") in ("refusal", "max_tokens"):
            return ""
        return "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )

    def usage(self, data: dict) -> tuple[int, int]:
        u = data.get("usage") or {}
        return int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)


class OpenAIProvider(HTTPProvider):
    """OpenAI chat completions (POST /v1/chat/completions)."""

    provider_name = "openai"
    extra_max_tokens = 4096  # reasoning tokens count against max_completion_tokens
    token_param = "max_completion_tokens"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigError("set OPENAI_API_KEY to use the openai provider")
        super().__init__(
            model or DEFAULT_MODELS["openai"],
            api_key=api_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            **kwargs,
        )

    @staticmethod
    def accepts_temperature(model: str) -> bool:
        # Reasoning models only accept the default; anything else learns it from a 400.
        return re.match(r"(o\d|gpt-5)", model) is None

    def endpoint(self) -> tuple[str, dict[str, str]]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return f"{self.base_url}/chat/completions", headers

    def payload(self, prompt: str, temperature: float | None, max_tokens: int) -> dict:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            self.token_param: max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        return body

    def extract_text(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices or choices[0].get("finish_reason") in ("length", "content_filter"):
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content

    def usage(self, data: dict) -> tuple[int, int]:
        u = data.get("usage") or {}
        return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


class OpenAICompatibleProvider(OpenAIProvider):
    """Any server speaking the OpenAI chat-completions API: Together, Groq, Ollama, vLLM..."""

    provider_name = "openai-compatible"
    extra_max_tokens = 0
    token_param = "max_tokens"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        preset: str = "openai-compatible",
        **kwargs: Any,
    ) -> None:
        if not model:
            raise ProviderConfigError(
                f"--model is required for {preset} (e.g. a Llama or Mistral model id "
                "as the server names it)"
            )
        if preset not in OPENAI_COMPATIBLE_PRESETS:
            raise ProviderConfigError(f"unknown preset {preset!r}")
        url = base_url or OPENAI_COMPATIBLE_PRESETS[preset]
        if url is None:
            url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        if not url:
            raise ProviderConfigError(
                "set OPENAI_COMPATIBLE_BASE_URL (e.g. http://localhost:11434/v1) or pass --base-url"
            )
        api_key = api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        if not api_key and preset in ("together", "groq"):
            raise ProviderConfigError(f"set OPENAI_COMPATIBLE_API_KEY to use {preset}")
        self.provider_name = preset
        HTTPProvider.__init__(self, model, api_key=api_key, base_url=url, **kwargs)


def _gemini_model(model: str) -> str:
    return model[len("models/"):] if model.startswith("models/") else model


class GeminiProvider(HTTPProvider):
    """Gemini via the REST generateContent endpoint."""

    provider_name = "gemini"
    extra_max_tokens = 4096  # thinking tokens count against maxOutputTokens

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderConfigError("set GEMINI_API_KEY to use the gemini provider")
        super().__init__(
            _gemini_model(model or DEFAULT_MODELS["gemini"]),
            api_key=api_key,
            base_url=(base_url or os.environ.get("GEMINI_BASE_URL")
                      or "https://generativelanguage.googleapis.com/v1beta"),
            **kwargs,
        )

    def endpoint(self) -> tuple[str, dict[str, str]]:
        return f"{self.base_url}/models/{self.model_name}:generateContent", {
            "x-goog-api-key": self.api_key or "",
            "content-type": "application/json",
        }

    def payload(self, prompt: str, temperature: float | None, max_tokens: int) -> dict:
        config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            config["temperature"] = temperature
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    def extract_text(self, data: dict) -> str:
        if (data.get("promptFeedback") or {}).get("blockReason"):
            return ""
        candidates = data.get("candidates") or []
        if not candidates or candidates[0].get("finishReason") not in (None, "STOP"):
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def usage(self, data: dict) -> tuple[int, int]:
        u = data.get("usageMetadata") or {}
        out = int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0)
        return int(u.get("promptTokenCount") or 0), out


# --------------------------------------------------------------------------- #
# Fake provider (tests, dry plumbing checks)
# --------------------------------------------------------------------------- #

_FAKE_SENTENCES = (
    "The morning started slowly, with rain tapping against the kitchen window.",
    "Most people underestimate how much small habits shape a whole week.",
    "There is a simple reason for this, and it has little to do with luck.",
    "We tried three different approaches before one of them finally worked.",
    "Looking back, the warning signs were there from the very beginning.",
    "It is worth checking the settings twice before changing anything else.",
    "Nobody expected the meeting to run for almost two hours.",
    "The results were mixed, but the overall trend was hard to ignore.",
    "She folded the letter carefully and put it back in the drawer.",
    "In practice, the cheapest option often ends up costing the most.",
)


def fake_text(prompt: str, n_sentences: int = 8) -> str:
    """Deterministic filler text (~80 words) derived from a hash of the prompt."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return " ".join(_FAKE_SENTENCES[digest[i] % len(_FAKE_SENTENCES)] for i in range(n_sentences))


class FakeProvider:
    """A Provider that makes no network calls.

    `responses` may be None (deterministic `fake_text(prompt)`), a sequence of
    strings (returned in call order, cycling), or a callable `prompt -> str`.
    Every call is recorded in `calls`.
    """

    provider_name = "fake"

    def __init__(
        self,
        model_name: str = DEFAULT_MODELS["fake"],
        responses: Sequence[str] | Callable[[str], str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.stats = UsageStats()
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        with self._lock:
            idx = len(self.calls)
            self.calls.append({"prompt": prompt, "temperature": temperature,
                               "max_tokens": max_tokens})
        if self.responses is None:
            text = fake_text(prompt)
        elif callable(self.responses):
            text = self.responses(prompt)
        else:
            text = self.responses[idx % len(self.responses)]
        self.stats.add(requests=1, input_tokens=approx_tokens(prompt),
                       output_tokens=approx_tokens(text))
        return text


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def resolve_model(provider: str, model: str | None = None) -> str:
    """The model id a provider will use, without needing an API key."""
    if provider not in PROVIDER_NAMES:
        raise ProviderConfigError(f"unknown provider {provider!r}; choose from {PROVIDER_NAMES}")
    if model:
        return _gemini_model(model) if provider == "gemini" else model
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    raise ProviderConfigError(f"--model is required for provider {provider!r}")


def generator_id(model: str) -> str:
    """The `generator` value written to records: the exact model id, lowercased."""
    return model.strip().lower()


def make_provider(
    name: str,
    model: str | None = None,
    *,
    base_url: str | None = None,
    effort: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 5,
) -> Provider:
    """Build a provider by name, reading keys from the environment."""
    model = resolve_model(name, model)
    common: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
    if name == "fake":
        return FakeProvider(model)
    if name == "anthropic":
        return AnthropicProvider(model, base_url=base_url, effort=effort, **common)
    if name == "openai":
        return OpenAIProvider(model, base_url=base_url, **common)
    if name == "gemini":
        return GeminiProvider(model, base_url=base_url, **common)
    return OpenAICompatibleProvider(model, base_url=base_url, preset=name, **common)
