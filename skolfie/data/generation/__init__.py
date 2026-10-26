"""
skolfie/data/generation/__init__.py
------------------------------------
Fresh AI-text generation: a prompt bank (`prompts`) and a pluggable layer
over the model APIs (`providers`). The CLI that ties them together and
writes `data/raw/fresh/<generator>.jsonl` is `scripts/generate_ai_texts.py`.
"""

from __future__ import annotations

from skolfie.data.generation.prompts import (
    PROMPT_TEMPLATES,
    STYLES,
    TARGET_DOMAINS,
    PromptSpec,
    count_combinations,
    sample_prompts,
)
from skolfie.data.generation.providers import (
    DEFAULT_MODELS,
    PROVIDER_NAMES,
    FakeProvider,
    Provider,
    ProviderConfigError,
    ProviderError,
    generator_id,
    make_provider,
    resolve_model,
)

__all__ = [
    "DEFAULT_MODELS",
    "PROMPT_TEMPLATES",
    "PROVIDER_NAMES",
    "STYLES",
    "TARGET_DOMAINS",
    "FakeProvider",
    "PromptSpec",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "count_combinations",
    "generator_id",
    "make_provider",
    "resolve_model",
    "sample_prompts",
]
