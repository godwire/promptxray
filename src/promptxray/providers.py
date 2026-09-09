"""Talk to a model.

Three backends, all over the standard library so the package has no runtime
dependencies:

  mock      - no network, no key. Answers with a tiny keyword rule so the whole
              tool can be tried out (and tested in CI) for free.
  openai    - any server that speaks the OpenAI chat format: OpenAI itself,
              Ollama, OpenRouter, Groq, vLLM, LM Studio.
  anthropic - the Anthropic messages API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Answer:
    """What came back from one model call."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    pass


class Provider:
    """Base class. A provider turns a prompt string into an Answer."""

    name = "base"

    def __init__(self, model: str, temperature: float = 0.0, timeout: int = 60) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str) -> Answer:  # pragma: no cover - interface
        raise NotImplementedError


class MockProvider(Provider):
    """A fake classifier so the tool runs without an API key.

    It is deliberately imperfect: it reacts to a few keywords and otherwise
    guesses. That gives a realistic mix of correct and wrong predictions, and
    it reacts to the prompt itself, so ablation produces visible deltas.
    """

    name = "mock"

    RULES = {
        "spam": ("free", "winner", "click here", "buy now", "prize", "offer", "$$$", "unsubscribe"),
        "urgent": ("asap", "urgent", "immediately", "right now", "emergency", "deadline today"),
    }

    def complete(self, prompt: str) -> Answer:
        lowered = prompt.lower()
        body = lowered.rsplit("text:", 1)[-1]

        label = "normal"
        for candidate, keywords in self.RULES.items():
            if any(k in body for k in keywords):
                label = candidate
                break

        # The mock reacts to instructions, so removing them changes the score.
        if label == "spam" and "ignore promotional wording in quotes" in lowered:
            if '"' in body:
                label = "normal"
        if label == "normal" and "treat all-caps messages as urgent" in lowered:
            raw = prompt.rsplit("Text:", 1)[-1]
            letters = [c for c in raw if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                label = "urgent"
        if "never answer urgent" in lowered and label == "urgent":
            label = "normal"

        return Answer(text=label, input_tokens=len(prompt) // 4, output_tokens=1)


class OpenAICompatibleProvider(Provider):
    """Works with OpenAI, Ollama, OpenRouter, Groq, vLLM, LM Studio."""

    name = "openai"

    # Every one of these speaks the same wire format, they just live at
    # different addresses and read a different environment variable.
    ENDPOINTS = {
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "ollama": ("http://localhost:11434/v1", None),
        "lmstudio": ("http://localhost:1234/v1", None),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    }

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 flavour: str = "openai", **kw):
        super().__init__(model, **kw)
        default_url, key_env = self.ENDPOINTS.get(flavour, self.ENDPOINTS["openai"])
        self.base_url = (base_url or os.getenv("PROMPTXRAY_BASE_URL") or default_url).rstrip("/")
        self.name = flavour
        self.api_key = (
            api_key
            or (os.getenv(key_env) if key_env else None)
            or os.getenv("OPENAI_API_KEY")
            or "not-needed"  # local servers ignore it
        )
        if key_env and self.api_key == "not-needed":
            raise ProviderError(
                f"Set {key_env} before using --provider {flavour}. "
                "For a run that costs nothing, use --provider ollama instead."
            )

    def complete(self, prompt: str) -> Answer:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 32,
        }
        data = _post(f"{self.base_url}/chat/completions", payload,
                     {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected response shape: {json.dumps(data)[:400]}") from exc
        usage = data.get("usage") or {}
        return Answer(
            text=text.strip(),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class AnthropicProvider(Provider):
    """The Anthropic messages API."""

    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ProviderError("Set ANTHROPIC_API_KEY before using --provider anthropic.")

    def complete(self, prompt: str) -> Answer:
        payload = {
            "model": self.model,
            "max_tokens": 32,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post("https://api.anthropic.com/v1/messages", payload,
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Answer(
            text=text.strip(),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


def _post(url: str, payload: dict, headers: dict, timeout: int, retries: int = 3) -> dict:
    """POST JSON, retrying on rate limits and temporary server errors."""
    body = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}

    last_error = ""
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=all_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise ProviderError(last_error) from exc
        except urllib.error.URLError as exc:
            last_error = f"Cannot reach {url}: {exc.reason}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise ProviderError(last_error) from exc
    raise ProviderError(last_error)


def build(provider: str, model: str, base_url: str | None = None) -> Provider:
    """Create a provider from the --provider flag."""
    provider = provider.lower()
    if provider == "mock":
        return MockProvider(model or "mock-classifier")
    if provider in OpenAICompatibleProvider.ENDPOINTS or provider == "local":
        flavour = "openai" if provider == "local" else provider
        return OpenAICompatibleProvider(model, base_url=base_url, flavour=flavour)
    if provider == "anthropic":
        return AnthropicProvider(model)
    raise ProviderError(f"Unknown provider: {provider}")