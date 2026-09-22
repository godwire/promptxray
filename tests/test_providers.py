"""Network providers, without any network: urlopen is stubbed out."""

from __future__ import annotations

import json
import urllib.error

import pytest

from promptxray import providers


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc) -> None:
        return None


def _stub_urlopen(monkeypatch, responder):
    """Replace urllib.request.urlopen with `responder(request) -> dict | Exception`."""

    def fake_urlopen(request, timeout=None):
        result = responder(request)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)


def test_openai_compatible_parses_content_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = {}

    def responder(request):
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        return {
            "choices": [{"message": {"content": " spam "}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 1},
        }

    _stub_urlopen(monkeypatch, responder)
    provider = providers.build("openai", "gpt-4o-mini")
    answer = provider.complete("classify this")

    assert answer.text == "spam"
    assert answer.input_tokens == 42
    assert answer.output_tokens == 1
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"


def test_openai_compatible_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="OPENROUTER_API_KEY"):
        providers.build("openrouter", "some-model")


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = providers.build("ollama", "llama3.2")
    assert provider.api_key == "not-needed"


def test_anthropic_parses_text_blocks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def responder(_request):
        return {
            "content": [{"type": "text", "text": "urgent"}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    _stub_urlopen(monkeypatch, responder)
    provider = providers.AnthropicProvider("claude-haiku-4-5")
    answer = provider.complete("classify this")

    assert answer.text == "urgent"
    assert answer.input_tokens == 10
    assert answer.output_tokens == 2


def test_anthropic_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="ANTHROPIC_API_KEY"):
        providers.AnthropicProvider("claude-haiku-4-5")


def test_gemini_parses_candidates_and_sends_key_as_header(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    seen = {}

    def responder(request):
        seen["url"] = request.full_url
        seen["key_header"] = request.headers.get("X-goog-api-key")
        return {
            "candidates": [{"content": {"parts": [{"text": "normal"}]}}],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 1},
        }

    _stub_urlopen(monkeypatch, responder)
    provider = providers.build("gemini", "gemini-1.5-flash")
    answer = provider.complete("classify this")

    assert answer.text == "normal"
    assert answer.input_tokens == 30
    assert answer.output_tokens == 1
    # The key must travel as a header, never in the URL - a retry's error
    # message includes the URL, and that must never leak the key.
    assert "gm-test" not in seen["url"]
    assert seen["key_header"] == "gm-test"


def test_gemini_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="GEMINI_API_KEY"):
        providers.build("gemini", "gemini-1.5-flash")


def test_post_retries_on_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)
    calls = {"n": 0}

    def responder(_request):
        calls["n"] += 1
        if calls["n"] == 1:
            return urllib.error.HTTPError(
                "http://x", 429, "rate limited", {}, None
            )
        return {"ok": True}

    def fake_urlopen(request, timeout=None):
        result = responder(request)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    data = providers._post("http://x", {}, {}, timeout=5)
    assert data == {"ok": True}
    assert calls["n"] == 2


def test_post_gives_up_after_persistent_error(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(providers.ProviderError, match="Cannot reach"):
        providers._post("http://x", {}, {}, timeout=5, retries=2)
