import pytest

from convo_assistant.llm import GroqLLM, OllamaLLM, make_llm


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    # CI has no real key; the client only checks that one is set, it isn't used.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert isinstance(make_llm(), GroqLLM)


def test_ollama_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert isinstance(make_llm(), OllamaLLM)


def test_unknown_provider_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        make_llm()
