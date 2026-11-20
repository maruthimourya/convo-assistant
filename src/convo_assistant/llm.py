import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM(Protocol):
    """The interface: anything with a reply() method counts as an LLM."""

    def reply(self, history: list[dict]) -> str: ...


class GroqLLM:
    """Cloud brain: a model hosted on Groq's servers. Needs GROQ_API_KEY."""

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.model = model

    def reply(self, history: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=history, max_tokens=400
        )
        return resp.choices[0].message.content or ""


class OllamaLLM:
    """Local brain: a model running on your Mac through Ollama. Free, offline."""

    def __init__(self, model: str = "qwen3:8b"):
        # Ollama speaks the same API format as OpenAI, so the same client works.
        # It ignores the API key, but the client requires one, so we pass a dummy.
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model

    def reply(self, history: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=history, max_tokens=400
        )
        return resp.choices[0].message.content or ""


def make_llm() -> LLM:
    """Factory: picks the brain from the LLM_PROVIDER setting (default: groq)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return GroqLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'groq' or 'ollama')")
