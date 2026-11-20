import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLM(Protocol):
    def reply(self, history: list[dict]) -> str: ...


class GroqLLM:
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
