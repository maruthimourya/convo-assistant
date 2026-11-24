"""rag_demo.py - full RAG: retrieve policy chunks, then generate a grounded answer.

Needs GROQ_API_KEY in .env (or LLM_PROVIDER=ollama with Ollama running).
Run from the project folder:
  uv run python scripts/retrieval/rag_demo.py
"""

from convo_assistant.retrieval.answer import answer

QUESTIONS = [
    "how much money can I take out in a day",
    "my card got stolen, what do I do",
    "do I pay extra for spending abroad",
    "what's the weather tomorrow",  # off-topic -> should refuse
    "do I pay extra to check my balance abroad",
]


def main() -> None:
    for q in QUESTIONS:
        reply, hits = answer(q, k=2)
        print(f"\nQ: {q}")
        print(f"A: {reply}")
        cited = ", ".join(f"{c['section']} ({score:.2f})" for c, score in hits)
        print(f"   sources: {cited}")


if __name__ == "__main__":
    main()
