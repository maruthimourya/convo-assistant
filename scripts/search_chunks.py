"""search_chunks.py - try the retrieval step on a few questions.

Run from the project folder:
  uv run python scripts/search_chunks.py
"""

from convo_assistant.rag.search import search

QUERIES = [
    "how much money can I take out in a day",  # no word "withdraw" -> tests meaning
    "my card got stolen last night",
    "why was my payment rejected",
    "do I pay extra for spending abroad",
]


def main() -> None:
    for q in QUERIES:
        print(f"\nQ: {q!r}")
        for chunk, score in search(q, k=2):
            print(f"  {score:.3f}  [{chunk['source']}] {chunk['section']}")


if __name__ == "__main__":
    main()
