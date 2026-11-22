"""
semantic_search.py - find the training messages closest in meaning to a new message.

Kotlin view:
  val scored = stored.map { it to cosine(embed(it.text), embed(query)) }
  return scored.sortedByDescending { it.score }.take(5)

Run from the project folder (default query is "I lost my card"):
  uv run python scripts/semantic_search.py
  uv run python scripts/semantic_search.py "my card was eaten by the atm"
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.intent.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs")
TOP_K = 5


def main() -> None:
    query = " ".join(sys.argv[1:]) or "I lost my card"

    # Load what Step 3 saved, plus the original texts for display.
    embeddings = np.load(OUT_DIR / "banking77_train_embeddings.npy")
    labels = np.load(OUT_DIR / "banking77_train_labels.npy")
    ds = load_banking77()
    texts = ds["train"]["text"]
    names = label_names(ds)

    model = SentenceTransformer(MODEL_NAME)
    # Same model, same normalization as Step 3, so vectors are comparable.
    query_vec = model.encode(query, normalize_embeddings=True)

    # Because every vector has length 1, the dot product IS the cosine similarity.
    scores = embeddings @ query_vec
    top = np.argsort(-scores)[:TOP_K]

    print(f"\nQuery: {query!r}\n")
    for rank, i in enumerate(top, start=1):
        print(f"{rank}. score={scores[i]:.3f}  label={names[labels[i]]}")
        print(f"     text: {texts[i]!r}")


if __name__ == "__main__":
    main()
