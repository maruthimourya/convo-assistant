"""
encode_chunks.py - precompute chunk embeddings ONCE and save them to disk.

Mirrors scripts/encode_banking77.py from Week 4: embed a fixed corpus once so
searches don't recompute it every time.

Run from the project folder (needs the embedding model):
  uv run python scripts/encode_chunks.py

Re-run this whenever outputs/chunks.jsonl changes, or the saved vectors go stale.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.rag.search import load_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("outputs/chunks_embeddings.npy")


def main() -> None:
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(texts, normalize_embeddings=True)

    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, vecs)
    print(f"Saved {vecs.shape[0]} chunk vectors ({vecs.shape[1]} dims each) to {OUT}")


if __name__ == "__main__":
    main()
