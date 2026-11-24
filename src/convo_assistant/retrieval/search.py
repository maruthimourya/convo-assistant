"""
search.py - the retrieval step: find the chunks most relevant to a question.

Week 4's semantic search, pointed at DOCUMENT CHUNKS. The chunk vectors are
PRECOMPUTED (see scripts/retrieval/encode_chunks.py) and loaded from disk; only the
query is embedded live, because it's new every time.

Pipeline:
  1. Load chunks (text + metadata) from outputs/chunks.jsonl.
  2. Load their precomputed vectors from outputs/chunks_embeddings.npy.
  3. Embed the incoming question.
  4. Score every chunk by cosine similarity and return the top k.

Kotlin view:
  val ranked = chunks.indices.sortedByDescending { cosine(chunkVecs[it], embed(query)) }
  return ranked.take(k)
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = Path("outputs/chunks.jsonl")
EMBEDDINGS_PATH = Path("outputs/chunks_embeddings.npy")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Read chunks.jsonl back into a list of dicts (one per line)."""
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@lru_cache(maxsize=1)
def _store() -> tuple[list[dict], np.ndarray]:
    """Load chunks and their precomputed vectors once, then reuse."""
    chunks = load_chunks()
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. Build it first:\n"
            f"  uv run python scripts/retrieval/encode_chunks.py"
        )
    vectors = np.load(EMBEDDINGS_PATH)
    if len(vectors) != len(chunks):
        raise ValueError(
            f"{len(vectors)} vectors but {len(chunks)} chunks. "
            f"Re-run scripts/retrieval/encode_chunks.py after changing chunks.jsonl."
        )
    return chunks, vectors


def search(query: str, k: int = 3) -> list[tuple[dict, float]]:
    """Return the top k chunks for a question, each with its similarity score."""
    chunks, chunk_vecs = _store()
    query_vec = _model().encode(query, normalize_embeddings=True)

    # Dot product = cosine similarity, because every vector has length 1.
    scores = chunk_vecs @ query_vec
    order = np.argsort(-scores)[:k]  # indices of the k highest scores, best first
    return [(chunks[i], float(scores[i])) for i in order]
