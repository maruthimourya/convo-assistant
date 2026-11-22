"""
semantic.py - classify a customer message by finding the closest training message.

How it works (Kotlin view):
  val best = trainingMessages.maxBy { cosine(embed(it), embed(query)) }
  if (best.score < CUTOFF) return null   // not sure: ask the user a clarifying question
  return best.label

Usage:
  from convo_assistant.intent.semantic import classify_intent
  classify_intent("I lost my card")   # -> "lost_or_stolen_card"
  classify_intent("blah blah")        # -> None (ask a clarifying question)

Needs outputs/banking77_train_embeddings.npy and outputs/banking77_train_labels.npy,
created by scripts/encode_banking77.py. Run from the project folder.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.intent.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = Path("outputs/banking77_train_embeddings.npy")
LABELS_PATH = Path("outputs/banking77_train_labels.npy")

# Chosen on validation data (see scripts/choose_cutoff.py).
# Below this score we ask a clarifying question instead of guessing.
CUTOFF = 0.80


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Loaded once, then reused for every message.
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _store() -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectors = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)
    names = label_names(load_banking77())
    return vectors, labels, names


def classify_intent(text: str) -> str | None:
    """Return the intent name, or None if the closest match is below CUTOFF."""
    vectors, labels, names = _store()
    query_vec = _model().encode(text, normalize_embeddings=True)
    scores = vectors @ query_vec  # cosine similarity, since vectors have length 1
    best = int(np.argmax(scores))
    if scores[best] < CUTOFF:
        return None
    return names[labels[best]]
