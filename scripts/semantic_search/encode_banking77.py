"""
encode_banking77.py - turn every Banking77 training message into a vector.

What it does (Kotlin view):
  texts = list of 10,003 strings
  embeddings = texts.map { model.encode(it) }   // each becomes 384 floats
  save the result to disk so we only pay this cost once

Run from the project folder:
  uv run python scripts/semantic_search/encode_banking77.py

Output:
  outputs/semantic_search/banking77_train_embeddings.npy  (shape: 10003 x 384)
  outputs/semantic_search/banking77_train_labels.npy      (shape: 10003, the intent id of each row)
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.shared.dataset import load_banking77

# A small pretrained sentence-transformer (about 90 MB).
# Each sentence becomes a list of 384 numbers.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs/semantic_search")


def main() -> None:
    ds = load_banking77()
    texts = ds["train"]["text"]
    labels = np.array(ds["train"]["label"])

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts):,} training messages...")
    # normalize_embeddings=True scales every vector to length 1,
    # so cosine similarity becomes a simple dot product later.
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "banking77_train_embeddings.npy", embeddings)
    np.save(OUT_DIR / "banking77_train_labels.npy", labels)
    print(f"Saved embeddings with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
