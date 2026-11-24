"""
compare_methods.py - the card_lost vs card_not_working check, for two methods.

Methods compared on the SAME 3,080 test messages:
  1. Semantic search (nearest training message, Week 4)
  2. TF-IDF + logistic regression (Week 3 baseline)

For each method we report:
  - overall accuracy
  - accuracy on lost_or_stolen_card, card_not_working, declined_card_payment
  - how many card_lost messages were predicted as card_not_working, and the reverse

Kotlin view: run two classifiers over the same list and tally a confusion table.

Run from the project folder:
  uv run python scripts/classifiers/compare_methods.py > outputs/compare_methods.txt 2>&1
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.classifiers.baseline import build_tfidf_lr_pipeline
from convo_assistant.shared.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs")
WATCHED = ["lost_or_stolen_card", "card_not_working", "declined_card_payment"]


def report(method: str, preds: np.ndarray, true: np.ndarray, names: list[str]) -> None:
    lost = names.index("lost_or_stolen_card")
    not_working = names.index("card_not_working")
    correct = preds == true
    print(f"\n=== {method} ===")
    print(f"Overall accuracy: {correct.mean() * 100:.1f}%")
    for name in WATCHED:
        idx = names.index(name)
        mask = true == idx
        print(
            f"  {name:<24} {correct[mask].mean() * 100:5.1f}%  ({mask.sum()} messages)"
        )
    lost_as_nw = int(np.sum((true == lost) & (preds == not_working)))
    nw_as_lost = int(np.sum((true == not_working) & (preds == lost)))
    print(f"  card_lost predicted as card_not_working: {lost_as_nw}")
    print(f"  card_not_working predicted as card_lost: {nw_as_lost}")


def main() -> None:
    ds = load_banking77()
    names = label_names(ds)
    train_texts = ds["train"]["text"]
    train_labels = np.array(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])

    # Method 1: semantic search, using the vectors saved in Step 3.
    train_vecs = np.load(OUT_DIR / "banking77_train_embeddings.npy")
    model = SentenceTransformer(MODEL_NAME)
    test_vecs = model.encode(
        test_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    semantic_preds = train_labels[np.argmax(test_vecs @ train_vecs.T, axis=1)]
    report("Semantic search (nearest neighbour)", semantic_preds, test_labels, names)

    # Method 2: TF-IDF + logistic regression, fitted on the same training split.
    print("\nFitting TF-IDF + logistic regression (this can take a minute)...")
    pipeline = build_tfidf_lr_pipeline()
    pipeline.fit(train_texts, train_labels)
    tfidf_preds = pipeline.predict(test_texts)
    report("TF-IDF + logistic regression", tfidf_preds, test_labels, names)


if __name__ == "__main__":
    main()
