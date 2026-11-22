"""
evaluate_semantic.py - score semantic search on the held-out TEST split.

For each test message:
  1. encode it (same model as Step 3)
  2. find the single closest TRAINING message (top-1 nearest neighbour)
  3. predict that training message's label
Then compare predictions with the true test labels.

Kotlin view:
  val predictions = testVecs.map { q -> trainLabels[argmax(cosine(q, trainVecs))] }
  val accuracy = predictions.zip(trueLabels).count { (p, t) -> p == t } / size.toDouble()

Run from the project folder:
  uv run python scripts/evaluate_semantic.py > outputs/eval_semantic.txt 2>&1

Compare with Week 3:
  TF-IDF + logistic regression: 85.8%
  DistilBERT fine-tune:         85.1%
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.intent.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs")
WATCHED = ["lost_or_stolen_card", "card_not_working", "declined_card_payment"]


def main() -> None:
    ds = load_banking77()
    names = label_names(ds)
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])

    # Training vectors and labels saved in Step 3.
    train_vecs = np.load(OUT_DIR / "banking77_train_embeddings.npy")
    train_labels = np.load(OUT_DIR / "banking77_train_labels.npy")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(test_texts):,} test messages...")
    test_vecs = model.encode(
        test_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )

    # Every row is length 1, so this matrix product gives all cosine scores at once.
    # Shape: (3080 test messages) x (10003 training messages).
    sims = test_vecs @ train_vecs.T
    nearest = np.argmax(sims, axis=1)
    predictions = train_labels[nearest]

    correct = predictions == test_labels
    print(f"\nOverall top-1 accuracy: {correct.mean() * 100:.1f}%")
    print("(Week 3 comparison: TF-IDF 85.8%, DistilBERT 85.1%)\n")

    print("Accuracy on the watched intents:")
    for name in WATCHED:
        idx = names.index(name)
        mask = test_labels == idx
        acc = correct[mask].mean() * 100
        print(f"  {name:<24} {acc:5.1f}%  ({mask.sum()} test messages)")

    # Show the most common confusions, so we can see if card_lost and card_not_working blur.
    print("\nWrong predictions, most common (true -> predicted):")
    wrong = ~correct
    pairs = list(zip(test_labels[wrong], predictions[wrong]))
    counts: dict[tuple[int, int], int] = {}
    for t, p in pairs:
        counts[(t, p)] = counts.get((t, p), 0) + 1
    for (t, p), n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:3d}x  {names[t]} -> {names[p]}")


if __name__ == "__main__":
    main()
