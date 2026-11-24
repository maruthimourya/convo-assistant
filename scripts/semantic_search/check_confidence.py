"""
check_confidence.py - are semantic search's mistakes made with high confidence?

For each test message we keep the top match's score (its "confidence").
Then we ask two questions:
  1. Which mistakes involve the watched intents, and how confident was each one?
  2. If we only trust matches above a cutoff score, how many messages do we
     answer, how accurate are those answers, and how many risky card_lost
     mistakes slip through?

Kotlin view:
  val answered = results.filter { it.score >= cutoff }
  val accuracy = answered.count { it.correct } / answered.size

Run from the project folder:
  uv run python scripts/semantic_search/check_confidence.py > outputs/check_confidence.txt 2>&1
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.shared.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs")
WATCHED = ["lost_or_stolen_card", "card_not_working", "declined_card_payment"]
CUTOFFS = [0.80, 0.85, 0.90]


def main() -> None:
    ds = load_banking77()
    names = label_names(ds)
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])
    train_vecs = np.load(OUT_DIR / "banking77_train_embeddings.npy")
    train_labels = np.load(OUT_DIR / "banking77_train_labels.npy")

    model = SentenceTransformer(MODEL_NAME)
    test_vecs = model.encode(
        test_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    sims = test_vecs @ train_vecs.T
    nearest = np.argmax(sims, axis=1)
    scores = sims[np.arange(len(test_texts)), nearest]
    predictions = train_labels[nearest]
    correct = predictions == test_labels

    watched_ids = {names.index(n) for n in WATCHED}
    lost = names.index("lost_or_stolen_card")

    print("=== Wrong predictions involving the watched intents ===")
    for i in np.where(~correct)[0]:
        if test_labels[i] in watched_ids or predictions[i] in watched_ids:
            print(
                f"  score={scores[i]:.3f}  true={names[test_labels[i]]}  "
                f"predicted={names[predictions[i]]}"
            )
            print(f"      text: {test_texts[i]!r}")

    print("\n=== Trusting only matches at or above a cutoff ===")
    print(
        f"{'cutoff':>6} {'answered':>9} {'accuracy on answered':>21} {'lost wrong (confident)':>23}"
    )
    for cutoff in CUTOFFS:
        answered = scores >= cutoff
        acc = correct[answered].mean() * 100 if answered.any() else float("nan")
        lost_wrong = int(np.sum(answered & ~correct & (test_labels == lost)))
        print(
            f"{cutoff:>6.2f} {answered.mean() * 100:8.1f}% {acc:20.1f}% {lost_wrong:23d}"
        )
    print(
        f"\nScore range, correct matches: {scores[correct].min():.3f} to {scores[correct].max():.3f}"
    )
    print(
        f"Score range, wrong matches:   {scores[~correct].min():.3f} to {scores[~correct].max():.3f}"
    )


if __name__ == "__main__":
    main()
