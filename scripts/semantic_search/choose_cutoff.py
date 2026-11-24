"""
choose_cutoff.py - pick the semantic-search confidence cutoff WITHOUT touching the test set.

Steps:
  1. Split the TRAINING data: 80% stays as the search store, 20% becomes a validation set.
  2. For each cutoff, measure on validation only:
       - answer rate (how many messages get a real answer)
       - accuracy on answered messages
       - confident card_lost mistakes
  3. Pick the LOWEST cutoff that meets the rule below (lowest = answers the most messages).
  4. Report the test set ONCE, at the chosen cutoff.

The rule is fixed before we look at results:
  accuracy on answered >= 95%  AND  zero confident card_lost mistakes on validation.

Run from the project folder:
  uv run python scripts/semantic_search/choose_cutoff.py > outputs/semantic_search/choose_cutoff.txt 2>&1
"""

from pathlib import Path

import numpy as np

from convo_assistant.shared.dataset import label_names, load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("outputs/semantic_search")
CUTOFFS = np.round(np.arange(0.70, 0.96, 0.05), 2)
MIN_ACCURACY = 0.95
SEED = 42


def evaluate(scores, preds, true, cutoff, lost_id):
    answered = scores >= cutoff
    correct = preds == true
    answer_rate = answered.mean()
    acc = correct[answered].mean() if answered.any() else float("nan")
    lost_wrong = int(np.sum(answered & ~correct & (true == lost_id)))
    return answer_rate, acc, lost_wrong


def main() -> None:
    ds = load_banking77()
    names = label_names(ds)
    lost_id = names.index("lost_or_stolen_card")

    all_vecs = np.load(OUT_DIR / "banking77_train_embeddings.npy")
    all_labels = np.load(OUT_DIR / "banking77_train_labels.npy")

    # 1. Split the training data, not the test data.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(all_labels))
    cut = int(0.8 * len(order))
    store_idx, val_idx = order[:cut], order[cut:]
    store_vecs, store_labels = all_vecs[store_idx], all_labels[store_idx]
    val_vecs, val_labels = all_vecs[val_idx], all_labels[val_idx]
    print(
        f"Search store: {len(store_idx):,} messages   Validation: {len(val_idx):,} messages"
    )

    def search(query_vecs):
        sims = query_vecs @ store_vecs.T
        nearest = np.argmax(sims, axis=1)
        return sims[np.arange(len(query_vecs)), nearest], store_labels[nearest]

    val_scores, val_preds = search(val_vecs)

    print("\n=== Validation (used to choose the cutoff) ===")
    print(
        f"{'cutoff':>6} {'answered':>9} {'acc on answered':>16} {'confident card_lost wrong':>26}"
    )
    chosen = None
    for c in CUTOFFS:
        rate, acc, lw = evaluate(val_scores, val_preds, val_labels, c, lost_id)
        ok = (not np.isnan(acc)) and acc >= MIN_ACCURACY and lw == 0
        mark = "  <- meets rule" if ok else ""
        print(f"{c:>6.2f} {rate * 100:8.1f}% {acc * 100:15.1f}% {lw:26d}{mark}")
        if ok and chosen is None:
            chosen = c

    if chosen is None:
        print(
            "\nNo cutoff meets the rule on validation. Do not ship semantic search yet."
        )
        return

    print(f"\nChosen cutoff (lowest that meets the rule): {chosen:.2f}")

    # 4. Test set, reported once, at the chosen cutoff.
    test_vecs = (
        np.load(OUT_DIR / "banking77_test_embeddings.npy")
        if (OUT_DIR / "banking77_test_embeddings.npy").exists()
        else None
    )
    if test_vecs is None:
        from sentence_transformers import SentenceTransformer as ST

        test_texts = ds["test"]["text"]
        test_vecs = ST(MODEL_NAME).encode(
            test_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    test_labels = np.array(ds["test"]["label"])
    test_scores, test_preds = search(test_vecs)
    rate, acc, lw = evaluate(test_scores, test_preds, test_labels, chosen, lost_id)
    print("\n=== Test set (reported once, at the chosen cutoff) ===")
    print(
        f"Answered: {rate * 100:.1f}%   Accuracy on answered: {acc * 100:.1f}%   "
        f"Confident card_lost wrong: {lw}"
    )
    print(f"Clarifying question for the other {(1 - rate) * 100:.1f}% of messages.")


if __name__ == "__main__":
    main()
