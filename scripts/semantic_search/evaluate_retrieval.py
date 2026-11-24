"""
evaluate_retrieval.py - measure how well the SEARCH step ranks training messages.

For each TEST message, a "good answer" is any TRAINING message with the same
intent label. We rank all training messages and check where the good answers land.

Metrics (computed per test message, then averaged):
  hit@k        1 if at least one good answer is in the top k, else 0
  precision@k  good answers in the top k / k
  MRR          1 / position of the first good answer (0 if none in the top 50)

Kotlin view:
  val ranked = trainVecs.indices.sortedByDescending { cosine(q, trainVecs[it]) }
  val top = ranked.take(k)
  val hit = top.any { trainLabels[it] == trueLabel }

Run from the project folder:
  uv run python scripts/semantic_search/evaluate_retrieval.py > outputs/semantic_search/eval_retrieval.txt 2>&1

Note: hit@1 here equals the top-1 accuracy printed by evaluate_semantic.py.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from convo_assistant.shared.dataset import load_banking77

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = "outputs/semantic_search"
KS = [1, 3, 5, 10]
MAX_RANK = 50


def main() -> None:
    ds = load_banking77()
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])

    train_vecs = np.load(f"{OUT_DIR}/banking77_train_embeddings.npy")
    train_labels = np.load(f"{OUT_DIR}/banking77_train_labels.npy")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(test_texts):,} test messages...")
    test_vecs = model.encode(
        test_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )

    # Shape: (test messages) x (training messages). Each row is one ranking.
    sims = test_vecs @ train_vecs.T
    # Best MAX_RANK training indices per test message, best first.
    top = np.argsort(-sims, axis=1)[:, :MAX_RANK]
    top_labels = train_labels[top]

    # True where the training message at that rank has the same label as the test message.
    is_good = top_labels == test_labels[:, None]

    print("\n  k   hit@k    precision@k")
    for k in KS:
        hit = is_good[:, :k].any(axis=1).mean() * 100
        prec = is_good[:, :k].mean() * 100
        print(f"  {k:<3} {hit:6.1f}%   {prec:6.1f}%")

    has_good = is_good.any(axis=1)
    first_good = np.argmax(is_good, axis=1)  # index of first True in each row
    reciprocal = np.where(has_good, 1.0 / (first_good + 1), 0.0)
    print(f"\nMRR (top {MAX_RANK}): {reciprocal.mean():.3f}")


if __name__ == "__main__":
    main()
