#!/usr/bin/env python3
"""
train_baseline.py — Run the full TF-IDF + Logistic Regression pipeline.

This is the "main()" of Week 3.  Run it with:
    uv run python scripts/train_baseline.py

It will:
  1. Download Banking77 (or load from cache)
  2. Train TF-IDF + Logistic Regression
  3. Print accuracy, precision, recall, F1
  4. Save a confusion-matrix heatmap PNG

Kotlin analogy: this is like your Application class or a @JvmStatic
main() that wires everything together and runs it.
"""

import time
from pathlib import Path

from convo_assistant.intent.baseline import build_tfidf_lr_pipeline, train
from convo_assistant.intent.dataset import describe, label_names, load_banking77
from convo_assistant.intent.evaluate import compute_metrics, plot_confusion_matrix


def main() -> None:
    print("=" * 60)
    print("Week 3 — Intent Classification Baseline")
    print("=" * 60)

    # ── Step 1: Load the dataset ─────────────────────────────
    print("\n📦 Loading Banking77 dataset...")
    ds = load_banking77()
    describe(ds)

    # Extract the raw text and label arrays from the dataset.
    # In Kotlin terms: ds["train"] is like a List<DataRow>,
    # and we're pulling out two parallel lists: texts and labels.
    train_texts = ds["train"]["text"]    # List[str], 10,003 items
    train_labels = ds["train"]["label"]  # List[int], 10,003 items
    test_texts = ds["test"]["text"]      # List[str],  3,080 items
    test_labels = ds["test"]["label"]    # List[int],  3,080 items

    # ── Step 2: Build and train the pipeline ─────────────────
    print("\n🔧 Building TF-IDF + Logistic Regression pipeline...")
    pipeline = build_tfidf_lr_pipeline(
        max_features=10_000,  # vocabulary size
        ngram_range=(1, 2),   # unigrams + bigrams
    )

    print("🏋️ Training... (this usually takes 10-30 seconds)")
    t0 = time.time()
    train(pipeline, train_texts, train_labels)
    elapsed = time.time() - t0
    print(f"✅ Training done in {elapsed:.1f}s")

    # ── Step 3: Predict on the test set ──────────────────────
    print("\n🔮 Predicting on the test set...")
    y_pred = pipeline.predict(test_texts)

    # ── Step 4: Evaluate ─────────────────────────────────────
    names = label_names(ds)
    metrics = compute_metrics(test_labels, y_pred, label_names=names)

    print(f"\n📊 Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"   ({metrics['accuracy'] * 100:.1f}% of 3,080 test messages classified correctly)\n")
    print("Per-intent Precision / Recall / F1:")
    print(metrics["report_str"])

    # ── Step 5: Confusion matrix ─────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cm_path = plot_confusion_matrix(
        test_labels, y_pred, label_names=names,
        save_path=out_dir / "confusion_matrix.png",
    )
    print(f"\n🎉 Done! Check {cm_path} for the confusion matrix heatmap.")


if __name__ == "__main__":
    main()
