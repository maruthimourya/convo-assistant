#!/usr/bin/env python3
"""
train_transformer.py — Fine-tune DistilBERT on Banking77 and
compare against the TF-IDF baseline.

Run with:
    uv run python scripts/train_transformer.py

Takes ~5-15 minutes on an M5 Mac (uses MPS GPU acceleration).
"""

import time
from pathlib import Path

from convo_assistant.intent.dataset import describe, label_names, load_banking77
from convo_assistant.intent.evaluate import compute_metrics, plot_confusion_matrix
from convo_assistant.intent.transformer import (
    IntentDataset,
    fine_tune,
    load_model_and_tokenizer,
    pick_device,
    predict,
    save_model,
    tokenize_texts,
)


def main() -> None:
    print("=" * 60)
    print("Week 3 — Transformer Fine-Tuning (DistilBERT)")
    print("=" * 60)

    device = pick_device()
    print(f"\n🖥️  Device: {device}")
    if device.type == "mps":
        print("   (Apple Silicon GPU — nice, this will be fast!)")
    elif device.type == "cuda":
        print("   (Nvidia GPU)")
    else:
        print("   (CPU only — training will be slower, ~20-40 min)")

    # ── Step 1: Load data ────────────────────────────────────
    print("\n📦 Loading Banking77 dataset...")
    ds = load_banking77()
    describe(ds)

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    names = label_names(ds)

    # ── Step 2: Load pre-trained DistilBERT + tokenizer ──────
    print("\n🤖 Loading DistilBERT (downloading on first run, ~250MB)...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="distilbert-base-uncased",
        num_labels=len(names),  # 77
    )

    # ── Step 3: Tokenize (convert text → token IDs) ─────────
    print("\n✂️  Tokenizing training data...")
    train_encodings = tokenize_texts(train_texts, tokenizer, max_length=128)
    train_dataset = IntentDataset(train_encodings, train_labels)

    print(f"   Tokenized {len(train_dataset):,} examples")
    print(f"   Example: {train_texts[0]!r}")
    # Show what tokenization looks like
    sample_tokens = tokenizer.convert_ids_to_tokens(
        train_encodings["input_ids"][0][:20]
    )
    print(f"   Tokens:  {sample_tokens}")

    # ── Step 4: Fine-tune ────────────────────────────────────
    print("\n🏋️  Fine-tuning DistilBERT (3 epochs)...")
    print("   This takes ~5-15 minutes on an M5 Mac.\n")
    t0 = time.time()
    fine_tune(
        model,
        train_dataset,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"\n✅ Fine-tuning done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Step 5: Predict on test set ──────────────────────────
    print("\n🔮 Predicting on the test set...")
    y_pred = predict(model, tokenizer, test_texts, device=device)

    # ── Step 6: Evaluate ─────────────────────────────────────
    metrics = compute_metrics(test_labels, y_pred, label_names=names)

    print(f"\n📊 Transformer accuracy: {metrics['accuracy']:.4f}")
    print(f"   ({metrics['accuracy'] * 100:.1f}% of 3,080 test messages)")
    print("\nPer-intent Precision / Recall / F1:")
    print(metrics["report_str"])

    # ── Step 7: Confusion matrix ─────────────────────────────
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cm_path = plot_confusion_matrix(
        test_labels, y_pred, label_names=names,
        save_path=out_dir / "confusion_matrix_transformer.png",
    )

    # ── Step 8: Save model for later use ─────────────────────
    save_model(model, tokenizer, path="models/intent-distilbert")

    # ── Step 9: Side-by-side comparison ──────────────────────
    print("\n" + "=" * 60)
    print("📋 BASELINE vs TRANSFORMER COMPARISON")
    print("=" * 60)

    # Load baseline results if available (from train_baseline.py)
    baseline_acc = 0.8578  # your baseline result
    transformer_acc = metrics["accuracy"]
    improvement = (transformer_acc - baseline_acc) * 100

    print(f"\n  TF-IDF + LogReg (baseline):  {baseline_acc:.1%}")
    print(f"  DistilBERT (transformer):    {transformer_acc:.1%}")
    print(f"  Improvement:                 {improvement:+.1f} percentage points")

    if transformer_acc > baseline_acc:
        print("\n  🎉 Transformer wins!")
    else:
        print("\n  🤔 Baseline held its ground — try more epochs or a larger model.")

    print(f"\n  Confusion matrix: {cm_path}")
    print("  Saved model:      models/intent-distilbert/")
    print("\n  The saved model can be loaded later with:")
    print("    from convo_assistant.intent.transformer import load_saved_model")
    print('    model, tokenizer = load_saved_model("models/intent-distilbert")')


if __name__ == "__main__":
    main()
