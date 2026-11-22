"""
transformer.py — Fine-tune DistilBERT for intent classification.

THE BIG PICTURE:
  In baseline.py we used TF-IDF (word counting) + Logistic Regression
  (learned multiplication).  That got us ~86% accuracy.

  Now we replace BOTH of those with a single transformer model that:
    1. Reads the sentence word by word (not bag-of-words)
    2. Uses "attention" to understand which words relate to each other
    3. Outputs a probability for each of the 77 intents

  We're using DistilBERT — a pre-trained transformer that already
  "speaks English" because it was trained on Wikipedia + books.  We
  just need to teach it our specific task (banking intents) by
  training a few extra rounds on Banking77.  This is called
  "fine-tuning."

NEW CONCEPTS IN THIS FILE:

  Tokenizer (text splitter):
    Before a transformer can read text, it splits each word into
    sub-pieces called "tokens."  Example:
      "unrecognised" → ["un", "##recogn", "##ised"]
    Why sub-words?  So the model can handle words it's never seen
    by combining pieces it knows.  Kotlin analogy: like how you
    can guess what "un-recognize-able" means from its parts.

  Input IDs:
    Each token gets mapped to a number (its position in the
    vocabulary).  "card" → 4003, "lost" → 2439, etc.
    The model only sees these numbers, not the text.

  Attention Mask:
    Messages have different lengths.  We pad shorter ones with
    zeros so they're all the same length (like right-padding a
    String with spaces).  The attention mask is a list of 1s and
    0s telling the model: "pay attention to these tokens (1),
    ignore the padding (0)."

  Epochs (training rounds):
    One epoch = the model sees every training example once.
    We typically do 3-5 epochs.  Like re-reading a textbook
    3 times — each pass you understand it a bit better.

  Batch Size:
    We don't feed all 10,003 examples at once (too much memory).
    Instead we feed them in batches of 16 or 32.  Like loading
    a RecyclerView page by page instead of all items at once.

  Learning Rate:
    How big a step the model takes when adjusting weights after
    each batch.  Too big = overshoots, too small = too slow.
    2e-5 (0.00002) is the standard for fine-tuning transformers.
    It's tiny because the model is already pre-trained — we're
    making small adjustments, not learning from scratch.

  MPS (Metal Performance Shaders):
    Apple's GPU acceleration for M-series chips.  Like using
    RenderScript or Vulkan on Android to speed up computation.
    Training on MPS is 3-5× faster than CPU on your M5 Mac.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── Helper: pick the best available device ───────────────────
def pick_device() -> torch.device:
    """
    Use GPU if available, otherwise CPU.

    Priority:
      1. MPS (Apple Silicon GPU — your M5 Mac)
      2. CUDA (Nvidia GPU — not on a Mac, but just in case)
      3. CPU (always works, just slower)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset class: wraps our data for PyTorch ────────────────
class IntentDataset(Dataset):
    """
    A PyTorch Dataset that holds tokenized messages + labels.

    Kotlin analogy: this is like implementing ListAdapter<ViewHolder>
    for a RecyclerView.  PyTorch's DataLoader calls __getitem__(i)
    to fetch one example at a time, just like onBindViewHolder(pos).

    __len__()      → getItemCount()
    __getitem__(i) → onBindViewHolder(holder, i)
    """

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings  # {"input_ids": [...], "attention_mask": [...]}
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Grab the idx-th token sequence + its label
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Tokenize texts ───────────────────────────────────────────
def tokenize_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> dict:
    """
    Convert raw strings into token IDs + attention masks.

    max_length=128 means: keep at most 128 tokens per message.
    Banking77 messages are short (avg ~15 words), so 128 is
    more than enough.  Longer sequences use more memory and
    are slower.

    Returns a dict with:
      "input_ids"      — token numbers, shape [N, 128]
      "attention_mask"  — 1 for real tokens, 0 for padding
    """
    # Convert to plain list[str] — Hugging Face datasets return a
    # special column type that newer tokenizers reject.
    if not isinstance(texts, list):
        texts = list(texts)

    return tokenizer(
        texts,
        padding=True,  # pad shorter messages to same length
        truncation=True,  # cut messages longer than max_length
        max_length=max_length,
        return_tensors="pt",  # return PyTorch tensors (pt), not lists
    )


# ── Load model + tokenizer ───────────────────────────────────
def load_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 77,
):
    """
    Download (or load from cache) a pre-trained transformer and
    its tokenizer.

    "distilbert-base-uncased":
      - distilbert = the smaller distilled version of BERT
      - base       = the standard size (66 million parameters)
      - uncased    = lowercased text ("Card" == "card")

    num_labels=77: we add a classification "head" (a final layer)
    on top of DistilBERT that outputs 77 scores, one per intent.
    Think of it as adding a custom View on top of a pre-built
    Layout — the Layout (DistilBERT) does the heavy lifting,
    your View (the head) makes the final decision.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


# ── Training loop ────────────────────────────────────────────
def fine_tune(
    model,
    train_dataset: IntentDataset,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> list[float]:
    """
    Fine-tune the transformer on our labelled data.

    This is the core training loop.  In Kotlin pseudocode:

        for (epoch in 1..3) {
            for (batch in dataLoader) {
                val predictions = model.forward(batch)
                val loss = computeLoss(predictions, batch.labels)
                loss.backward()       // compute gradients
                optimizer.step()      // update weights
                optimizer.zeroGrad()  // reset for next batch
            }
        }

    Parameters
    ----------
    epochs : how many times to loop through all training data.
        3 is standard for fine-tuning.  More risks overfitting
        (memorising the training data instead of learning patterns).

    batch_size : examples per training step.  16 is good for a
        24GB Mac.  Bigger = faster but uses more memory.

    learning_rate : step size for weight updates.  2e-5 is the
        "magic number" that the BERT paper established.

    Returns
    -------
    List of average loss per epoch (should decrease each epoch —
    lower loss = better predictions on training data).
    """
    if device is None:
        device = pick_device()

    model.to(device)
    model.train()  # put model in training mode (enables dropout etc.)

    # DataLoader batches our dataset — like a paging adapter.
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer: AdamW is the standard for transformers.
    # It's like gradient descent but smarter — it adapts the
    # learning rate per-weight and includes "weight decay"
    # (a regularisation trick that prevents weights from
    # growing too large).
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler: starts at learning_rate, then
    # linearly decreases to 0 by the end of training.
    # Like easing in an Android animation — start fast, slow down.
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps,
    )

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # Move data to GPU/MPS (like moving a Bitmap to GPU memory)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass: feed tokens through the model, get loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass: compute how each weight should change
            loss.backward()

            # Gradient clipping: prevent exploding gradients
            # (like clamping a value to prevent overflow)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    return epoch_losses


# ── Prediction ───────────────────────────────────────────────
def predict(
    model,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = 64,
    device: torch.device | None = None,
) -> list[int]:
    """
    Predict intent labels for a list of messages.

    Returns a list of predicted label IDs (ints 0-76).
    """
    if device is None:
        device = pick_device()

    model.to(device)
    model.eval()  # evaluation mode: disables dropout

    # Ensure plain list[str] — Hugging Face datasets return a
    # special column type that the tokenizer doesn't accept.
    if not isinstance(texts, list):
        texts = list(texts)

    all_preds = []

    # Process in batches to avoid running out of memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenize_texts(batch_texts, tokenizer)

        with torch.no_grad():  # no gradient computation needed for prediction
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.logits = raw scores for each class (shape: [batch, 77])
        # argmax picks the highest-scoring intent per message
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)

    return all_preds


# ── Save / Load ──────────────────────────────────────────────
def save_model(model, tokenizer, path: str | Path = "models/intent-distilbert"):
    """Save the fine-tuned model to disk so you can reload it later."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")


def load_saved_model(path: str | Path = "models/intent-distilbert"):
    """Reload a previously saved fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer
