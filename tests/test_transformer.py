"""
Tests for the transformer module.

These use a TINY model ("hf-internal-testing/tiny-random-DistilBertForSequenceClassification")
that Hugging Face publishes specifically for testing. It's only ~1MB
and produces random garbage — but it proves our code runs without
errors.  Real accuracy testing happens when you run train_transformer.py.

Kotlin analogy: like using a mock Retrofit client in unit tests —
you don't hit the real API, you just verify the wiring is correct.
"""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from convo_assistant.transformers.transformer import (
    IntentDataset,
    predict,
    tokenize_texts,
)

# A tiny random model for fast tests (no real DistilBERT download)
TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"

SAMPLE_TEXTS = [
    "I want to activate my card",
    "my card was stolen",
    "what is the exchange rate",
]


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Load the tiny test model once for all tests in this file."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_MODEL, num_labels=3, ignore_mismatched_sizes=True
    )
    return model, tokenizer


def test_tokenize_returns_tensors(tiny_model_and_tokenizer):
    """Tokenizer produces input_ids and attention_mask tensors."""
    _, tokenizer = tiny_model_and_tokenizer
    encodings = tokenize_texts(SAMPLE_TEXTS, tokenizer, max_length=32)

    assert "input_ids" in encodings
    assert "attention_mask" in encodings
    assert encodings["input_ids"].shape[0] == 3  # 3 messages
    assert encodings["input_ids"].dtype == torch.long


def test_intent_dataset_length(tiny_model_and_tokenizer):
    """IntentDataset reports correct length."""
    _, tokenizer = tiny_model_and_tokenizer
    encodings = tokenize_texts(SAMPLE_TEXTS, tokenizer, max_length=32)
    dataset = IntentDataset(encodings, [0, 1, 2])

    assert len(dataset) == 3


def test_intent_dataset_getitem(tiny_model_and_tokenizer):
    """IntentDataset returns a dict with input_ids, attention_mask, labels."""
    _, tokenizer = tiny_model_and_tokenizer
    encodings = tokenize_texts(SAMPLE_TEXTS, tokenizer, max_length=32)
    dataset = IntentDataset(encodings, [0, 1, 2])

    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["labels"].item() == 0


def test_predict_returns_list(tiny_model_and_tokenizer):
    """predict() returns a list of ints with one prediction per message."""
    model, tokenizer = tiny_model_and_tokenizer
    preds = predict(model, tokenizer, SAMPLE_TEXTS, device=torch.device("cpu"))

    assert isinstance(preds, list)
    assert len(preds) == 3
    assert all(isinstance(p, int) for p in preds)
