"""
Tests for the intent classification module.

These tests run WITHOUT downloading the real Banking77 dataset
or training a real model — they use small fake data, just like
your existing tests use a FakeLLM.

Kotlin analogy: think of these as unit tests with mock data,
like using a FakeRepository in your Android tests.
"""

import pytest
from convo_assistant.intent.baseline import build_tfidf_lr_pipeline, train
from convo_assistant.intent.evaluate import compute_metrics


# ── Fixtures: tiny fake dataset ──────────────────────────────
# In pytest, a @pytest.fixture is like a @Before setup method
# in JUnit/Android tests — it prepares shared test data.

FAKE_TEXTS = [
    "I want to activate my card",
    "activate card please",
    "please activate my new card",
    "how do I activate this card",
    "my card is lost",
    "I lost my debit card",
    "card was stolen",
    "someone stole my card",
]
FAKE_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]  # 0 = activate, 1 = lost/stolen
FAKE_LABEL_NAMES = ["activate_my_card", "lost_or_stolen_card"]


def test_pipeline_builds():
    """Pipeline object is created without errors."""
    pipe = build_tfidf_lr_pipeline()
    # A Pipeline has named steps; check both are there.
    assert "tfidf" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_trains_and_predicts():
    """Pipeline can fit on tiny data and return predictions."""
    pipe = build_tfidf_lr_pipeline(max_features=50, ngram_range=(1, 1))
    train(pipe, FAKE_TEXTS, FAKE_LABELS)

    preds = pipe.predict(["activate my card", "I lost it"])
    # Should return a list/array of ints
    assert len(preds) == 2
    assert all(isinstance(int(p), int) for p in preds)


def test_pipeline_learns_simple_pattern():
    """With clear-cut training data, the model should get these right."""
    pipe = build_tfidf_lr_pipeline(max_features=50, ngram_range=(1, 1))
    train(pipe, FAKE_TEXTS, FAKE_LABELS)

    preds = pipe.predict(["activate my card now", "my card was stolen yesterday"])
    assert preds[0] == 0  # activate
    assert preds[1] == 1  # lost/stolen


def test_compute_metrics():
    """Metrics function returns accuracy, report string, and dict."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]  # one mistake

    result = compute_metrics(y_true, y_pred, label_names=FAKE_LABEL_NAMES)
    assert "accuracy" in result
    assert result["accuracy"] == pytest.approx(0.75)  # 3 out of 4
    assert "report_str" in result
    assert "report_dict" in result
    assert "activate_my_card" in result["report_str"]
