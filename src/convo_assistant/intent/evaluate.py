"""
evaluate.py — Metrics and confusion-matrix plotting.

WHAT ARE WE MEASURING?

  Accuracy: % of messages classified correctly overall.
    Simple, but misleading if classes are imbalanced (e.g. 90% of
    messages are "check_balance" — a model that always guesses
    "check_balance" gets 90% accuracy but is useless).

  Precision (per class): of all messages the model CALLED class X,
    what fraction actually ARE class X?
    High precision = few false positives.
    Kotlin analogy: if your spam filter marks 100 emails as spam
    and 95 really are spam, precision = 0.95.

  Recall (per class): of all messages that ARE class X, what
    fraction did the model correctly find?
    High recall = few false negatives (missed cases).
    Kotlin analogy: if 100 emails are truly spam and the filter
    catches 80, recall = 0.80.

  F1-score: the harmonic mean of precision and recall — a single
    number that balances both.  It's the standard "how good is
    this classifier?" metric.

  Macro average: average the metric across all 77 classes equally.
    Treats rare intents as important as common ones.

  Weighted average: average weighted by how many examples each
    class has.  More realistic if you care about overall
    correctness.

CONFUSION MATRIX:
  A 77 × 77 grid.  Row i = true label, Column j = predicted label.
  Cell (i, j) = how many examples with true label i got predicted
  as j.  Perfect classifier → diagonal is lit up, off-diagonal
  is zero.  The heatmap makes it easy to spot which intents the
  model confuses (e.g. "card_not_working" vs "card_payment_error").

  Kotlin analogy: it's like a 2D IntArray where
  matrix[actual][predicted] counts occurrences.
"""

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
) -> dict:
    """
    Compute accuracy + per-class precision/recall/F1.

    Returns
    -------
    dict with keys:
      "accuracy"   — float, overall accuracy
      "report_str" — human-readable table (print it!)
      "report_dict"— nested dict for programmatic access
    """
    from sklearn.metrics import classification_report as _cr

    acc = accuracy_score(y_true, y_pred)
    report_str = _cr(y_true, y_pred, target_names=label_names, zero_division=0)
    report_dict = _cr(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": acc,
        "report_str": report_str,
        "report_dict": report_dict,
    }


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str] | None = None,
    save_path: str | Path = "confusion_matrix.png",
    figsize: tuple[int, int] = (20, 18),
) -> Path:
    """
    Plot and save a confusion-matrix heatmap.

    With 77 classes the full matrix is big, so we use a colour
    scale (heatmap) rather than printing numbers in every cell.
    Darker colour = more predictions in that cell.

    Parameters
    ----------
    save_path : where to save the PNG image.
    figsize   : (width, height) in inches; 20×18 fits 77 labels.

    Returns
    -------
    Path to the saved image.
    """
    import matplotlib

    # Use a non-interactive backend so it works without a display
    # (like running on a server or in CI — no GUI window pops up).
    # In Android terms: headless rendering, like drawing to a
    # Bitmap instead of a SurfaceView.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = confusion_matrix(y_true, y_pred)
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    # imshow renders the 2D array as a colour image.
    # cmap="Blues" = lighter blue for low counts, darker for high.
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix — Banking77 Intents", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if label_names:
        tick_marks = np.arange(len(label_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(label_names, rotation=90, fontsize=5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(label_names, fontsize=5)

    ax.set_ylabel("True Intent")
    ax.set_xlabel("Predicted Intent")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # free memory
    print(f"Confusion matrix saved to {save_path}")
    return save_path
