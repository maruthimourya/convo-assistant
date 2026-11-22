"""
baseline.py — TF-IDF + Logistic Regression intent classifier.

THE BIG PICTURE (for a Kotlin developer):
  Imagine you're building a chat-routing system: a customer types
  "I lost my card", and you need to route it to the right support
  queue — one of 77 categories.

  An Android dev might think: "I'd write 77 if-else rules checking
  for keywords."  That's fragile.  Instead, we let a model *learn*
  the rules from thousands of labelled examples.

STEP 1 — TF-IDF (Term Frequency–Inverse Document Frequency):
  Turns each text message into a vector (a fixed-length list of
  numbers).  Think of it as an embedding, but simpler:

    "I lost my card"  →  [0.0, 0.0, 0.42, ..., 0.31]  (say, 10,000 slots)

  - TF (Term Frequency): how often a word appears in THIS message.
    Like counting word occurrences in a String.
  - IDF (Inverse Document Frequency): penalises common words like
    "the", "I", "my" that appear in EVERY message.  Rare words like
    "lost", "stolen" get higher weight because they're more
    informative.
  - The combo TF × IDF gives each word a score per message.

  Kotlin analogy: imagine each message is a Map<String, Double>
  where the key is a word and the value is its TF-IDF weight.
  Scikit-learn's TfidfVectorizer does this for all messages at once
  and returns a sparse matrix (like a SparseArray in Android — most
  entries are zero because each message uses only a few words).

STEP 2 — Logistic Regression:
  A linear classifier (a simple equation: y = Wx + b) that takes
  the TF-IDF vector and outputs a probability for each of the 77
  intents.  The highest-probability intent wins.

  Kotlin analogy: it's like a when() expression, but instead of
  hand-written conditions, the model learned the weights W from the
  training data.

  "Logistic" means it uses a sigmoid/softmax function to squeeze
  raw scores into 0-1 probabilities — same idea as the sigmoid in
  a neural network, just one layer.

WHY THIS BASELINE?
  - Fast: trains in seconds, no GPU needed.
  - Surprisingly strong: often 85-92% on Banking77.
  - Standard: every ML interview expects you to compare a fancy
    model against a simple baseline.  If your transformer only
    barely beats TF-IDF + LR, the simpler model wins.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_lr_pipeline(
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline that chains TF-IDF → Logistic Regression.

    Parameters
    ----------
    max_features : int
        Keep only the top N words by TF-IDF score across the whole
        training set.  10,000 is a good default for Banking77 —
        enough to capture domain words without blowing up memory.

    ngram_range : (min_n, max_n)
        (1, 2) means use both single words ("card") AND two-word
        phrases ("lost card").  Bigrams (two-word combos) often
        help because "wrong exchange" is more specific than "wrong"
        or "exchange" alone.

    C : float
        Regularisation strength (inverse). Higher C = less
        regularisation = the model fits training data more tightly.
        Think of it like a learning-rate knob: too high and the
        model memorises noise, too low and it underfits.
        1.0 is scikit-learn's default and usually fine.

    max_iter : int
        How many optimisation steps the solver can take.  1000 is
        plenty for convergence on Banking77.

    Returns
    -------
    Pipeline
        An unfitted (untrained) pipeline.  Call pipeline.fit(X, y)
        to train it, then pipeline.predict(X_new) to classify.

        A Pipeline chains steps so you don't forget to transform
        new data the same way.  Like a Kotlin extension function
        chain:  text.tfidf().classify()
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    # sublinear_tf=True applies log(1 + TF) instead of
                    # raw TF — dampens the effect of a word appearing
                    # 10× vs 1×.  Almost always helps.
                    sublinear_tf=True,
                    # strip_accents and lowercase are on by default,
                    # so "Card" and "card" are treated the same.
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    # 'lbfgs' is the default solver; works well for
                    # multi-class (77 classes).  It's a quasi-Newton
                    # optimiser — think gradient descent but smarter.
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train(pipeline: Pipeline, texts: list[str], labels: list[int]) -> Pipeline:
    """
    Fit (train) the pipeline on labelled examples.

    Parameters
    ----------
    texts  : the customer messages (raw strings).
    labels : the intent ids (ints 0-76).

    Returns the fitted pipeline (same object, mutated — like a
    Builder pattern that returns `this`).
    """
    pipeline.fit(texts, labels)
    return pipeline
