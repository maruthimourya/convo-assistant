"""
dataset.py — Load the Banking77 intent-classification dataset.

WHAT IS BANKING77?
  A public dataset of 13,083 customer-service messages, each labelled
  with one of 77 banking intents (like "activate_my_card",
  "card_payment_wrong_exchange_rate", "lost_or_stolen_card").

  Think of it like a labelled training set for a classifier:
    input  = "I still have not received my new card"
    label  = "card_arrival"

  In Kotlin/Android terms, imagine a RecyclerView of chat messages
  where each message already has a tag. We'll train a model to
  predict the tag for *new*, unseen messages.

WHY THIS DATASET?
  - It's real banking text — directly relevant to conversational AI
    at a financial company.
  - 77 classes is hard enough to be impressive, easy enough to train
    on a laptop.
  - Public and well-known, so interviewers recognise it.

HOW WE LOAD IT:
  Hugging Face `datasets` is like a Maven/Gradle dependency for ML
  datasets.  `load_dataset("PolyAI/banking77")` downloads it once
  and caches it locally (~/.cache/huggingface/), just like Gradle
  caches JARs in ~/.gradle/.
"""

from datasets import DatasetDict, load_dataset


def load_banking77() -> DatasetDict:
    """
    Download (or load from cache) the Banking77 dataset.

    Returns
    -------
    DatasetDict
        A dict-like object with two splits:
          ds["train"]  — 10,003 examples (the data the model learns from)
          ds["test"]   —  3,080 examples (held-out data we evaluate on)

        Each example has:
          "text"  — the customer message  (str)
          "label" — the intent id 0-76    (int)

    Kotlin analogy
    --------------
    This is like a sealed class with 77 subclasses, each with a
    numeric ordinal.  The ClassFeature on the dataset maps ordinals
    back to human-readable names.
    """
    ds = load_dataset("PolyAI/banking77")
    return ds


def label_names(ds: DatasetDict) -> list[str]:
    """
    Return the 77 human-readable intent names in ordinal order.

    Example: label_names(ds)[0] == "activate_my_card"

    Under the hood, Hugging Face stores a ClassLabel feature
    (like an enum in Kotlin) that maps int → string.
    """
    return ds["train"].features["label"].names


def describe(ds: DatasetDict) -> None:
    """Print a quick summary — handy for exploring in a notebook."""
    names = label_names(ds)
    print(f"Splits   : {list(ds.keys())}")
    print(f"Train    : {len(ds['train']):,} examples")
    print(f"Test     : {len(ds['test']):,} examples")
    print(f"Labels   : {len(names)} intents")
    print(f"First 5  : {names[:5]}")
    print("\nSample row:")
    row = ds["train"][0]
    print(f"  text  = {row['text']!r}")
    print(f"  label = {row['label']} → {names[row['label']]!r}")
