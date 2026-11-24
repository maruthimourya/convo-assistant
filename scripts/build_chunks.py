"""build_chunks.py - chunk every policy doc and save to outputs/chunks.jsonl.

Each line is one chunk: {"text", "section", "source", "index"}.
JSONL (one JSON object per line) is easy to read back one chunk at a time.

Run:  PYTHONPATH=src python3 scripts/build_chunks.py
"""

import json
from dataclasses import asdict
from pathlib import Path

from convo_assistant.rag.chunking import chunk_markdown

POLICY_DIR = Path("data/policies")
OUT = Path("outputs/chunks.jsonl")


def main() -> None:
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        chunks += chunk_markdown(path.read_text(), path.name)  # default 500 / 50

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"Wrote {len(chunks)} chunks to {OUT}")


if __name__ == "__main__":
    main()
