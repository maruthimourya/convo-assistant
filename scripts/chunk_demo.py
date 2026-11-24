"""chunk_demo.py - show what the chunker produces. Reads data/policies/*.md."""

from pathlib import Path

from convo_assistant.rag.chunking import chunk_markdown

POLICY_DIR = Path("data/policies")


def run(max_chars: int, overlap: int) -> None:
    print(f"\n===== max_chars={max_chars}, overlap={overlap} =====")
    all_chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        all_chunks += chunk_markdown(path.read_text(), path.name, max_chars, overlap)
    print(f"Total chunks: {len(all_chunks)}")
    for c in all_chunks:
        print(f"\n[{c.source} #{c.index}] section={c.section!r} ({len(c.text)} chars)")
        print(f"  {c.text[:90]}...")


if __name__ == "__main__":
    run(max_chars=500, overlap=50)  # normal
    run(max_chars=200, overlap=40)  # forced split, to see overlap
