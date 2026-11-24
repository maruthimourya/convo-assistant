"""
chunking.py - cut a document into retrievable pieces ("chunks").

WHY (for a Kotlin developer):
  An embedding (a list of numbers that represents meaning) works best on a
  short, focused piece of text. A 40-page policy embedded as ONE vector is a
  blurry average of everything. So before indexing, we cut each document into
  small chunks and embed each chunk on its own.

THE TWO FAILURE MODES we design around:
  1. Split by heading only  -> some sections are huge, back to the blurry average.
  2. Split every N chars     -> a sentence (and its heading) can get cut in half.

OUR APPROACH (both, plus two safety nets):
  - Split by markdown heading first, so each piece is one topic.
  - If a section is longer than max_chars, cut it into overlapping windows.
  - overlap: repeat the last `overlap` characters so a sentence straddling a
    boundary still appears whole in at least one chunk.
  - Stamp the heading onto EVERY chunk as metadata (labels saved with a chunk),
    so even piece 2 of a long section still knows which section it came from.

Kotlin view of a chunk:
  data class Chunk(val text: String, val section: String, val source: String, val index: Int)
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str  # the piece of text to embed
    section: str  # the heading it came from (metadata)
    source: str  # the file it came from (metadata)
    index: int  # position of this chunk within the document


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs, one per '#' heading.

    Kotlin view: fold over the lines, starting a new section each time a line
    begins with '#'.
    """
    sections: list[tuple[str, str]] = []
    heading = "(no heading)"
    body: list[str] = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            if body:
                sections.append((heading, "\n".join(body).strip()))
                body = []
            heading = line.lstrip("#").strip()
        else:
            body.append(line)
    if body:
        sections.append((heading, "\n".join(body).strip()))
    # Drop empty bodies (e.g. a heading with no text under it).
    return [(h, b) for h, b in sections if b]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Cut one long string into overlapping windows of <= max_chars.

    Kotlin view: a sliding window that steps forward by (max_chars - overlap).
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap  # step back by `overlap` so windows share an edge
    return pieces


def chunk_markdown(
    markdown: str,
    source: str,
    max_chars: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Turn one markdown document into a list of Chunks.

    source   : the file name, stored on every chunk so we can cite it later.
    max_chars: longest a chunk may be before we cut it further.
    overlap  : characters shared between neighbouring windows of a long section.
    """
    chunks: list[Chunk] = []
    index = 0
    for heading, body in split_sections(markdown):
        for piece in _split_long(body, max_chars, overlap):
            chunks.append(
                Chunk(text=piece, section=heading, source=source, index=index)
            )
            index += 1
    return chunks
