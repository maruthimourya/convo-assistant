"""
answer.py - the GENERATION step: turn retrieved chunks into a written answer.

This closes the RAG loop:
    question --> [retrieve chunks] --> [stuff them into a prompt] --> [LLM writes answer]
                     search.py                 this file              llm.py (your providers)

KEY IDEA - "grounding":
  We do NOT ask the LLM to answer from its own memory. We hand it the retrieved
  policy text and say "answer ONLY from this." That's grounding (forcing the
  model to rely on supplied facts, not its training). It's how RAG reduces
  hallucination (a confident but made-up answer).

THE PROMPT SANDWICH (what we send the model):
  1. system  : the rules ("only use these excerpts, say 'I don't know' otherwise")
  2. context : the retrieved chunks, pasted in
  3. question: the user's actual question
  The model reads all three and writes an answer constrained by the middle part.

Kotlin view:
  val context = hits.joinToString("\n\n") { it.text }
  val reply = llm.reply(listOf(system(rules), user("$context\n\n$question")))
"""

from convo_assistant.llm import make_llm
from convo_assistant.retrieval.search import search

SYSTEM = (
    "You are a banking support assistant. Answer the question using ONLY the "
    "policy excerpts provided. If the excerpts do not contain the answer, say "
    "'I don't have a policy on that.' Do not use any outside knowledge. Keep the "
    "answer to two or three sentences."
)


def answer(question: str, k: int = 3) -> tuple[str, list[tuple[dict, float]]]:
    """Retrieve the top k chunks, then have the LLM answer from them.

    Returns (answer_text, hits) so the caller can show the answer AND cite which
    chunks it came from - important for a banking bot you have to trust.
    """
    hits = search(question, k=k)

    # Paste the retrieved chunks in, each labelled with its section (metadata).
    context = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c, _ in hits)

    history = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Policy excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]
    reply = make_llm().reply(history)
    return reply, hits
