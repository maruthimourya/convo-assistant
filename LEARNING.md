# Learning Map

The project is organized by concept to match my AI engineer plan. Each concept
has its library code under `src/convo_assistant/<concept>/` and its runnable
scripts under `scripts/<concept>/`.

## Week 3 — Intent classification

**Classifiers** (TF-IDF + logistic regression)
- Code: `src/convo_assistant/classifiers/baseline.py`
- Train: `uv run python scripts/classifiers/train_baseline.py`
- Compare methods: `uv run python scripts/classifiers/compare_methods.py`

**Transformers** (DistilBERT fine-tune)
- Code: `src/convo_assistant/transformers/transformer.py`
- Train: `uv run python scripts/transformers/train_transformer.py`

## Week 4 — Semantic search
- Code: `src/convo_assistant/semantic_search/semantic.py`  (classify_intent)
- Build embeddings: `uv run python scripts/semantic_search/encode_banking77.py`
- Try it: `uv run python scripts/semantic_search/semantic_search.py`
- Evaluate: `uv run python scripts/semantic_search/evaluate_semantic.py`
- Pick the confidence cutoff: `uv run python scripts/semantic_search/choose_cutoff.py`
- Retrieval metrics (recall@k, precision@k, MRR): `uv run python scripts/semantic_search/evaluate_retrieval.py`

## Week 5 — Retrieval & RAG

**Chunking** (cut documents into pieces)
- Code: `src/convo_assistant/chunking/chunking.py`
- Demo: `uv run python scripts/chunking/chunk_demo.py`
- Build + save chunks: `uv run python scripts/chunking/build_chunks.py`

**Retrieval + generation** (RAG)
- Search code: `src/convo_assistant/retrieval/search.py`
- Answer code: `src/convo_assistant/retrieval/answer.py`
- Build the index: `uv run python scripts/retrieval/encode_chunks.py`
- Search chunks: `uv run python scripts/retrieval/search_chunks.py`
- Full RAG demo: `uv run python scripts/retrieval/rag_demo.py`

## Shared helpers
- `src/convo_assistant/shared/dataset.py` — loads Banking77
- `src/convo_assistant/shared/evaluate.py` — accuracy / confusion matrix
- `src/convo_assistant/llm.py` — Groq / Ollama providers
- `src/convo_assistant/api.py` — FastAPI app

## Generated outputs
Each concept writes its artifacts under `outputs/<concept>/` (the `.npy` vectors
and `.txt` logs are gitignored; regenerate them by running the scripts):
- `outputs/classifiers/`     — confusion_matrix.png, compare_methods.txt
- `outputs/transformers/`    — confusion_matrix_transformer.png
- `outputs/semantic_search/` — banking77 vectors + labels, eval / cutoff / confidence logs
- `outputs/chunking/`        — chunks.jsonl
- `outputs/retrieval/`       — chunks_embeddings.npy
