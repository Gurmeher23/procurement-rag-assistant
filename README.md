# Procurement RAG Assistant

A retrieval-augmented assistant over procurement documents (vendor contracts,
framework agreements, purchasing policy) that answers plain-language questions
grounded strictly in retrieved text, with source citations — plus an eval
harness to measure retrieval quality.

Built as a package with a **hybrid retriever** (dense embeddings fused with
sparse BM25) and an optional cross-encoder reranker, so exact-term matches
(payment-term numbers, vendor IDs) aren't lost by pure vector search.

## Architecture

```
rag/
├── config.py       # env-overridable config (weights, top-k, models)
├── loaders.py      # .txt / .md / .pdf loaders
├── chunking.py     # sentence-aware chunking with overlap
├── embeddings.py   # sentence-transformers + offline hashing fallback
├── vectorstore.py  # FAISS backend + numpy fallback, save/load
├── retriever.py    # BM25 (implemented here) + hybrid fusion + reranker
├── pipeline.py     # ingest → retrieve → grounded, cited answer
└── cli.py          # Typer CLI
eval/
├── questions.jsonl # labelled Q → expected source + keywords
└── run_eval.py     # source hit@k + keyword recall
```

## Retrieval

1. **Dense** — cosine similarity over embeddings (sentence-transformers, or a
   deterministic char-n-gram hashing fallback so it runs with zero downloads).
2. **Sparse** — BM25 over tokenized chunks.
3. **Fusion** — min-max normalize each score, combine with configurable weights.
4. **Rerank** *(optional)* — cross-encoder over the top candidates.

## Run

```bash
pip install -r requirements.txt

# Query (offline hashing embedder + BM25 works with no key/model):
python -m rag.cli "What are Nordwind's payment terms?" --docs data --top-k 3

# Grounded answer generation:
export OPENAI_API_KEY=sk-...

# Better embeddings / reranking:
#   installs sentence-transformers; set RAG_USE_RERANKER=1 to enable rerank

# Evaluate retrieval:
python -m eval.run_eval        # → source_hit@k, keyword_recall

# Tests:
pytest -q
```

On the bundled corpus the hybrid retriever scores **hit@k 1.0** and
**keyword recall 1.0** even with the offline embedder.

## Extend

- Point `loaders` at a folder of real PDFs (pypdf path is already wired).
- Swap the numpy store for `faiss-cpu` (drop-in) as the corpus grows.
- Add answer-faithfulness scoring to the eval harness.
