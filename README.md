# Procurement RAG Assistant

A retrieval-augmented (RAG) assistant that answers plain-language questions over
procurement documents — vendor contracts, framework agreements, purchasing
policy — and grounds every answer in the retrieved text with a source citation.

This is the "document understanding" layer of an ERP-automation stack: turning
unstructured business documents into answerable, cited knowledge instead of
manual lookups.

## How it works

1. **Chunk** each document into ~120-word passages.
2. **Embed** them locally with `sentence-transformers` (all-MiniLM-L6-v2), so
   indexing runs offline.
3. **Retrieve** the top-k passages for a question via cosine similarity.
4. **Answer** with an LLM constrained to the retrieved context, citing the
   source file. If the answer isn't in context, it says so instead of guessing.

The index here is a tiny in-memory matrix — swap in **FAISS** or **Chroma** for
larger corpora without changing the interface.

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py "What is the payment term for Nordwind Components?"
```

Retrieval works without a key (you'll see the ranked passages); the key is only
needed for the final grounded answer.

## Example questions

- "What is the payment term for Nordwind Components?"
- "When do price deviations get flagged as an exception?"
- "What are the Incoterms for Suedbahn Logistik?"
