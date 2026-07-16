"""
Procurement RAG Assistant
-------------------------
A retrieval-augmented assistant that answers plain-language questions over a set
of procurement documents (vendor contracts, POs, delivery terms). It chunks the
documents, embeds them, retrieves the most relevant passages for a question, and
asks an LLM to answer grounded strictly in what was retrieved.

This mirrors the "document understanding" layer of an ERP-automation platform:
turning unstructured business documents into answerable, cited knowledge.

Run:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...          # needed for the LLM answer step
    python main.py "What is the payment term for Nordwind Components?"

Embeddings use a local sentence-transformers model, so indexing works offline;
only the final answer step calls an LLM.
"""
from __future__ import annotations

import glob
import os
import sys
import textwrap

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 120
TOP_K = 4


def load_documents(folder: str) -> list[dict]:
    docs = []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path) as fh:
            docs.append({"source": os.path.basename(path), "text": fh.read()})
    return docs


def chunk(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


class VectorIndex:
    """Tiny in-memory cosine-similarity index (swap for FAISS/Chroma at scale)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None

    def build(self, docs: list[dict]):
        for d in docs:
            for c in chunk(d["text"]):
                self.chunks.append({"source": d["source"], "text": c})
        vectors = self.model.encode([c["text"] for c in self.chunks],
                                    normalize_embeddings=True)
        self.matrix = np.asarray(vectors)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def answer(question: str, passages: list[dict]) -> str:
    from langchain_openai import ChatOpenAI

    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the source file in brackets. If the answer is not in the context, "
        "say you don't have that information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(prompt).content


def main():
    question = " ".join(sys.argv[1:]) or "What is the payment term for Nordwind Components?"
    here = os.path.dirname(__file__)

    index = VectorIndex()
    index.build(load_documents(os.path.join(here, "data")))
    passages = index.search(question)

    print("Question:", question)
    print("\nRetrieved passages:")
    for p in passages:
        print(f"  ({p['score']:.2f}) [{p['source']}] "
              + textwrap.shorten(p["text"], 90))

    if os.getenv("OPENAI_API_KEY"):
        print("\nAnswer:\n" + answer(question, passages))
    else:
        print("\n[Set OPENAI_API_KEY to generate a grounded answer from the "
              "retrieved passages above.]")


if __name__ == "__main__":
    main()
