"""End-to-end RAG pipeline: ingest → hybrid retrieve → grounded answer.

Answers are constrained to retrieved context and carry source citations; when
no LLM key is present the pipeline still returns ranked, cited passages so the
retrieval half is fully demoable offline.
"""
from __future__ import annotations

import os

from .chunking import chunk_corpus
from .config import RAGConfig
from .embeddings import build_embedder
from .retriever import HybridRetriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        self.cfg = config or RAGConfig()
        self.embedder = build_embedder(self.cfg.embed_model)
        self.store: VectorStore | None = None
        self.retriever: HybridRetriever | None = None

    def ingest(self, docs: list[dict]) -> "RAGPipeline":
        chunks = chunk_corpus(docs, self.cfg.chunk_words, self.cfg.chunk_overlap)
        vectors = self.embedder.encode([c["text"] for c in chunks])
        self.store = VectorStore(self.embedder.dim)
        self.store.add(vectors, chunks)
        self.retriever = HybridRetriever(
            self.embedder, self.store, chunks,
            self.cfg.dense_weight, self.cfg.bm25_weight)
        if self.cfg.use_reranker:
            try:
                self.retriever.enable_reranker(self.cfg.reranker_model)
            except Exception:  # noqa: BLE001
                pass
        return self

    def retrieve(self, question: str) -> list[dict]:
        assert self.retriever, "Call ingest() first."
        return self.retriever.retrieve(question, self.cfg.top_k, self.cfg.candidate_k)

    def answer(self, question: str) -> dict:
        passages = self.retrieve(question)
        citations = [{"source": p["source"], "score": round(p["score"], 3)}
                     for p in passages]
        if os.getenv("OPENAI_API_KEY"):
            text = _generate(question, passages, self.cfg.llm_model)
        else:
            text = ("[No OPENAI_API_KEY set — showing top retrieved passage.]\n"
                    + f"[{passages[0]['source']}] {passages[0]['text']}"
                    if passages else "No passages found.")
        return {"question": question, "answer": text, "citations": citations,
                "passages": passages}


def _generate(question: str, passages: list[dict], model: str) -> str:
    from langchain_openai import ChatOpenAI

    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
    prompt = (
        "You are a procurement assistant. Answer the question using ONLY the "
        "context. Cite the source file in square brackets after each claim. "
        "If the answer is not in the context, say you don't have that "
        f"information.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return ChatOpenAI(model=model, temperature=0).invoke(prompt).content
