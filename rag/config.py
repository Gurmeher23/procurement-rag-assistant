"""RAG configuration (env-overridable via RAG_* variables)."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class RAGConfig:
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_words: int = int(os.getenv("RAG_CHUNK_WORDS", "120"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "30"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "12"))  # before rerank
    dense_weight: float = float(os.getenv("RAG_DENSE_WEIGHT", "0.6"))
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))
    use_reranker: bool = os.getenv("RAG_USE_RERANKER", "0") == "1"
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL",
                                    "cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    index_dir: str = os.getenv("RAG_INDEX_DIR", ".rag_index")
