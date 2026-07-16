"""Hybrid retrieval: dense (vector store) fused with sparse BM25, with an
optional cross-encoder reranker.

BM25 is implemented here (no external dependency) so lexical matches on exact
terms like payment-term numbers or vendor IDs are not lost by pure dense search.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .embeddings import Embedder
from .vectorstore import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d) for d in docs]
        self.doc_len = [len(d) for d in self.corpus]
        self.avg_len = sum(self.doc_len) / max(len(self.corpus), 1)
        self.freqs = [Counter(d) for d in self.corpus]
        df: Counter = Counter()
        for d in self.corpus:
            df.update(set(d))
        n = len(self.corpus)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.corpus), dtype=np.float32)
        for i, freqs in enumerate(self.freqs):
            dl = self.doc_len[i]
            s = 0.0
            for term in q:
                if term not in freqs:
                    continue
                f = freqs[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or float(x.max() - x.min()) == 0:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, chunks: list[dict],
                 dense_weight: float = 0.6, bm25_weight: float = 0.4):
        self.embedder = embedder
        self.store = store
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._reranker = None

    def enable_reranker(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder
        self._reranker = CrossEncoder(model_name)

    def retrieve(self, query: str, k: int, candidate_k: int = 12) -> list[dict]:
        # Dense similarity across the whole corpus.
        q_vec = self.embedder.encode([query])[0]
        dense = (self.store.vectors @ q_vec)
        sparse = self.bm25.scores(query)
        fused = self.dense_weight * _minmax(dense) + self.bm25_weight * _minmax(sparse)

        top = np.argsort(fused)[::-1][:candidate_k]
        candidates = [{**self.chunks[i], "score": float(fused[i]),
                       "dense": float(dense[i]), "bm25": float(sparse[i])}
                      for i in top]

        if self._reranker is not None:
            pairs = [(query, c["text"]) for c in candidates]
            for c, r in zip(candidates, self._reranker.predict(pairs)):
                c["rerank"] = float(r)
            candidates.sort(key=lambda c: c["rerank"], reverse=True)

        return candidates[:k]
