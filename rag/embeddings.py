"""Embedding backends behind a common interface.

  * SentenceTransformerEmbedder — real dense embeddings (needs the model).
  * HashingEmbedder             — deterministic char-n-gram hashing vectors,
                                  zero downloads, so the pipeline + tests run
                                  fully offline. Lower quality but same shape.

`build_embedder()` prefers sentence-transformers and silently falls back.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Hashing vectorizer over 3-char shingles → fixed-dim L2-normalized vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _shingles(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text.lower())
        return [text[i:i + 3] for i in range(max(len(text) - 2, 1))]

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for sh in self._shingles(text):
                h = int(hashlib.md5(sh.encode()).hexdigest(), 16)
                mat[r, h % self.dim] += 1.0
        return _normalize(mat)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True),
                          dtype=np.float32)


def build_embedder(model_name: str) -> Embedder:
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:  # noqa: BLE001 - missing package or offline
        return HashingEmbedder()
