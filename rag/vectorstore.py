"""Vector store with a FAISS backend and a numpy fallback, plus save/load.

Same interface either way, so scaling from the demo (numpy) to production
(FAISS) needs no call-site changes.
"""
from __future__ import annotations

import json
import os

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.metadata: list[dict] = []
        self._faiss = None
        try:
            import faiss  # type: ignore
            self._faiss = faiss.IndexFlatIP(dim)
        except Exception:  # noqa: BLE001
            self._faiss = None

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata)
        if self._faiss is not None:
            self._faiss.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[dict]:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, min(k, len(self.metadata)))
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = (self.vectors @ q[0])
            order = np.argsort(sims)[::-1][:k]
            pairs = ((int(i), float(sims[i])) for i in order)
        return [{**self.metadata[i], "dense_score": s} for i, s in pairs if i >= 0]

    # ---- persistence ----
    def save(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "vectors.npy"), self.vectors)
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh)

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        vectors = np.load(os.path.join(folder, "vectors.npy"))
        store = cls(dim=vectors.shape[1])
        with open(os.path.join(folder, "metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        store.add(vectors, meta)
        return store
