"""Sentence-aware chunking with word-count targets and overlap.

Splits on sentence boundaries first, then packs sentences into chunks of about
`chunk_words` words, carrying `overlap` words of context between adjacent chunks
so answers that straddle a boundary stay retrievable.
"""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def chunk_document(doc: dict, chunk_words: int = 120, overlap: int = 30) -> list[dict]:
    sentences = split_sentences(doc["text"])
    chunks: list[dict] = []
    cur: list[str] = []
    cur_len = 0

    def flush():
        if cur:
            chunks.append({"source": doc["source"], "text": " ".join(cur)})

    for sent in sentences:
        words = sent.split()
        if cur_len + len(words) > chunk_words and cur:
            flush()
            # carry overlap words from the tail of the previous chunk
            tail = " ".join(cur).split()[-overlap:] if overlap else []
            cur = [" ".join(tail)] if tail else []
            cur_len = len(tail)
        cur.append(sent)
        cur_len += len(words)
    flush()
    return chunks


def chunk_corpus(docs: list[dict], chunk_words: int = 120, overlap: int = 30) -> list[dict]:
    out: list[dict] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_words, overlap))
    for i, c in enumerate(out):
        c["chunk_id"] = i
    return out
