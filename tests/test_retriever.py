"""Tests for chunking + hybrid retrieval. Run: pytest -q"""
from rag.chunking import chunk_corpus, split_sentences
from rag.config import RAGConfig
from rag.loaders import load_documents
from rag.pipeline import RAGPipeline
from rag.retriever import BM25


def test_split_sentences():
    assert split_sentences("A. B! C?") == ["A.", "B!", "C?"]


def test_chunking_has_overlap():
    doc = {"source": "d", "text": " ".join(f"w{i}." for i in range(200))}
    chunks = chunk_corpus([doc], chunk_words=40, overlap=10)
    assert len(chunks) > 1
    assert all("chunk_id" in c for c in chunks)


def test_bm25_ranks_exact_term():
    docs = ["net 30 payment terms", "delivery incoterms and lead time"]
    scores = BM25(docs).scores("payment terms net 30")
    assert scores[0] > scores[1]


def test_retrieval_finds_expected_source():
    pipe = RAGPipeline(RAGConfig(top_k=3)).ingest(load_documents("data"))
    passages = pipe.retrieve("What are Nordwind's payment terms?")
    assert any(p["source"] == "nordwind_contract.txt" for p in passages)
