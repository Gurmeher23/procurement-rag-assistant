"""CLI for the RAG assistant.

    python -m rag.cli query --docs data "What is Nordwind's payment term?"
"""
from __future__ import annotations

import textwrap

from .config import RAGConfig
from .loaders import load_documents
from .pipeline import RAGPipeline

try:
    import typer
except ImportError:  # pragma: no cover
    typer = None


def _run(question: str, docs: str, top_k: int, rerank: bool) -> None:
    cfg = RAGConfig(top_k=top_k, use_reranker=rerank)
    pipe = RAGPipeline(cfg).ingest(load_documents(docs))
    result = pipe.answer(question)

    print(f"Q: {question}\n")
    print("Retrieved:")
    for p in result["passages"]:
        tag = f"score={p['score']:.2f} dense={p['dense']:.2f} bm25={p['bm25']:.2f}"
        print(f"  [{p['source']}] ({tag})")
        print("    " + textwrap.shorten(p["text"], 110))
    print("\nAnswer:\n" + result["answer"])


if typer:
    app = typer.Typer(help="Procurement RAG assistant")

    @app.command()
    def query(question: str, docs: str = "data", top_k: int = 4, rerank: bool = False):
        _run(question, docs, top_k, rerank)

    def main():
        app()
else:  # pragma: no cover
    def main():
        raise SystemExit("pip install typer to use the CLI")


if __name__ == "__main__":
    main()
